"""Deterministic scenario executor (PER-18).

Walks the steps of a single scenario in order, looking up each
referenced UI element on the current screen and executing the
specified action. Emits ``scenario.step_*`` events to the worker so
the UI can render a "M из N" progress indicator on /runs/{id}/results.

The scenario format (from backend's Scenario.steps_json["steps"]):
    {
      "screen_name": "Login",         # informational; not enforced
      "action": "tap" | "input" | "assert" | "swipe" | "back",
      "element_label": "Войти",       # how to find the element
      "value": "user@example.com",    # for input; supports {{test_data.X}}
      "expected_result": "..."        # informational; not enforced
    }

Failure handling: a step that can't find its element after retries
emits ``scenario.step_failed`` with the reason and the runner moves
on to the next step (instead of bombing the whole run). This matches
human tester behaviour — note the failure, keep going, don't lose
the entire session over one missing button.

When the scenario finishes (or every step has been attempted), the
caller (LLMExplorationLoop / MC loop) takes over for free
exploration.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger("explorer.scenario_runner")


# How many times to refresh elements + retry the lookup before giving
# up on a step. Apps need a beat to settle after navigation, so the
# first miss is normal — by the third we're confident the element
# isn't on this screen.
_LOOKUP_RETRIES = 3
_LOOKUP_DELAY_SEC = 1.0

# Settle time between actions — long enough for navigation animations
# but short enough that 5-step scenarios don't take 30 seconds. Same
# constant LLMExplorationLoop uses for its own settle (SETTLE_DELAY).
_ACTION_SETTLE_SEC = 1.2


def _substitute_test_data(text: str, test_data: dict[str, str]) -> str:
    """Replace ``{{test_data.KEY}}`` and ``{{KEY}}`` with values from
    ``test_data``. Same regex shape as llm_loop._substitute_test_data
    so the syntax is consistent across the worker. Unresolved keys
    are left as the literal placeholder — the caller logs them via
    the same path as the LLM-prompt substitution."""
    if not text or "{{" not in text:
        return text

    def _repl(match: "re.Match[str]") -> str:
        key = match.group(1).strip()
        return test_data.get(key, match.group(0))

    text = re.sub(r"\{\{\s*test_data\.(\w+)\s*\}\}", _repl, text)
    text = re.sub(r"\{\{\s*(\w+)\s*\}\}", _repl, text)
    return text


class ScenarioRunner:
    """Drives one scenario's steps to completion (or first hard error).

    ``controller`` must implement the AXe-controller interface used by
    the rest of the worker: ``tap_by_label``, ``tap_by_id``,
    ``set_text_in_field``, ``swipe``, ``go_back``, ``get_ui_elements``,
    ``take_screenshot``.

    ``event_callback`` is the same async sink LLMExplorationLoop uses;
    we emit ``{type: "scenario.step_*", scenario_id, step_idx, ...}``
    events so the backend can persist progress.
    """

    def __init__(
        self,
        controller,
        scenarios: list[dict[str, Any]],
        test_data: dict[str, str] | None = None,
        event_callback: Callable[[dict], Awaitable[None]] | None = None,
        # PER-37: optional RAG hookup. When all three are provided AND
        # a step has both ``expected_result`` and the scenario has
        # ``rag_document_ids``, the runner cross-checks the observed
        # post-action screen against the spec. Mismatch → step_failed
        # with reason="spec_mismatch" + a defect callback fires.
        rag_base_url: str | None = None,
        rag_token: str | None = None,
        defect_callback: Callable[[dict], Awaitable[None]] | None = None,
        run_id: str | None = None,
    ) -> None:
        self.controller = controller
        self.scenarios = scenarios or []
        self.test_data = test_data or {}
        self.event_callback = event_callback
        self.rag_base_url = (rag_base_url or "").rstrip("/") or None
        self.rag_token = rag_token or ""
        self.defect_callback = defect_callback
        self.run_id = run_id
        # Cache scenario_id → rag_document_ids so each step doesn't
        # re-walk the scenarios list.
        self._docs_by_scenario: dict[str, list[str]] = {
            str(s.get("id")): [str(d) for d in (s.get("rag_document_ids") or [])]
            for s in self.scenarios
        }

    async def run_all(self) -> dict[str, Any]:
        """Execute every scenario in order. Returns a summary dict so
        the caller can log "executed N steps across M scenarios"
        without rummaging through events."""
        completed = 0
        failed = 0
        for sc in self.scenarios:
            res = await self._run_one(sc)
            completed += res["completed"]
            failed += res["failed"]
        summary = {
            "scenarios": len(self.scenarios),
            "completed": completed,
            "failed": failed,
        }
        logger.info("[scenario] all scenarios done: %s", summary)
        return summary

    async def _run_one(self, scenario: dict[str, Any]) -> dict[str, int]:
        """PER-81 dispatcher. Picks the graph traversal when the
        scenario carries a v2 ``graph`` payload, otherwise falls back
        to the legacy linear walk over ``steps``."""
        sid = scenario.get("id", "?")
        title = scenario.get("title") or "(без названия)"

        graph = scenario.get("graph")
        if (
            isinstance(graph, dict)
            and isinstance(graph.get("nodes"), list)
            and isinstance(graph.get("edges"), list)
        ):
            return await self._run_graph(sid, title, graph)

        # Legacy linear path — kept around so an old worker config
        # without ``graph`` still works (and so freshly-flattened v2
        # scenarios from internal_runs.py keep running while we roll
        # out the graph traversal end-to-end).
        return await self._run_linear(sid, title, scenario.get("steps") or [])

    # ---------------------------------------------------------------- linear

    async def _run_linear(
        self, sid: str, title: str, steps: list[dict[str, Any]]
    ) -> dict[str, int]:
        await self._emit({
            "type": "scenario.started",
            "scenario_id": sid,
            "title": title,
            "total_steps": len(steps),
        })
        completed = 0
        failed = 0
        for idx, step in enumerate(steps):
            ok, reason = await self._run_step(sid, idx, step)
            if ok:
                completed += 1
            else:
                failed += 1
                logger.warning(
                    "[scenario] step %d/%d failed (%s) — continuing",
                    idx + 1, len(steps), reason,
                )
            # Settle between steps so the next lookup sees the new screen.
            await asyncio.sleep(_ACTION_SETTLE_SEC)
        await self._emit({
            "type": "scenario.finished",
            "scenario_id": sid,
            "completed_steps": completed,
            "failed_steps": failed,
            "total_steps": len(steps),
        })
        return {"completed": completed, "failed": failed}

    # ---------------------------------------------------------------- graph

    # Hard cap on total node visits per scenario, regardless of loops or
    # branching. A degenerate cycle with no exit would otherwise spin
    # the worker forever. PER-84 will let users tune per-loop iteration
    # limits; this top-level cap stays as a safety net.
    _MAX_NODE_VISITS = 200

    async def _run_graph(
        self, sid: str, title: str, graph: dict[str, Any]
    ) -> dict[str, int]:
        nodes_by_id: dict[str, dict[str, Any]] = {
            n["id"]: n
            for n in graph.get("nodes", [])
            if isinstance(n, dict) and isinstance(n.get("id"), str)
        }
        edges_from: dict[str, list[dict[str, Any]]] = {}
        for e in graph.get("edges", []):
            if isinstance(e, dict) and isinstance(e.get("source"), str):
                edges_from.setdefault(e["source"], []).append(e)

        action_count = sum(
            1 for n in nodes_by_id.values() if n.get("type") == "action"
        )
        await self._emit({
            "type": "scenario.started",
            "scenario_id": sid,
            "title": title,
            # ``total_steps`` stays the field name the UI already binds
            # to. For graphs it's the count of action nodes (upper
            # bound — branching may execute fewer).
            "total_steps": action_count,
        })

        # Find start node. We accept any node with type=start; if there
        # are several (the schema validator should have caught this on
        # save) we pick the first.
        start_id = next(
            (nid for nid, n in nodes_by_id.items() if n.get("type") == "start"),
            None,
        )
        if start_id is None:
            await self._emit({
                "type": "scenario.step_failed",
                "scenario_id": sid,
                "reason": "no_start_node",
            })
            await self._emit({
                "type": "scenario.finished",
                "scenario_id": sid,
                "completed_steps": 0,
                "failed_steps": 1,
                "total_steps": action_count,
            })
            return {"completed": 0, "failed": 1}

        completed = 0
        failed = 0
        # action_idx assigns a stable monotonic 0..N-1 to action node
        # executions so the existing UI (which still keys on step_idx)
        # keeps lining up timeline rows. node_id is also propagated so
        # PER-82's UI can highlight the live node.
        action_idx = 0
        # Per-node loop counter. PER-84 will read max_iterations from
        # the node payload; for now we just trip after the first repeat
        # via a loop-edge so misconfigured cycles stop early instead of
        # spinning all the way to _MAX_NODE_VISITS.
        loop_visits: dict[str, int] = {}

        current = start_id
        visits = 0
        while current is not None and visits < self._MAX_NODE_VISITS:
            visits += 1
            node = nodes_by_id.get(current)
            if node is None:
                await self._emit({
                    "type": "scenario.step_failed",
                    "scenario_id": sid,
                    "node_id": current,
                    "reason": "node_id_not_found",
                })
                failed += 1
                break

            ntype = node.get("type")
            if ntype == "end":
                break

            if ntype == "action":
                step = node.get("data") or {}
                ok, reason = await self._run_step(sid, action_idx, step, node_id=current)
                if ok:
                    completed += 1
                else:
                    failed += 1
                    logger.warning(
                        "[scenario] action %s failed (%s) — continuing",
                        current, reason,
                    )
                action_idx += 1
                await asyncio.sleep(_ACTION_SETTLE_SEC)
            elif ntype in ("start",):
                pass
            else:
                # decision / wait / screen_check / loop_back — full
                # semantics land in PER-83/84/85. For now just announce
                # we saw the node and keep walking the first edge.
                await self._emit({
                    "type": "scenario.node_skipped",
                    "scenario_id": sid,
                    "node_id": current,
                    "node_type": ntype,
                    "reason": "not_implemented_yet",
                })

            outgoing = edges_from.get(current, [])
            if not outgoing:
                # Nowhere to go and we're not at end → dangling.
                await self._emit({
                    "type": "scenario.step_failed",
                    "scenario_id": sid,
                    "node_id": current,
                    "reason": "dangling_node_no_outgoing_edge",
                })
                failed += 1
                break

            # PER-83 will replace this with condition-aware edge picking.
            picked = outgoing[0]
            next_id = picked.get("target") if isinstance(picked, dict) else None

            # Per-loop guard — fires when we cross a back-edge twice
            # before PER-84's max_iterations support arrives.
            if isinstance(picked, dict) and (picked.get("data") or {}).get("loop"):
                loop_visits[next_id or ""] = loop_visits.get(next_id or "", 0) + 1
                if loop_visits[next_id or ""] > 1:
                    await self._emit({
                        "type": "scenario.loop_exceeded",
                        "scenario_id": sid,
                        "node_id": next_id,
                        "reason": "loop_iteration_limit_pre_per84",
                    })
                    break

            current = next_id

        if visits >= self._MAX_NODE_VISITS:
            await self._emit({
                "type": "scenario.step_failed",
                "scenario_id": sid,
                "reason": f"max_node_visits ({self._MAX_NODE_VISITS}) exceeded",
            })
            failed += 1

        await self._emit({
            "type": "scenario.finished",
            "scenario_id": sid,
            "completed_steps": completed,
            "failed_steps": failed,
            "total_steps": action_count,
        })
        return {"completed": completed, "failed": failed}

    async def _run_step(
        self,
        scenario_id: str,
        step_idx: int,
        step: dict[str, Any],
        node_id: str | None = None,
    ) -> tuple[bool, str | None]:
        action = (step.get("action") or "tap").lower()
        label = step.get("element_label") or ""
        value = _substitute_test_data(step.get("value") or "", self.test_data)
        expected = (step.get("expected_result") or "").strip()
        started_evt: dict[str, Any] = {
            "type": "scenario.step_started",
            "scenario_id": scenario_id,
            "step_idx": step_idx,
            "action": action,
            "element_label": label,
        }
        if node_id is not None:
            # PER-81: the visual editor wants to highlight the live
            # node — propagate the id when we're walking a graph.
            started_evt["node_id"] = node_id
        await self._emit(started_evt)
        try:
            ok, reason = await self._dispatch(action, label, value)
        except Exception as exc:
            logger.exception("[scenario] step crashed")
            ok, reason = False, f"crash: {exc}"

        # PER-37: spec verification. Only runs when the action itself
        # succeeded — a tap on a missing button shouldn't ALSO get a
        # spec_mismatch defect on top. Requires expected_result on
        # the step + rag_document_ids on the scenario + RAG endpoint
        # configured. Failure here downgrades ok to False with a
        # specific reason and emits a P1 spec_mismatch defect.
        rag_verdict: dict | None = None
        if ok and expected and self.rag_base_url and self._docs_by_scenario.get(scenario_id):
            rag_verdict = await self._verify_expected_against_rag(
                expected=expected,
                step_label=label,
                step_value=value,
                document_ids=self._docs_by_scenario[scenario_id],
            )
            if rag_verdict is not None and not rag_verdict.get("matched", True):
                ok = False
                reason = "spec_mismatch"
                # Best-effort defect: don't kill the step if the
                # callback raises (e.g. backend down).
                try:
                    if self.defect_callback and self.run_id:
                        await self.defect_callback({
                            "run_id": self.run_id,
                            "step_idx": step_idx,
                            "screen_name": (step.get("screen_name") or "")[:500] or None,
                            "priority": "P1",
                            "kind": "spec_mismatch",
                            "title": f"Несоответствие спеке на шаге {step_idx + 1}",
                            "description": (
                                f"Ожидалось: «{expected}».\n"
                                f"Найденный фрагмент спеки: «{rag_verdict.get('snippet') or '—'}».\n"
                                f"Score: {rag_verdict.get('score', 0):.2f}."
                            ),
                            "llm_analysis_json": rag_verdict,
                        })
                except Exception:
                    logger.exception("[scenario] defect callback failed")

        evt: dict[str, Any] = {
            "type": "scenario.step_completed" if ok else "scenario.step_failed",
            "scenario_id": scenario_id,
            "step_idx": step_idx,
            "action": action,
            "element_label": label,
            "reason": reason,
        }
        if node_id is not None:
            evt["node_id"] = node_id
        if rag_verdict is not None:
            evt["rag_verdict"] = rag_verdict
        await self._emit(evt)
        return ok, reason

    async def _verify_expected_against_rag(
        self,
        *,
        expected: str,
        step_label: str,
        step_value: str,
        document_ids: list[str],
    ) -> dict | None:
        """Ask the workspace's RAG corpus whether ``expected`` matches
        the spec. Returns the structured verdict (same shape as PER-36
        Edge.rag_verdict_json) or None on transport failure.

        Threshold mirror PER-36: matched=True when distance < 0.7
        (i.e. score > 0.3). Tighter than the visual ✓/⚠/✗ split
        because spec verification is harsher — partial matches
        shouldn't pass."""
        import httpx
        # Compose a query that gives the embedder both the user
        # expectation and the action that produced it. The action
        # context narrows search to the right section of the spec.
        query = (
            f"Expected behaviour after action '{step_label or 'step'}'"
            + (f" with value '{step_value}'" if step_value else "")
            + f": {expected}"
        )
        payload = {"query": query, "top_k": 3, "document_ids": document_ids}
        try:
            # PER-XX: use the internal RAG endpoint (worker-token-protected)
            # rather than /api/admin/knowledge/query (admin-JWT only),
            # otherwise we get 401 and silently skip every spec check.
            # trust_env=False mirrors BackendClient — bypasses macOS
            # system proxy on corporate dev macs (PER-51).
            async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
                resp = await client.post(
                    f"{self.rag_base_url}/api/internal/runs/knowledge-query",
                    headers={"Authorization": f"Bearer {self.rag_token}"},
                    json=payload,
                )
                if resp.status_code != 200:
                    return None
                data = resp.json()
                matches = data.get("matches", [])
                if not matches:
                    return {
                        "matched": False, "score": 0.0,
                        "snippet": "", "document_id": None,
                        "document_title": None,
                    }
                top = matches[0]
                dist = float(top.get("distance", 1.0))
                return {
                    "matched": dist < 0.7,
                    "score": round(max(0.0, 1.0 - dist), 3),
                    "snippet": (top.get("text") or "")[:300],
                    "document_id": top.get("document_id"),
                    "document_title": top.get("document_title"),
                }
        except Exception as exc:
            logger.warning("[scenario] RAG verify failed: %s", exc)
            return None

    async def _dispatch(
        self, action: str, label: str, value: str
    ) -> tuple[bool, str | None]:
        """Map a scenario action onto a controller call."""
        if action == "back":
            ok = await self.controller.go_back()
            return ok, None if ok else "go_back returned False"

        if action == "swipe":
            # Generic vertical swipe up by default — scenarios that
            # need a specific direction should encode coordinates in
            # value as "x1,y1,x2,y2". Keeps the scenario format simple
            # for the common case.
            coords = self._parse_swipe(value)
            ok = await self.controller.swipe(*coords)
            return ok, None if ok else "swipe returned False"

        if action == "assert":
            # Soft check: present in current elements? Lookup uses
            # the same retry loop as actions so async screens settle.
            element = await self._find_element(label)
            return (element is not None,
                    None if element is not None else f"element {label!r} not found")

        # tap / input — both need to find the element first.
        element = await self._find_element(label)
        if element is None:
            return False, f"element {label!r} not found after {_LOOKUP_RETRIES} tries"

        if action == "input":
            test_id = element.get("test_id") or element.get("identifier")
            if test_id:
                ok = await self.controller.set_text_in_field(test_id, value)
                return ok, None if ok else "set_text_in_field returned False"
            # No test_id — fall back to tap-then-type (same as the LLM
            # loop does when the element doesn't expose an identifier).
            tapped = await self.controller.tap_by_label(label)
            if not tapped or not getattr(tapped, "ok", True):
                return False, "tap before type failed"
            typed = await self.controller.type_text(value)
            return typed, None if typed else "type_text returned False"

        # Default: tap.
        result = await self.controller.tap_by_label(label)
        ok = bool(result and getattr(result, "ok", True))
        return ok, None if ok else "tap_by_label returned not-ok"

    async def _find_element(self, label: str) -> dict | None:
        """Look the element up on the current screen by label, with
        retries to absorb settle-after-navigation timing."""
        if not label:
            return None
        for attempt in range(_LOOKUP_RETRIES):
            try:
                elements = await asyncio.wait_for(
                    self.controller.get_ui_elements(), timeout=10
                )
            except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                logger.warning(
                    "[scenario] get_ui_elements failed (attempt %d): %s",
                    attempt + 1, exc,
                )
                await asyncio.sleep(_LOOKUP_DELAY_SEC)
                continue
            for el in elements:
                if (el.get("label") or "").strip().lower() == label.strip().lower():
                    return el
                if el.get("test_id") and el.get("test_id") == label:
                    return el
            await asyncio.sleep(_LOOKUP_DELAY_SEC)
        return None

    @staticmethod
    def _parse_swipe(value: str) -> tuple[int, int, int, int]:
        """Parse "x1,y1,x2,y2" out of the step value, with a sensible
        default upward swipe in the middle of a typical phone."""
        parts = [p.strip() for p in (value or "").split(",")]
        if len(parts) == 4:
            try:
                return tuple(int(p) for p in parts)  # type: ignore[return-value]
            except ValueError:
                pass
        return (200, 600, 200, 200)  # default: swipe up

    async def _emit(self, event: dict) -> None:
        if self.event_callback is None:
            return
        try:
            await self.event_callback(event)
        except Exception:
            logger.exception("[scenario] event callback failed (event=%s)", event.get("type"))
