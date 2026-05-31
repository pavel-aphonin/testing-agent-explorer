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
import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from explorer.axe_client import (
    has_loading_indicator,
    screen_fingerprint,
    screen_fingerprint_structural,
)
from explorer.expression import ExprError, evaluate as eval_expr
from explorer.goal_schema import (
    _COORD_ONLY_ACTIONS,
    _ELEMENT_TARGETED_ACTIONS,
    _is_container_id,
    _is_editable_kind,
    build_actions_block,
    build_elements_block,
    build_goal_schema,
    build_test_data_block,
    normalize_decision,
    resolve_value,
)

logger = logging.getLogger("explorer.scenario_runner")

# PER-203 Phase 3: planner hints extracted to explorer.planning.hints so
# the bus planner-runner reuses the identical intelligence. Imported
# under the old private name to keep call sites unchanged.
from explorer.planning.hints import (  # noqa: E402
    append_pin_submit as _append_pin_submit,
    count_digit_taps as _count_digit_taps,
    credential_routing_hint as _credential_routing_hint_fn,
    loop_breaker_hint as _loop_breaker_hint,
    pin_keypad_hint as _pin_keypad_hint,
    pin_submit_hint as _pin_submit_hint,
)


# PER-111: prompt template fetched from backend (/api/internal/system-
# prompts). Cache for 60s — admin edits propagate within a minute,
# which is fast enough for hot-reload and slow enough that thousands
# of LLM-calls during a single run don't all hammer the backend.
_PROMPT_CACHE_TTL_SEC = 60.0
_prompt_cache: dict[str, tuple[float, str]] = {}


async def _load_prompt(code: str, backend_url: str, worker_token: str) -> str | None:
    """Fetch a prompt template by code, with 60-second TTL cache.

    Returns the ``content`` string from the backend's
    ``/api/internal/system-prompts/{code}`` endpoint, or None when the
    backend is unreachable / the slot was never seeded. Callers
    decide whether to fall back to a baked-in default — for goal-decide
    we keep one in ``_GOAL_DECIDE_SYSTEM_FALLBACK`` below.
    """
    cached = _prompt_cache.get(code)
    if cached is not None and time.monotonic() - cached[0] < _PROMPT_CACHE_TTL_SEC:
        return cached[1]
    url = f"{backend_url.rstrip('/')}/api/internal/system-prompts/{code}"
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            resp = await client.get(
                url, headers={"Authorization": f"Bearer {worker_token}"}
            )
            if resp.status_code != 200:
                return None
            content = resp.json().get("content")
            if isinstance(content, str) and content:
                _prompt_cache[code] = (time.monotonic(), content)
                return content
            return None
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Failed to load prompt %s: %s", code, exc)
        return None


# Fallbacks for when the backend has not been migrated yet (first boot
# after deploy) — match the migration seed text. Keep these in sync
# with alembic/versions/20260513_system_prompts.py.
# PER-111 v2: fallbacks the worker uses only when the backend's
# /api/internal/system-prompts endpoint is unreachable at boot. The
# canonical prompts live in the system_prompts table (migration
# 20260518_per111_v2); these are a degraded-mode safety net so the
# agent doesn't crash on a fresh install before migrations finish.
# Both fallback texts are *deliberately* minimal — they describe the
# new contract enough for the worker to keep working, but rely on the
# DB-stored prompt for the full ruleset.
_GOAL_DECIDE_SYSTEM_FALLBACK = (
    "Ты — автотестер мобильного приложения. Возвращай строго один "
    "JSON-объект по схеме (response_format уже включён). Поля: "
    "done, reason, action, action_args, element_id, element_label, "
    "value_source, value_literal, reasoning. action бери из списка "
    "доступных действий в user-сообщении. value_source: "
    "test_data.<ключ> если есть подходящие данные; goal_literal "
    "если значение указано буквально в цели; improvised если ни то "
    "ни другое; none если действие не требует ввода. element_id — "
    "идентификатор из списка элементов экрана."
)
_GOAL_DECIDE_USER_FALLBACK = (
    "Режим: {{mode}}\n"
    "Цель: {{goal}}\n"
    "Шаг: {{step_idx}} из {{max_steps}}\n"
    "Критерий успеха: {{success_criteria}}\n\n"
    "Текущий экран — элементы:\n{{elements_block}}\n\n"
    "Доступные действия:\n{{actions_block}}\n\n"
    "Доступные данные (test_data):\n{{test_data_block}}\n\n"
    "История действий:\n{{history_block}}"
)


def _render_template(template: str, values: dict[str, str]) -> str:
    """Substitute ``{{key}}`` / ``{{ key }}`` placeholders with values.

    Unknown placeholders stay verbatim — easier to spot a missing key
    in the prompt output than to silently drop content. We do NOT
    reuse ``_substitute_test_data`` because that one specifically
    walks the ``test_data.*`` namespace; here we want a generic
    "render this whole template" pass.
    """
    if not template or "{{" not in template:
        return template

    def _repl(match: "re.Match[str]") -> str:
        key = match.group(1).strip()
        v = values.get(key)
        return v if v is not None else match.group(0)

    return re.sub(r"\{\{\s*([\w.]+)\s*\}\}", _repl, template)


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
        # PER-86: scenarios referenced via sub_scenario nodes but not
        # executed as entry points. The runner mixes these into its
        # by-id lookup table for sub-call resolution but never runs
        # them at the top level.
        linked_scenarios: list[dict[str, Any]] | None = None,
        # PER-37: optional RAG hookup. When all three are provided AND
        # a step has both ``expected_result`` and the scenario has
        # ``rag_document_ids``, the runner cross-checks the observed
        # post-action screen against the spec. Mismatch → step_failed
        # with reason="spec_mismatch" + a defect callback fires.
        rag_base_url: str | None = None,
        rag_token: str | None = None,
        defect_callback: Callable[[dict], Awaitable[None]] | None = None,
        run_id: str | None = None,
        # PER-85: optional LLM client for semantic screen matching.
        # When provided AND a node carries ``screen_description``, the
        # runner asks the LLM whether the current screen matches the
        # description before executing the action. Without this client
        # ``screen_description`` is silently ignored — backwards-compat
        # with scenarios authored before the feature shipped.
        llm_client: Any | None = None,
        # PER-111 v2: workspace-enabled action dictionary, shipped by
        # backend in RunClaimResponse.actions. Each entry has code /
        # name / description / arguments_schema (JSON-schema for that
        # action's args). The runner feeds this to goal-node decode so
        # the LLM picks ``action`` from a real dictionary instead of a
        # hardcoded enum baked into the prompt. Empty list = degraded
        # mode (LLM gets no action constraints — workspace is broken,
        # operator needs to seed action types).
        actions: list[dict[str, Any]] | None = None,
        # PER-127: screen-stability tuning shipped by the backend per
        # workspace. ``settle_timeout_ms`` caps how long we wait for
        # the AXe tree to stop changing between actions;
        # ``settle_poll_ms`` is the polling cadence;
        # ``loading_indicator_keywords`` is the list of substrings
        # the worker scans element labels/values for to detect "still
        # loading" screens (e.g. "Секундочку, пожалуйста"). All
        # optional; defaults are conservative.
        settle_timeout_ms: int = 5000,
        settle_poll_ms: int = 500,
        loading_indicator_keywords: list[str] | None = None,
        # PER-131-lite: thinking-mode passport. Comes from the
        # backend's claim_next, sourced from the LLMModel row. When
        # ``supports_thinking`` is true the goal-decide call splits
        # into two passes (free-form reasoning, then constrained
        # JSON) so any model the operator plugs in — Gemma 4 today,
        # something else tomorrow — gets reasoning without code
        # changes. ``thinking_activation`` is the model-specific
        # token (Gemma: ``<|think|>``);
        # ``thinking_extract_regex`` peels the final answer out of
        # the reasoning-wrapped response (Gemma:
        # ``<channel\|>\n?(.*)``).
        supports_thinking: bool = False,
        thinking_activation: str | None = None,
        thinking_extract_regex: str | None = None,
        # PER-138: full capabilities. ``supports_json_schema`` controls
        # whether goal-decide sends ``response_format`` at all;
        # ``supports_multimodal_image`` decides whether to attach the
        # screenshot to chat calls. Defaults match the legacy
        # llama-server + Gemma 4 setup.
        supports_json_schema: bool = True,
        supports_multimodal_image: bool = True,
        # PER-145 L1: coordinate space the model emits for ``tap_at``.
        # ``points`` — raw screen points (Gemma family default).
        # ``normalized_1000`` — 0–1000 (Qwen2.5/3-VL convention).
        # ``pixels`` — raw pixels (Nemotron-style; scale down by
        # device retina factor).
        tap_at_coord_space: str = "points",
        # PER-163 retry: per-model screenshot resize ceiling. ``None``
        # = legacy logical-points behaviour (controller default).
        # Integer = pass to controller.take_screenshot(max_dim=N) so
        # vision-grounding models see near-native pixel detail.
        screenshot_max_dim: int | None = None,
        # PER-164: dedicated UI-grounder client. When the chat-LLM
        # emits ``tap_at`` with ``args.target_description`` (e.g.
        # "digit 8 on PIN keypad"), the dispatcher hands the
        # screenshot + description to this client instead of
        # trusting the chat-LLM's own (x, y) guess. ``None`` =
        # pre-PER-164 behaviour: always use the chat-LLM's coords.
        # The client itself silently falls back when no active
        # grounder is configured in the DB, so passing one here is
        # opt-in for the worker but transparent for the user.
        grounder: Any | None = None,
        # PER-164 followup: per-model sampling profile. Default
        # values match the LLMModel column defaults so legacy runs
        # without a passport keep the old behaviour (T=0.7 average).
        # NULL on top_k/min_p = omit from request (server picks).
        sampling_temperature: float = 0.7,
        sampling_top_p: float = 0.9,
        sampling_top_k: int | None = None,
        sampling_min_p: float | None = None,
        # PER-169: episodic memory layer (Graphiti + FalkorDB). When
        # set, every dispatched action is written into the agent's
        # knowledge graph (fire-and-forget) and each goal-decide
        # call queries the graph for relevant past actions to
        # inject as a memory_block into the prompt. None disables —
        # the worker falls back to history_block + visited hints
        # (pre-PER-169 behaviour). One EpisodicMemory instance is
        # shared across runs; group_id scoping inside the wrapper
        # isolates each run+goal.
        memory: Any | None = None,
    ) -> None:
        self.controller = controller
        self.scenarios = scenarios or []
        self.test_data = test_data or {}
        self.event_callback = event_callback
        self.rag_base_url = (rag_base_url or "").rstrip("/") or None
        self.rag_token = rag_token or ""
        self.defect_callback = defect_callback
        self.run_id = run_id
        self.llm_client = llm_client
        # PER-198: optional Context Identifier agent. Set externally by
        # the worker after construction (sr.context_agent = ...). When
        # present, each goal decision is preceded by a screen-type
        # classification; a PIN_entry verdict + >=4 digit taps in
        # history injects a hard «press submit now» rule into the
        # prompt — the behavioural fix the routing-only refactor
        # (PER-196) couldn't deliver. None → legacy behaviour.
        self.context_agent: Any | None = None
        # PER-204: last PIN-screen verdict from _context_pin_hint, read
        # by _goal_decide to stamp the decision so the submit-macro
        # fires in _run_goal_node. Defensive default before first classify.
        self._context_is_pin_last: bool = False
        # PER-200: Reflection agent — invoked when the loop-breaker
        # detects the agent is stuck, to produce a specific «try this
        # instead» recommendation. Set externally by the worker.
        self.reflection_agent: Any | None = None
        # PER-203 Phase 4: when TA_BUS_MODE=1, the goal loop publishes
        # screen.captured and awaits ground.produced from the bus
        # runners instead of calling _goal_decide + per-action grounding
        # synchronously. Sync path stays the default (flag off). The
        # BusClient is lazy-created on first use.
        import os as _os
        self._bus_mode: bool = _os.environ.get("TA_BUS_MODE", "0") == "1"
        self._bus = None  # type: ignore[assignment]
        self._actions: list[dict[str, Any]] = list(actions or [])
        # PER-127 settle config. Sane defaults if the backend ships
        # nothing (legacy workspace pre-migration, or test fixtures).
        self._settle_timeout_ms: int = max(0, int(settle_timeout_ms or 5000))
        self._settle_poll_ms: int = max(50, int(settle_poll_ms or 500))
        self._loading_keywords: list[str] = list(loading_indicator_keywords or [])
        # PER-131-lite thinking config.
        self._supports_thinking: bool = bool(supports_thinking)
        self._thinking_activation: str = thinking_activation or ""
        self._thinking_extract_regex: str = thinking_extract_regex or ""
        # PER-138 transport capabilities.
        self._supports_json_schema: bool = bool(supports_json_schema)
        self._supports_multimodal_image: bool = bool(supports_multimodal_image)
        # PER-145 L1 / PER-164 coord-space normalization for tap_at.
        # ``points``           — raw screen logical points (Gemma family).
        # ``normalized_1000``  — 0–1000 (Qwen2.5/3-VL convention).
        # ``pixels``           — raw native-device pixels (Nemotron-style;
        #                       worker scales down by retina factor).
        # ``image_pixels``     — PER-164: pixels of the IMAGE that was
        #                       sent to the model (UI-TARS family). The
        #                       image was resized to ``screenshot_max_dim``
        #                       before sending, so coords must scale back
        #                       to phone logical points by
        #                       ``raw * screen_logical / image_dim``,
        #                       NOT by retina factor.
        allowed_spaces = {"points", "normalized_1000", "pixels", "image_pixels"}
        space = (tap_at_coord_space or "points").lower()
        self._tap_at_coord_space: str = space if space in allowed_spaces else "points"
        # PER-163 retry: forward to controller.take_screenshot(max_dim=N).
        self._screenshot_max_dim: int | None = (
            int(screenshot_max_dim) if screenshot_max_dim else None
        )
        # PER-164: dedicated grounder client (None = pre-PER-164 behaviour).
        self._grounder = grounder
        # PER-164 followup: sampling profile threaded into every
        # _goal_decide LLM call.
        self._sampling_temperature: float = float(sampling_temperature)
        self._sampling_top_p: float = float(sampling_top_p)
        self._sampling_top_k: int | None = (
            int(sampling_top_k) if sampling_top_k is not None else None
        )
        self._sampling_min_p: float | None = (
            float(sampling_min_p) if sampling_min_p is not None else None
        )
        # PER-169: episodic memory layer.
        self._memory = memory

    async def _take_screenshot_bytes(self) -> tuple[bytes, int, int]:
        """PER-164: capture screenshot + return (raw_bytes, width_px, height_px).

        ``controller.take_screenshot`` returns base64-encoded PNG
        (the chat-LLM wants it that way for image_url inlining).
        The grounder wants raw bytes so it can do its own encoding,
        AND when the grounder's coord_space is ``image_pixels``
        the dispatcher needs the actual image dimensions to scale
        UI-TARS-style "pixel of the input image" coordinates back
        to phone logical points. We decode once here and probe
        PIL for the dims — keeps that knowledge out of the
        grounder client.
        """
        import base64 as _b64
        import io as _io
        from PIL import Image as _Image
        b64 = await self.controller.take_screenshot(
            max_dim=self._screenshot_max_dim,
        )
        if not isinstance(b64, str):
            raise RuntimeError("take_screenshot returned non-string payload")
        raw = _b64.b64decode(b64)
        with _Image.open(_io.BytesIO(raw)) as img:
            w, h = img.width, img.height
        return raw, w, h
        # PER-85: cache (screen-fingerprint, description) → (matches, reason)
        # so each unique description is only verified once per visited
        # screen. Reset per scenario in _run_graph.
        self._screen_match_cache: dict[tuple[str, str], tuple[bool, str]] = {}
        # Cache scenario_id → rag_document_ids so each step doesn't
        # re-walk the scenarios list.
        self._docs_by_scenario: dict[str, list[str]] = {
            str(s.get("id")): [str(d) for d in (s.get("rag_document_ids") or [])]
            for s in self.scenarios
        }
        # Same lookup for the full scenario payload, keyed by id, used
        # by sub_scenario nodes to find the linked graph at runtime.
        # Includes BOTH entry-point scenarios and the linked-only
        # library so sub-calls can resolve either way.
        self._scenarios_by_id: dict[str, dict[str, Any]] = {
            str(s.get("id")): s for s in self.scenarios if s.get("id")
        }
        for s in linked_scenarios or []:
            sid = str(s.get("id") or "")
            if sid and sid not in self._scenarios_by_id:
                self._scenarios_by_id[sid] = s
        # Sub-scenario call stack — protects against direct or
        # transitive recursion (A links to B, B links back to A).
        self._sub_call_stack: list[str] = []
        # PER-83: rolling context fed to edge-condition evaluator. Lives
        # at instance scope so a decision node can read state from the
        # action that ran upstream of it. Reset between scenarios in
        # ``_run_graph``.
        self._cond_ctx: dict[str, Any] = {
            "test_data": dict(self.test_data),
            "last_action_result": None,
            "last_screen": None,
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
        # PER-83: reset rolling context per scenario so a previous
        # scenario's last_screen doesn't leak into this one's branches.
        # Sub-scenario calls preserve the parent's last_action_result
        # / last_screen on the way IN by snapshotting them; this is the
        # default reset for top-level entry.
        self._cond_ctx = {
            "test_data": dict(self.test_data),
            "last_action_result": None,
            "last_screen": None,
        }
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
        # Per-node loop-edge visit counter. When traversal crosses an
        # edge with ``data.loop = True`` (or ``data.max_iterations``
        # set), we increment this map keyed by the edge's target and
        # cap the count using the resolved max_iterations (edge first,
        # node second — see the resolution block ~line 520). The
        # counter never spins to ``_MAX_NODE_VISITS`` for a properly
        # configured loop edge; that ceiling is the safety net for
        # cycles WITHOUT a loop-edge cap.
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
                # PER-83: feed the action's outcome into the rolling
                # context so a downstream decision can branch on it.
                self._cond_ctx["last_action_result"] = {
                    "ok": ok,
                    "reason": reason,
                    "action": (step.get("action") or "tap").lower(),
                    "element_label": step.get("element_label") or "",
                }
                action_idx += 1
                await asyncio.sleep(_ACTION_SETTLE_SEC)
            elif ntype in ("start", "decision", "loop_back", "group"):
                # Decision nodes don't *do* anything by themselves —
                # the branching happens on outgoing edges via
                # _pick_edge below. start, loop_back, and group are
                # no-ops at runtime (group is purely a UI affordance).
                pass
            elif ntype == "wait":
                # PER-85 (cheap): honor a ``ms`` field on wait nodes.
                ms = (node.get("data") or {}).get("ms")
                try:
                    seconds = max(0.0, float(ms or 0) / 1000.0)
                except (TypeError, ValueError):
                    seconds = 0.0
                if seconds > 0:
                    await asyncio.sleep(seconds)
            elif ntype == "screen_check":
                # PER-85: dedicated assertion node. Mandatory
                # ``screen_description`` — fail fast if missing rather
                # than silently passing the check.
                desc = ((node.get("data") or {}).get("screen_description") or "").strip()
                if not desc:
                    await self._emit({
                        "type": "scenario.step_failed",
                        "scenario_id": sid,
                        "node_id": current,
                        "reason": "screen_check_missing_description",
                    })
                    failed += 1
                    break
                ok, reason = await self._check_screen_with_retries(
                    desc, sid, current,
                )
                # Surface a step_completed/step_failed so the timeline
                # treats the check the same as an action would.
                self._cond_ctx["last_screen"] = {
                    "matches": ok,
                    "reason": reason,
                    "description": desc,
                }
                evt: dict[str, Any] = {
                    "type": "scenario.step_completed" if ok else "scenario.step_failed",
                    "scenario_id": sid,
                    "node_id": current,
                    "action": "screen_check",
                    "element_label": desc,
                    "reason": None if ok else f"screen_mismatch: {reason}",
                }
                await self._emit(evt)
                if ok:
                    completed += 1
                else:
                    failed += 1
                    # Match the runner's "keep going on soft failure"
                    # philosophy from action steps — the user can wire
                    # a decision node downstream if they want a hard stop.
            elif ntype == "goal":
                # PER-110: high-level NL instruction. The runner runs a
                # mini LLM-loop until the LLM declares the goal done
                # (or max_steps is exhausted). Behaves like a single
                # scenario step from the timeline's point of view —
                # one step_started, one step_completed/step_failed.
                step = node.get("data") or {}
                ok, reason = await self._run_goal_node(
                    sid, action_idx, step, node_id=current,
                )
                if ok:
                    completed += 1
                else:
                    failed += 1
                    logger.warning(
                        "[scenario] goal %s failed (%s) — continuing",
                        current, reason,
                    )
                self._cond_ctx["last_action_result"] = {
                    "ok": ok,
                    "reason": reason,
                    "action": "goal",
                    "element_label": (step.get("description") or "")[:80],
                }
                action_idx += 1
                await asyncio.sleep(_ACTION_SETTLE_SEC)
            elif ntype == "sub_scenario":
                target_id = ((node.get("data") or {}).get("linked_scenario_id") or "").strip()
                if not target_id:
                    await self._emit({
                        "type": "scenario.step_failed",
                        "scenario_id": sid,
                        "node_id": current,
                        "reason": "sub_scenario_link_missing",
                    })
                    failed += 1
                    break
                # Direct + transitive recursion guard. We don't want
                # A → B → A to spin until the global node-visit cap
                # kicks in — be loud about it instead.
                if target_id in self._sub_call_stack or target_id == sid:
                    await self._emit({
                        "type": "scenario.step_failed",
                        "scenario_id": sid,
                        "node_id": current,
                        "reason": f"sub_scenario_cycle: {' -> '.join(self._sub_call_stack + [sid, target_id])}",
                    })
                    failed += 1
                    break
                target = self._scenarios_by_id.get(target_id)
                if target is None:
                    await self._emit({
                        "type": "scenario.step_failed",
                        "scenario_id": sid,
                        "node_id": current,
                        "reason": f"sub_scenario_not_loaded: {target_id}",
                    })
                    failed += 1
                    break
                await self._emit({
                    "type": "scenario.sub_started",
                    "scenario_id": sid,
                    "node_id": current,
                    "sub_scenario_id": target_id,
                    "sub_scenario_title": target.get("title") or target_id,
                })
                # Run the linked scenario inline with full traversal.
                # _run_one will reset _cond_ctx; we snapshot the
                # parent's so condition data lives on after we return.
                ctx_snapshot = dict(self._cond_ctx)
                self._sub_call_stack.append(sid)
                try:
                    sub_summary = await self._run_one(target)
                finally:
                    self._sub_call_stack.pop()
                # Restore parent context + remember the sub's outcome
                # for downstream decisions to branch on.
                self._cond_ctx = ctx_snapshot
                self._cond_ctx["last_sub_scenario"] = {
                    "id": target_id,
                    "title": target.get("title"),
                    "completed": sub_summary.get("completed", 0),
                    "failed": sub_summary.get("failed", 0),
                }
                await self._emit({
                    "type": "scenario.sub_finished",
                    "scenario_id": sid,
                    "node_id": current,
                    "sub_scenario_id": target_id,
                    **sub_summary,
                })
                if sub_summary.get("failed", 0) > 0:
                    failed += 1
                else:
                    completed += 1
            else:
                # Unknown node type — should be impossible after
                # backend validation, but emit a diagnostic instead of
                # crashing the worker.
                await self._emit({
                    "type": "scenario.node_skipped",
                    "scenario_id": sid,
                    "node_id": current,
                    "node_type": ntype,
                    "reason": "unknown_node_type",
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

            # PER-83: condition-aware edge picking.
            #
            # An edge with an empty / missing condition is the
            # ``default`` branch — used when no other edge matches and
            # also as the only sensible pick on plain action → action
            # transitions where authoring conditions makes no sense.
            #
            # Walk outgoing in order; first edge with a truthy
            # condition wins. If every edge has a condition and none
            # matches, we fall back to the first edge without one. If
            # there is no default either → step_failed with the list
            # of conditions tried so the user can debug.
            picked = await self._pick_edge(outgoing, sid, current)
            if picked is None:
                # ``_pick_edge`` already emitted the diagnostic event.
                failed += 1
                break
            next_id = picked.get("target") if isinstance(picked, dict) else None

            # PER-84: per-loop iteration guard. Three places provide
            # the cap, in order of precedence:
            #   1. ``data.max_iterations`` on the back-edge itself
            #   2. ``data.max_iterations`` on the target loop_back node
            #   3. default 10
            # Crossing the loop edge ``max_iterations`` times in a row
            # without finding an exit emits a ``scenario.loop_exceeded``
            # event and stops the scenario.
            if isinstance(picked, dict) and (picked.get("data") or {}).get("loop"):
                edge_cap = (picked.get("data") or {}).get("max_iterations")
                target_node = nodes_by_id.get(next_id or "")
                node_cap = (
                    (target_node.get("data") or {}).get("max_iterations")
                    if target_node
                    else None
                )
                cap = (
                    edge_cap if isinstance(edge_cap, int) and edge_cap > 0
                    else node_cap if isinstance(node_cap, int) and node_cap > 0
                    else 10
                )
                loop_visits[next_id or ""] = loop_visits.get(next_id or "", 0) + 1
                if loop_visits[next_id or ""] > cap:
                    await self._emit({
                        "type": "scenario.loop_exceeded",
                        "scenario_id": sid,
                        "node_id": next_id,
                        "max_iterations": cap,
                        "reason": "loop_iteration_limit",
                    })
                    failed += 1
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

    async def _pick_edge(
        self,
        outgoing: list[dict[str, Any]],
        sid: str,
        from_node: str,
    ) -> dict[str, Any] | None:
        """Choose the next edge to traverse from a node with multiple
        out-edges (PER-83).

        Algorithm:
          1. Walk outgoing in array order; first edge with a non-empty
             condition that evaluates truthy wins.
          2. Failing that, fall back to the first edge with no
             condition at all (the "default" branch).
          3. If neither yields a winner, emit a diagnostic
             ``step_failed`` event with the list of conditions tried
             and return None — caller stops the scenario.
        """
        default_edge: dict[str, Any] | None = None
        tried: list[str] = []
        for edge in outgoing:
            if not isinstance(edge, dict):
                continue
            data = edge.get("data") or {}
            cond = (data.get("condition") or "").strip()
            if not cond:
                if default_edge is None:
                    default_edge = edge
                continue
            try:
                ok = eval_expr(cond, self._cond_ctx)
            except ExprError as exc:
                # Bad expressions don't kill the scenario — log,
                # remember as tried, and keep walking. If every other
                # edge also fails to match we'll fall through to the
                # default (or step_failed below).
                logger.warning(
                    "[scenario] bad condition on edge from %s: %r — %s",
                    from_node, cond, exc,
                )
                tried.append(f"{cond!r} (parse error: {exc})")
                continue
            tried.append(f"{cond!r} → {ok}")
            if ok:
                return edge
        if default_edge is not None:
            return default_edge
        # No condition matched and there's no default — emit a
        # diagnostic event so the user can see WHICH conditions ran
        # before the run summary.
        await self._emit({
            "type": "scenario.step_failed",
            "scenario_id": sid,
            "node_id": from_node,
            "reason": "no_branch_matched",
            "tried": tried,
        })
        return None

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

        # PER-85: optional pre-flight screen check. When the action
        # node has a ``screen_description``, ask the LLM to confirm
        # we're on the expected screen before we tap. Three retries
        # absorb async navigation; final mismatch fails the step
        # cleanly with the LLM's own reason in the log.
        screen_desc = (step.get("screen_description") or "").strip()
        if screen_desc:
            ok, reason = await self._check_screen_with_retries(
                screen_desc, scenario_id, node_id,
            )
            if not ok:
                fail_evt: dict[str, Any] = {
                    "type": "scenario.step_failed",
                    "scenario_id": scenario_id,
                    "step_idx": step_idx,
                    "action": action,
                    "element_label": label,
                    "reason": f"screen_mismatch: {reason}",
                }
                if node_id is not None:
                    fail_evt["node_id"] = node_id
                await self._emit(fail_evt)
                return False, f"screen_mismatch: {reason}"

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
                        # PER-120: defects now reference priority +
                        # severity rows by ``code``. Spec mismatches
                        # default to high / critical — the LLM
                        # confirmed the screen disagrees with the
                        # written specification, which is by definition
                        # a regression.
                        await self.defect_callback({
                            "run_id": self.run_id,
                            "step_idx": step_idx,
                            "screen_name": (step.get("screen_name") or "")[:500] or None,
                            "priority_code": "high",
                            "severity_code": "critical",
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
        # PER-106 #4: always include the run_id so the backend can
        # narrow to the run's workspace when ``document_ids`` is empty.
        # Without this fallback the worker would search every tenant's
        # corpus and match the wrong document.
        payload = {
            "query": query,
            "top_k": 3,
            "document_ids": document_ids,
            "run_id": self.run_id,
        }
        try:
            # Use the internal RAG endpoint (worker-token-protected)
            # rather than /api/admin/knowledge/query (admin-JWT only) —
            # otherwise we get 401 and silently skip every spec check.
            # trust_env=False mirrors BackendClient and bypasses macOS
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

    # ─────────────────────────────────────── PER-85: screen match

    # How many times to wait for the screen to settle before deciding
    # the LLM-rendered "no" is final. Async navigation animations and
    # loaders take a moment; without a retry the very first frame
    # after a tap loses the verdict.
    _SCREEN_MATCH_RETRIES = 3
    _SCREEN_MATCH_DELAY_SEC = 1.0

    async def _verify_screen(
        self, description: str
    ) -> tuple[bool, str]:
        """Ask the LLM whether the current screen matches the user's
        free-form ``description``. Returns (matches, reason).

        Caches per (fingerprint, description) so a multi-step scenario
        on the same screen pays the LLM cost only once. ``reason`` is a
        short sentence the editor can surface in run logs.
        """
        if not self.llm_client:
            # Without an LLM we can't reason about descriptions. Treat
            # as match so legacy scenarios that filled in
            # ``screen_description`` still run.
            return True, "no LLM configured"

        try:
            elements = await asyncio.wait_for(
                self.controller.get_ui_elements(), timeout=10
            )
        except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
            logger.warning("[scenario] get_ui_elements failed: %s", exc)
            return False, f"could not read screen ({exc})"

        # Cheap stable fingerprint of the visible screen state. Order
        # of elements typically reflects render order, which we keep.
        fingerprint = "|".join(
            f"{(getattr(el, 'kind', None) or '?')}:{(el.get('label') if isinstance(el, dict) else getattr(el, 'label', '')) or ''}"
            for el in elements[:50]
        )
        cached = self._screen_match_cache.get((fingerprint, description))
        if cached is not None:
            return cached

        prompt_lines = ["Текущий экран:"]
        for i, el in enumerate(elements[:25]):
            if isinstance(el, dict):
                label = el.get("label") or "(без подписи)"
                kind = el.get("kind") or "element"
            else:
                label = getattr(el, "label", "") or "(без подписи)"
                kind = getattr(el, "kind", "element")
                if hasattr(kind, "value"):
                    kind = kind.value
            prompt_lines.append(f"  {i + 1}. [{kind}] {label}")
        if len(elements) > 25:
            prompt_lines.append(f"  …и ещё {len(elements) - 25}.")
        prompt_lines.append("")
        prompt_lines.append(f'Ожидание пользователя: "{description}"')
        prompt_lines.append(
            'Это тот же экран? Ответь строго в формате: первая строка "yes" или "no", '
            "вторая строка — короткое объяснение (не больше одного предложения, любой язык)."
        )
        user_prompt = "\n".join(prompt_lines)
        system_prompt = (
            "Ты помогаешь автотесту проверять, что приложение находится на ожидаемом "
            "экране. Сравнивай по смыслу, а не по точному совпадению слов. Имена "
            "элементов могут быть на разных языках. Отвечай кратко."
        )

        try:
            resp = await self.llm_client.chat(
                system=system_prompt, user=user_prompt, max_tokens=80
            )
        except Exception as exc:
            logger.warning("[scenario] LLM screen-match failed: %s", exc)
            return False, f"LLM error: {exc}"

        if not resp:
            return False, "LLM returned no response"

        # Parse: first non-empty line is the verdict, rest is the reason.
        parts = [line.strip() for line in resp.strip().splitlines() if line.strip()]
        if not parts:
            return False, "empty LLM response"
        verdict_raw = parts[0].lower().rstrip(".:")
        matches = verdict_raw.startswith("yes") or verdict_raw.startswith("да")
        reason = " ".join(parts[1:])[:200] or parts[0][:200]
        result = (matches, reason)
        self._screen_match_cache[(fingerprint, description)] = result
        return result

    async def _quick_ax_snapshot(self) -> list[dict] | None:
        """One-shot AXe poll without settle / retries — used between
        batch items to cheaply detect that the screen drifted from the
        one the LLM planned the batch on.

        Returns the element list, or None on failure. Bounded at 3s
        so a slow AXe doesn't blow up batch dispatch latency more than
        a single tap would. PER-170 retry (Codex QA #2)."""
        try:
            return await asyncio.wait_for(
                self.controller.get_ui_elements(), timeout=3
            )
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            return None

    async def _wait_for_screen_stable(
        self,
    ) -> tuple[list[dict], bool]:
        """Poll AXe until two consecutive snapshots have the same
        fingerprint AND no loading-indicator keywords are visible.

        Returns ``(elements, stable)``:
          * ``elements`` — the last snapshot we read, even on timeout
            (so callers can proceed best-effort).
          * ``stable`` — True when convergence was actually reached
            within ``settle_timeout_ms``; False on timeout. The caller
            can pass this to the LLM as a hint ("экран ещё грузится").

        PER-127: instead of a flat ``asyncio.sleep`` between actions,
        the runner uses this to wait for the app to settle — typing
        animations, lazy navigation, network round-trips. If the
        workspace's ``loading_indicator_keywords`` are configured, an
        explicit spinner/«Секундочку, пожалуйста» on the screen
        defeats convergence even when fingerprints match (a frozen
        spinner is the *same* AXe tree across two polls).
        """
        timeout_ms = self._settle_timeout_ms
        poll_ms = self._settle_poll_ms
        if timeout_ms <= 0:
            # Feature disabled per workspace — just grab one frame and
            # return. Useful for super-fast sandboxes where any wait
            # is wasted.
            try:
                elements = await asyncio.wait_for(
                    self.controller.get_ui_elements(), timeout=10
                )
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                elements = []
            return elements, True

        last_fingerprint: str | None = None
        elements: list[dict] = []
        deadline = asyncio.get_event_loop().time() + (timeout_ms / 1000.0)
        # We need at least two polls separated by poll_ms to declare
        # stability; loop budget is timeout_ms / poll_ms iterations.
        iterations = max(2, int(timeout_ms / poll_ms))
        for _ in range(iterations):
            try:
                elements = await asyncio.wait_for(
                    self.controller.get_ui_elements(), timeout=10
                )
            except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                logger.warning("[settle] get_ui_elements failed: %s", exc)
                if asyncio.get_event_loop().time() >= deadline:
                    return elements, False
                await asyncio.sleep(poll_ms / 1000.0)
                continue
            fp = screen_fingerprint(elements)
            loading = has_loading_indicator(elements, self._loading_keywords)
            if last_fingerprint == fp and not loading:
                return elements, True
            last_fingerprint = fp
            if asyncio.get_event_loop().time() >= deadline:
                # Timed out — return whatever we have so the caller
                # can still act, but flag instability so it can
                # downgrade its confidence (e.g. nudge the LLM toward
                # ``wait``).
                return elements, False
            await asyncio.sleep(poll_ms / 1000.0)
        return elements, False

    async def _check_screen_with_retries(
        self, description: str, sid: str, node_id: str | None
    ) -> tuple[bool, str]:
        """Wrap ``_verify_screen`` with the retry policy: try N times
        with a delay so navigation animations and loaders get a chance
        to finish before we decide the screen is wrong."""
        last_reason = ""
        for attempt in range(self._SCREEN_MATCH_RETRIES):
            ok, reason = await self._verify_screen(description)
            last_reason = reason
            if ok:
                return True, reason
            if attempt < self._SCREEN_MATCH_RETRIES - 1:
                await asyncio.sleep(self._SCREEN_MATCH_DELAY_SEC)
        # Final mismatch — emit a diagnostic event so it shows in the
        # timeline alongside the failure.
        await self._emit({
            "type": "scenario.screen_mismatch",
            "scenario_id": sid,
            "node_id": node_id,
            "description": description,
            "reason": last_reason,
        })
        return False, last_reason

    # ─────────────────────────────────────── PER-110: goal node loop

    # Default cap on inner LLM-loop iterations for a single goal node.
    # ~15 actions is enough for "login + navigate + fill one form".
    # Authors can override per-node via data.max_steps.
    _GOAL_DEFAULT_MAX_STEPS = 15

    async def _run_goal_node(
        self,
        scenario_id: str,
        step_idx: int,
        data: dict[str, Any],
        node_id: str | None = None,
    ) -> tuple[bool, str | None]:
        """Drive a 'goal' node — the LLM picks actions until it
        declares the task done or max_steps is exhausted.

        PER-111 contract:
            * The system + user prompts come from ``system_prompts.*``
              rows in the backend (loaded once per node via
              ``_load_prompt`` with TTL cache). Admin edits propagate
              within a minute, no redeploy.
            * The LLM does NOT return a raw ``value`` to type. Instead
              it picks ``value_source`` from a constrained enum
              (``test_data.<key>`` / ``goal_literal`` / ``improvised``
              / ``none``). The worker — not the model — substitutes
              real values via :func:`goal_schema.resolve_value`. This
              physically prevents fabricated phones / emails / etc.
            * ``improvised_memory`` lives for the lifetime of this
              goal: when the model invents a value for a label, the
              next visit to the same label reuses the same string.

        Emits one ``scenario.step_started`` then a
        ``scenario.step_completed`` or ``scenario.step_failed`` so the
        timeline treats the goal as a single high-level step. Inner
        actions emit ``scenario.goal_action`` events for the UI's
        verbose log view.
        """
        description = _substitute_test_data(
            (data.get("description") or "").strip(), self.test_data
        )
        expected = (data.get("expected_outcome") or "").strip()
        try:
            max_steps = int(data.get("max_steps") or self._GOAL_DEFAULT_MAX_STEPS)
        except (TypeError, ValueError):
            max_steps = self._GOAL_DEFAULT_MAX_STEPS
        max_steps = max(1, min(max_steps, 50))

        # PER-111 v2: empty description = explore mode (free wandering).
        # The label shown in the timeline still needs *something*, so
        # mark it explicitly instead of an empty string.
        started_evt: dict[str, Any] = {
            "type": "scenario.step_started",
            "scenario_id": scenario_id,
            "step_idx": step_idx,
            "action": "goal",
            "element_label": (description or "(explore)")[:200],
        }
        if node_id is not None:
            started_evt["node_id"] = node_id
        await self._emit(started_evt)

        if not self.llm_client:
            # Goals require an LLM by definition — no fallback that
            # makes sense without one. Fail loudly so the user sees
            # the misconfiguration in the timeline rather than a
            # silent skip.
            await self._emit({
                "type": "scenario.step_failed",
                "scenario_id": scenario_id,
                "step_idx": step_idx,
                "node_id": node_id,
                "action": "goal",
                "reason": "llm_client_unavailable",
            })
            return False, "llm_client_unavailable"

        # Load prompt templates from the DB once per goal — they don't
        # change between inner steps. ``self.rag_base_url`` doubles as
        # the worker's "backend URL" because the same param is used
        # for the knowledge-query endpoint; if either bit of glue is
        # missing we degrade to the baked-in fallback constants above.
        system_prompt = _GOAL_DECIDE_SYSTEM_FALLBACK
        user_template = _GOAL_DECIDE_USER_FALLBACK
        if self.rag_base_url and self.rag_token:
            sys_from_db = await _load_prompt(
                "goal_decide.system", self.rag_base_url, self.rag_token,
            )
            usr_from_db = await _load_prompt(
                "goal_decide.user", self.rag_base_url, self.rag_token,
            )
            if sys_from_db:
                system_prompt = sys_from_db
            if usr_from_db:
                user_template = usr_from_db

        # PER-111 v2: action dictionary shipped with the claim (in
        # config["actions"]). Empty list = no workspace-side dictionary
        # — schema falls back to a permissive ``action: string``, but
        # the worker logs a warning so the operator notices.
        actions_dict: list[dict] = list(self._actions or [])
        if not actions_dict:
            logger.warning(
                "[goal] no actions in run config — schema will be permissive"
            )
        # Pre-build per-goal blocks that don't depend on the screen.
        # actions_block is identical across inner steps; elements_block
        # is rebuilt each step from the live AXe snapshot.
        actions_block = build_actions_block(actions_dict)
        # PER-145 L1: append a per-model coord-space hint for tap_at so
        # the model knows what space to emit (x, y) in. Worker scales
        # back to AXe screen points before the actual tap — see
        # ``_dispatch`` tap_at branch. Without this hint a normalized-
        # 1000 model (Qwen2.5/3-VL) and a points model (Gemma) would
        # both look correct in the prompt but produce wildly different
        # tap targets.
        tap_at_hint = self._tap_at_hint_for_prompt()
        if tap_at_hint:
            actions_block = f"{actions_block}\n\n{tap_at_hint}"
        test_data_block = build_test_data_block(self.test_data)
        success_criteria = expected or "—"
        # PER-111 v2: mode comes from whether the goal has a description.
        # No description → explore (free wandering with no completion
        # criterion).
        mode = "scenario" if description else "explore"
        goal_text = description or "—"

        history: list[str] = []
        # PER-111 v2: improvised values are memoized by element_id so
        # re-visits of the same field reproduce the same value.
        improvised_memory: dict[str, str] = {}
        last_reason: str | None = None
        done = False
        # PER-111 v2: anti-loop guard, two patterns.
        #
        # `recent_attempts` (last 3): catches the "repeat the same
        # failing call N times" pattern. First live run caught Gemma 4
        # repeating the same failing input 15 times despite the FAIL
        # marker in history — the model rationalised "let me try
        # again" and burned the whole max_steps budget on one stuck
        # field. P3 in the prompt is advisory; this is structural.
        #
        # `oscillation_window` (last 6): catches the A,B,A,B pattern
        # — second live run had the model bouncing between "Войти"
        # and "Back" until max_steps ran out. Identical-failures
        # check doesn't fire because the tuples aren't identical;
        # but if the last 6 steps contain only 2 unique tuples we're
        # in a 3-cycle and won't escape.
        from collections import deque
        recent_attempts: deque[tuple[str, str | None, str, bool]] = deque(maxlen=3)
        oscillation_window: deque[tuple[str, str | None]] = deque(maxlen=6)
        # PER-128: fingerprint of the screen we last saw, used to mark
        # in history whether each action actually changed anything.
        # ``None`` until the first iteration completes.
        last_pre_fingerprint: str | None = None
        # PER-161: algorithmic quick wins (Fastbot2 / AutoDroid / LLMDroid).
        #
        # ``visited_actions[structural_fp]`` — set of (action, element_id)
        # already executed (and succeeded) on this logical screen. We
        # surface it in the user-prompt history block so the model
        # doesn't keep retrying the same dead-end on the same screen.
        # Keyed by *structural* fingerprint (label-free) so a balance
        # card with a different ₽ number is still recognised as the
        # same screen.
        #
        # ``plateau_window`` — last N structural fingerprints. If the
        # window contains 0 fingerprints we haven't seen before, the
        # goal is making no progress in screen-coverage terms and we
        # cut it short with ``coverage_plateau`` rather than burning
        # the rest of max_steps.
        #
        # ``seen_fingerprints`` — running set of every structural
        # fingerprint visited during this goal. Counted against the
        # plateau window.
        #
        # ``backtrack_stack`` — chronological list of structural
        # fingerprints visited in this goal. When the current screen
        # is fully explored (visited_actions covers every interactive
        # element id on it), we synthesize a ``back`` so the agent
        # leaves rather than spamming the LLM for "one more action".
        visited_actions: dict[str, set[tuple[str, str | None]]] = {}
        seen_fingerprints: set[str] = set()
        plateau_window: deque[str] = deque(maxlen=10)
        backtrack_stack: list[str] = []

        # PER-170: per-screen batch — each ``_goal_decide`` LLM call
        # returns a list of actions to dispatch on the current screen,
        # not a single decision. ``inner_step`` counts *dispatched
        # actions*, not LLM calls — so a batch of 4 (e.g. PIN 8520)
        # bumps it by 4 while costing only one Gemma 4 round-trip.
        # The outer while replaces the old fixed-range for-loop so we
        # can advance the counter from the inner batch loop too.
        inner_step = 0
        # Signal from the inner per-item loop that the outer loop must
        # terminate (anti-loop break, max_steps hit, etc.). We can't
        # use plain ``break`` because it only exits the inner ``for``.
        outer_break = False
        while inner_step < max_steps:
            # 1. Read the current screen. Both representations matter:
            #   - AXe accessibility dump → element_id enum + the text
            #     description the LLM uses to ground its decision
            #   - screenshot PNG → vision input. PER-119 confirmed
            #     that without the rendered screen Gemma 4 can't tell
            #     a PIN-entry screen from a login screen when both
            #     have a button labelled "Войти". A 440×956 q4 vision
            #     encoder costs ~256 tokens — cheap at goal-node cadence.
            #
            # PER-127: replace the flat ``asyncio.sleep(_ACTION_SETTLE_SEC)``
            # with a screen-stability poll. The poll returns as soon as
            # two consecutive AXe snapshots match AND no loading
            # keywords are visible, or when settle_timeout_ms is hit
            # (whichever first). When the screen times out without
            # stabilising, we still proceed — but log a warning and
            # tell the LLM via history so it can choose `wait`.
            elements, stable = await self._wait_for_screen_stable()
            if not elements:
                last_reason = "get_ui_elements failed (empty after settle)"
                continue
            if not stable:
                logger.warning(
                    "[goal] screen did not stabilise within %d ms at step %d; "
                    "proceeding anyway",
                    self._settle_timeout_ms, inner_step,
                )
                # Hint the LLM that the screen is mid-transition so
                # it can pick `wait` rather than try to interact with
                # a half-rendered view. The history block is part of
                # every goal-decide prompt, so the model sees this
                # on the next turn.
                history.append(
                    "(система) экран ещё грузится — стоит дать ему время или "
                    "выбрать действие `wait`",
                )

            # PER-128: compare the current fingerprint against the
            # one we saw before the previous action. If they match,
            # the last tap/input did NOT change the screen — the LLM
            # would otherwise be tempted to repeat it. We rewrite
            # the most recent history entry to flag this so it shows
            # up in the next goal-decide prompt.
            current_fingerprint = screen_fingerprint(elements)
            if (
                last_pre_fingerprint is not None
                and current_fingerprint == last_pre_fingerprint
                and history
                and "[OK" in history[-1]
            ):
                history[-1] = history[-1].replace(
                    "[OK]", "[OK, но экран не изменился]"
                )
            last_pre_fingerprint = current_fingerprint

            # PER-161: structural fingerprint (label-free) for
            # visited_actions / plateau / backtrack. Labels can carry
            # rolling counters (balance, timestamps, message count)
            # that bump screen_fingerprint without us really being on
            # a new screen — structural ignores those.
            structural_fp = screen_fingerprint_structural(elements)
            is_new_screen = structural_fp not in seen_fingerprints
            seen_fingerprints.add(structural_fp)
            # plateau_window records True for "a screen we'd never
            # seen before in this goal", False for revisits. When the
            # window is full and contains no True → no new screens
            # for ``window.maxlen`` steps → coverage plateau.
            plateau_window.append(structural_fp if is_new_screen else None)
            if not backtrack_stack or backtrack_stack[-1] != structural_fp:
                backtrack_stack.append(structural_fp)

            # Plateau guard. Cheap, runs every step. The 10-step
            # window mirrors the LLMDroid paper's escalation
            # threshold; small enough that we don't waste many turns,
            # large enough that one slow LLM decision (where the
            # screen happens not to change) doesn't trip it.
            if (
                len(plateau_window) == plateau_window.maxlen
                and all(entry is None for entry in plateau_window)
            ):
                logger.warning(
                    "[goal] coverage plateau: 0 new screens in last %d "
                    "steps; aborting goal early",
                    plateau_window.maxlen,
                )
                last_reason = (
                    f"coverage_plateau: 0 new screens in last "
                    f"{plateau_window.maxlen} steps"
                )
                break

            screenshot_b64: str | None = None
            take_screenshot_fn = getattr(self.controller, "take_screenshot", None)
            # PER-138: skip the screenshot entirely when the model's
            # capability passport says it can't accept an image
            # content block. Saves a controller call + saves wasted
            # tokens on text-only models like Nemotron.
            if self._supports_multimodal_image and callable(take_screenshot_fn):
                try:
                    # PER-163 retry: pass max_dim when the model wants
                    # near-native pixel detail. Probe the controller's
                    # signature first so FakeController in tests (no
                    # max_dim kwarg) still works. Accept either an
                    # explicit ``max_dim`` parameter or a ``**kwargs``
                    # catch-all (the AXe controller has the explicit
                    # param; recording fakes use **kwargs).
                    shot_kwargs: dict[str, int] = {}
                    if self._screenshot_max_dim:
                        from inspect import signature as _sig, Parameter
                        try:
                            params = _sig(take_screenshot_fn).parameters
                            accepts_max_dim = (
                                "max_dim" in params
                                or any(
                                    p.kind == Parameter.VAR_KEYWORD
                                    for p in params.values()
                                )
                            )
                            if accepts_max_dim:
                                shot_kwargs["max_dim"] = self._screenshot_max_dim
                        except (TypeError, ValueError):
                            pass
                    screenshot_b64 = await asyncio.wait_for(
                        take_screenshot_fn(**shot_kwargs), timeout=10
                    )
                except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                    # Vision input is best-effort: a flaky screenshot
                    # shouldn't poison a goal step. Log it and fall
                    # through to text-only — the LLM still has the
                    # accessibility dump to work from.
                    logger.warning(
                        "[goal] take_screenshot failed (step %d): %s; "
                        "decision will be text-only",
                        inner_step, exc,
                    )
                    screenshot_b64 = None

            # PER-161: assemble a "уже пробовали на этом экране" hint
            # from visited_actions for the current screen. Empty when
            # no successful action has run on this screen yet (first
            # visit or all attempts failed).
            visited_summary = self._format_visited_actions(
                visited_actions.get(structural_fp, set())
            )

            # PER-169: pull episodic memory recall for this goal —
            # the agent gets explicit "you already did X" context so
            # it stops looping. Recall is time-bounded (3s); on
            # timeout / no memory configured we get "" and fall
            # through to the legacy history_block + visited hints.
            memory_block = ""
            if self._memory is not None:
                goal_scope_recall = node_id or f"step_{step_idx}"
                # Query phrased to surface action history relevant
                # to the current goal.
                recall_query = (
                    f"What actions have I already taken toward this goal: {goal_text}. "
                    f"What is the current progress?"
                )
                try:
                    memory_block = await self._memory.summary_for_prompt(
                        run_id=str(self.run_id or "no_run"),
                        goal_id=goal_scope_recall,
                        query=recall_query,
                    )
                except Exception:
                    logger.exception("[memory] recall failed (non-fatal)")
                    memory_block = ""

            # 2. Ask the LLM what to do (or whether we're done).
            # PER-203 Phase 4: in bus mode the plan+ground happens on the
            # message bus (context→planner→grounder runners), returning a
            # decision with pre-resolved coords. Sync path unchanged.
            _decide_kwargs = dict(
                goal_text=goal_text,
                mode=mode,
                success_criteria=success_criteria,
                elements=elements,
                history=history,
                step_idx=inner_step,
                max_steps=max_steps,
                system_prompt=system_prompt,
                user_template=user_template,
                actions=actions_dict,
                actions_block=actions_block,
                test_data_block=test_data_block,
                screenshot_b64=screenshot_b64,
                visited_summary=visited_summary,
                memory_block=memory_block,
            )
            if self._bus_mode:
                decision = await self._bus_goal_decide(**_decide_kwargs)
            else:
                decision = await self._goal_decide(**_decide_kwargs)
            if decision is None:
                last_reason = "llm_no_decision"
                break

            # PER-170: parse into the canonical batch shape — even if
            # the model emitted a legacy single-action object we get a
            # uniform ``{done, reason, expected_next_screen, actions}``
            # back, so the inner dispatch loop has one branch to walk.
            # PER-204: read the PIN verdict BEFORE normalize_decision
            # (which drops non-canonical keys), then apply the submit
            # macro. Bus path already grounded a submit on the bus →
            # the macro detects it and no-ops; sync path gets the submit
            # appended here and grounded inline at dispatch.
            _context_is_pin = bool(decision.get("context_is_pin"))
            decision = normalize_decision(decision)
            decision["actions"] = _append_pin_submit(
                decision.get("actions"), _context_is_pin
            )
            batch_actions = list(decision.get("actions") or [])

            # Log the batch envelope once per LLM call (cheap one line).
            # The per-item details are logged inside the inner loop with
            # the same "[goal] decision step=…" prefix as before so
            # downstream parsers / kibana dashboards keep working.
            logger.info(
                "[goal] batch decision: done=%s actions=%d reason=%r",
                decision.get("done"),
                len(batch_actions),
                (decision.get("reason") or "")[:120],
            )

            if decision.get("done") and not batch_actions:
                # "done with nothing to do" — terminal verdict with no
                # actions to dispatch. Honour it (except explore mode,
                # where the prompt forbids done=true and we just keep
                # exploring until max_steps).
                if mode == "explore":
                    logger.warning(
                        "[goal] model returned done=true in explore mode; "
                        "ignoring and continuing"
                    )
                    continue
                done = True
                last_reason = (decision.get("reason") or "goal reported as done")[:300]
                break

            # PER-170 inner loop: dispatch each action in the batch
            # without another LLM call. Between items there's no
            # _wait_for_screen_stable — that gates only the next
            # _goal_decide. If the screen drifts mid-batch (modal pops,
            # error appears), the *next* outer iteration will see the
            # new screenshot and adapt; the in-batch dispatcher just
            # plows ahead with what the LLM planned.
            # PER-205: map element_id → is-editable for THIS screen so
            # the batch validator can reject input / enter_text aimed at
            # a non-field (heading, label, button) — typing there is a
            # silent device no-op that makes the agent loop.
            _editable_by_id: dict[str, bool] = {}
            for _e in elements:
                if isinstance(_e, dict):
                    _eid = str(
                        _e.get("id") or _e.get("identifier")
                        or _e.get("test_id") or ""
                    )
                    if _eid:
                        _editable_by_id[_eid] = _is_editable_kind(
                            _e.get("kind") or _e.get("type")
                        )

            for batch_item_idx, action_item in enumerate(batch_actions):
                if inner_step >= max_steps:
                    outer_break = True
                    break

                # Diagnostic log: every dispatched action lands in
                # /tmp/ta-worker.log so we can debug "goal didn't progress"
                # without instrumenting the prompt itself. Cheap, one line
                # per inner step. The ``batch_item_idx/total`` suffix
                # makes it obvious whether the model is running a
                # multi-action plan or one-shotting per LLM call.
                logger.info(
                    "[goal] decision step=%d (batch %d/%d) done=%s action=%s args=%s "
                    "element_id=%s label=%r value_source=%s reasoning=%r",
                    inner_step,
                    batch_item_idx + 1,
                    len(batch_actions),
                    decision.get("done"),
                    action_item.get("action"),
                    action_item.get("action_args"),
                    action_item.get("element_id"),
                    action_item.get("element_label"),
                    action_item.get("value_source"),
                    (action_item.get("reasoning") or "")[:200],
                )

                action = (action_item.get("action") or "").strip().lower()
                label = (action_item.get("element_label") or "").strip() or None
                element_id = (action_item.get("element_id") or "").strip() or None
                action_args = action_item.get("action_args") or {}
                if not isinstance(action_args, dict):
                    action_args = {}
                # PER-111 v2: value resolved by the WORKER from value_source.
                value = resolve_value(action_item, self.test_data, improvised_memory)
                value_source = action_item.get("value_source") or ""

                # PER-170 retry (Codex QA #1+#3): validate the batch
                # item BEFORE dispatch — schema-side enforcement is
                # unreliable because llama-server's GBNF compiler
                # ignores per-branch oneOf constraints (we already saw
                # this for tap/element_id=null at the base level).
                # Without this guard, an action like
                # ``{"action":"tap","element_id":null}`` reached
                # _dispatch which called _find_element('') and either
                # returned a misleading FAIL or — worse — landed on the
                # app root container, corrupting visited_actions and
                # anti-loop. Now we skip the item, push a synthetic
                # «[SKIPPED: …]» history entry so the LLM sees it next
                # turn and adapts, and break the batch — continuing
                # would dispatch siblings that depended on this one
                # (e.g. a Forward tap whose precondition was a digit
                # tap that just got skipped).
                skip_reason: str | None = None
                if not action:
                    skip_reason = "action is empty"
                elif (
                    action in ("input", "enter_text")
                    and element_id
                    and element_id in _editable_by_id
                    and not _editable_by_id[element_id]
                ):
                    # PER-205: the model aimed text entry at a non-field
                    # element (e.g. a heading). Typing there does nothing
                    # on the device. Skip with explicit feedback so the
                    # next LLM turn taps a real field / keypad instead of
                    # silently looping on a no-op. element_id must be in
                    # the map (i.e. an element actually on this screen) —
                    # an unknown id falls through to other validators.
                    skip_reason = (
                        f"{action!r} targets element_id={element_id!r}, "
                        f"which is NOT a text field — typing there does "
                        f"nothing. Tap a text field first, or for an "
                        f"on-screen keypad use tap_at on the digit/letter "
                        f"buttons."
                    )
                elif action in _ELEMENT_TARGETED_ACTIONS and not element_id:
                    skip_reason = (
                        f"{action!r} requires element_id but model "
                        f"emitted element_id=null"
                    )
                elif (
                    action in _ELEMENT_TARGETED_ACTIONS
                    and element_id
                    and _is_container_id(element_id)
                ):
                    # PER-172: the LLM picked a root / app-shell
                    # container id when it really wanted a specific
                    # control that isn't in the AX-tree (the classic
                    # "PIN-screen Submit button is canvas-rendered"
                    # failure mode). Tapping a container does nothing
                    # visible, looks to the model like a no-op screen,
                    # and the next turn re-emits the same plan. Skip
                    # with a clear hint: the model gets «[SKIPPED:
                    # container_tap_forbidden]» in history and is
                    # expected to retry with tap_at + target_description.
                    skip_reason = (
                        f"{action!r} on container element_id={element_id!r} "
                        f"is forbidden — use tap_at with target_description "
                        f"to localise a non-AX button instead"
                    )
                elif action in _COORD_ONLY_ACTIONS:
                    # tap_at: x,y OR target_description. The
                    # arguments_schema (PER-164 followup) explicitly
                    # makes both optional and delegates the «at least
                    # one of (x,y)|target_description» check to the
                    # runtime — when a grounder is wired up, the
                    # model is encouraged to emit description only
                    # and let UI-TARS resolve the pixel. Reject only
                    # when BOTH are missing.
                    _ax = action_args.get("x")
                    _ay = action_args.get("y")
                    _desc = action_args.get("target_description")
                    has_xy = True
                    try:
                        int(_ax); int(_ay)
                    except (TypeError, ValueError):
                        has_xy = False
                    has_desc = isinstance(_desc, str) and _desc.strip() != ""
                    if not has_xy and not has_desc:
                        skip_reason = (
                            f"{action!r} requires either numeric x,y "
                            f"or target_description, got neither"
                        )
                if skip_reason is not None:
                    logger.warning(
                        "[batch] item %d/%d invalid — %s; skipping and "
                        "breaking batch so next LLM call sees the failure",
                        batch_item_idx + 1, len(batch_actions), skip_reason,
                    )
                    history.append(
                        f"{action or '?'} «{label or element_id or '?'}» "
                        f"[SKIPPED: {skip_reason}]"
                    )
                    last_reason = f"invalid_batch_item: {skip_reason}"
                    inner_step += 1
                    # Break batch — don't run dependent siblings against
                    # a now-uncertain screen.
                    break

                # Emit a granular event so the user's timeline can show
                # what the LLM is actually doing inside the goal. Not
                # persisted — broadcast-only via redis_bus.
                await self._emit({
                    "type": "scenario.goal_action",
                    "scenario_id": scenario_id,
                    "step_idx": step_idx,
                    "node_id": node_id,
                    "inner_step": inner_step,
                    "action": action,
                    "element_label": label,
                    "value_source": value_source,
                    "reasoning": (action_item.get("reasoning") or "")[:300],
                })

                # 3. Execute via the existing _dispatch contract.
                # PER-111 v2: pass element_id + action_args alongside
                # label. The dispatcher's id-first lookup matches the LLM's
                # stable identifier even when the visible label changes
                # (re-render, localization); action_args carry direction /
                # ms / duration_ms for the gesture- and timing-based
                # actions.
                try:
                    ok, reason = await self._dispatch(
                        action,
                        label or element_id or "",
                        value or "",
                        element_id=element_id,
                        action_args=action_args,
                    )
                except Exception as exc:
                    ok, reason = False, f"crash: {exc}"
                # PER-163: for coordinate-only actions (``tap_at``) the
                # element_id the LLM picks is the app root container —
                # carries no semantics. The real identity of the action
                # is the (x, y) it tapped. We bucket coordinates to a
                # 50pt grid so two visually-different taps don't get
                # collapsed by anti-loop, but micro-jitter (model
                # generating 670/675/680 for the same logical target)
                # still groups into one bucket. ``tap_id`` is what we
                # feed into visited_actions / oscillation_window /
                # recent_attempts in place of bare element_id.
                tap_id: str | None = element_id
                tap_coord_hint = ""
                if action == "tap_at":
                    _ax = action_args.get("x") if isinstance(action_args, dict) else None
                    _ay = action_args.get("y") if isinstance(action_args, dict) else None
                    try:
                        _bx = int(_ax) // 50 * 50
                        _by = int(_ay) // 50 * 50
                        tap_id = f"@({_bx},{_by})"
                        tap_coord_hint = f" ({int(_ax)},{int(_ay)})"
                    except (TypeError, ValueError):
                        pass

                history_line = (
                    f"{action} «{label or tap_id or element_id or '?'}»"
                    + tap_coord_hint
                    + (
                        # PER-143: enter_text also benefits from the
                        # resolved-value annotation. Keeps the LLM's history
                        # honest about WHICH source produced the typed
                        # string when value_source was used.
                        f" → {value} (via {value_source})"
                        if action in ("input", "enter_text") and value else ""
                    )
                    + (" [OK]" if ok else f" [FAIL: {reason}]")
                )
                history.append(history_line)
                last_reason = reason

                # PER-169: record the action into episodic memory so the
                # next decision step can recall "you already tapped digit
                # 8 here". Fire-and-forget — Graphiti's entity extraction
                # takes ~15s per episode on the local Qwen3-8B and we
                # must not block the agent on it.
                if self._memory is not None:
                    goal_scope = node_id or f"step_{step_idx}"
                    target_hint = ""
                    if isinstance(action_args, dict):
                        desc = action_args.get("target_description")
                        if isinstance(desc, str) and desc.strip():
                            target_hint = f" target='{desc[:80]}'"
                    # ``inner_step`` is the index within the current goal
                    # (0..max_steps-1) — what we want in the memory so the
                    # chat-LLM sees "Step 0, 1, 2…" as it walks PIN entry.
                    # ``step_idx`` is the goal's position in the parent
                    # scenario, which would always print "Step 0" for the
                    # first goal (PER-169 smoke #1 bug — all episodes were
                    # labelled Step 0 because each goal restarted the count).
                    episode_text = (
                        f"Step {inner_step} (goal {goal_scope}): "
                        f"{action} on '{label or element_id or '?'}'"
                        f"{tap_coord_hint}{target_hint}"
                        + (f" with value '{value}' (via {value_source})"
                           if action in ("input", "enter_text") and value else "")
                        + (" — OK" if ok else f" — FAIL: {reason}")
                    )
                    try:
                        await self._memory.add_action_fire_and_forget(
                            run_id=str(self.run_id or "no_run"),
                            goal_id=goal_scope,
                            episode_text=episode_text,
                            episode_name=f"step_{inner_step}_{action}",
                        )
                    except Exception:
                        logger.exception("[memory] add_action dispatch failed (non-fatal)")

                # PER-161: remember what we've successfully tried on this
                # screen so the next turn's prompt can warn the model
                # away from it. Only record successful actions — a failed
                # try might genuinely deserve a retry on a different
                # element (e.g. after the keyboard finally opens).
                #
                # PER-163: use ``tap_id`` (the bucketed coord) for tap_at
                # so distinct taps don't all hash to (tap_at, app_root).
                if ok:
                    visited_actions.setdefault(structural_fp, set()).add(
                        (action, tap_id)
                    )

                # Anti-loop check #1: if the same (action, element_id,
                # value_source) failed `recent_attempts.maxlen` times in
                # a row, the model isn't going to find a way out and the
                # remaining max_steps would just be more of the same.
                # PER-170: anti-loop breaks BOTH the inner batch loop
                # and the outer LLM loop — once we've decided the goal
                # is stuck, dispatching the rest of the batch would
                # only burn budget on the same failing pattern.
                recent_attempts.append((action, tap_id, value_source, ok))
                if (
                    len(recent_attempts) == recent_attempts.maxlen
                    and all(not entry[3] for entry in recent_attempts)
                    and len({entry[:3] for entry in recent_attempts}) == 1
                ):
                    logger.warning(
                        "[goal] anti-loop: %d identical failures of (%s, %s, %s); "
                        "aborting goal early",
                        recent_attempts.maxlen, action, element_id, value_source,
                    )
                    last_reason = (
                        f"stuck_loop: {recent_attempts.maxlen}× same failing "
                        f"({action}, element_id={element_id}, "
                        f"value_source={value_source})"
                    )
                    outer_break = True
                    break

                # Anti-loop check #2: oscillation pattern (A,B,A,B,…).
                # If the last 6 steps have only 2 unique (action, id)
                # pairs the model is stuck in a 3-cycle between two
                # actions and won't escape on its own. Mostly catches
                # "tap Войти ↔ tap Back" loops where the model thinks
                # each branch will help but the screen just toggles.
                # PER-163: bucket the coord into the tuple so two
                # tap_ats at meaningfully different points don't look
                # identical here either.
                oscillation_window.append((action, tap_id))
                if (
                    len(oscillation_window) == oscillation_window.maxlen
                    and len(set(oscillation_window)) <= 2
                ):
                    unique_pairs = ", ".join(
                        f"({a},{e})" for a, e in set(oscillation_window)
                    )
                    logger.warning(
                        "[goal] anti-loop: oscillation in last %d steps "
                        "(only 2 unique actions: %s); aborting goal early",
                        oscillation_window.maxlen, unique_pairs,
                    )
                    last_reason = (
                        f"stuck_loop: oscillation between {unique_pairs} "
                        f"over last {oscillation_window.maxlen} steps"
                    )
                    outer_break = True
                    break

                # PER-170: tick the inner-step counter after each
                # dispatched action — batch of N consumes N steps of
                # max_steps budget. Outer ``while`` checks the same
                # counter so we exit cleanly when budget is depleted.
                inner_step += 1
                await asyncio.sleep(0)

                # PER-170 retry (Codex QA #2): in-batch screen-change
                # check. If the structural fingerprint of the screen
                # drifted significantly from the one we read at the
                # start of the batch, the LLM's plan is operating on
                # stale assumptions — break and let the next outer
                # iteration grab a fresh screenshot + replan.
                #
                # We only check when there's a next item to dispatch
                # (last-item break is meaningless), and only for
                # actions that can plausibly change the screen
                # (tap / tap_at / input). Navigation actions like
                # back/swipe always change the screen — that's their
                # whole point — so for those we trust the model
                # planned the follow-up correctly.
                #
                # The check uses ``_quick_ax_snapshot`` (a single fast
                # AXe poll, no settle / no screenshot) and compares
                # the structural fingerprint that already lives at
                # the top of the outer loop. Skip on failure — the
                # check is a tripwire, not a hard requirement.
                if batch_item_idx + 1 < len(batch_actions) and action in (
                    "tap", "tap_at", "input", "enter_text"
                ):
                    try:
                        snap = await self._quick_ax_snapshot()
                    except Exception:
                        snap = None
                    if snap is not None:
                        post_fp = screen_fingerprint_structural(snap)
                        if post_fp != structural_fp:
                            logger.info(
                                "[batch] screen changed after item %d/%d "
                                "(fingerprint %s → %s); breaking batch for "
                                "fresh replan",
                                batch_item_idx + 1, len(batch_actions),
                                structural_fp[:12], post_fp[:12],
                            )
                            break
            # ── end for action_item in batch_actions ──

            # PER-170: propagate anti-loop / budget break from the
            # inner batch loop to the outer LLM loop.
            if outer_break:
                break

            # PER-170: a batch where every item completed without
            # tripping anti-loop, AND the model said done=true alongside,
            # is the normal "goal complete" exit. (Pure done=true with
            # no actions is handled before the inner loop above.)
            # Explore mode never honours done — the prompt forbids it
            # and the goal terminates on max_steps instead, mirroring
            # the pre-PER-170 behaviour exercised by T13.
            if decision.get("done") and mode != "explore":
                done = True
                last_reason = (decision.get("reason") or "goal reported as done")[:300]
                break
            elif decision.get("done") and mode == "explore":
                logger.warning(
                    "[goal] model returned done=true in explore mode; "
                    "ignoring and continuing"
                )

        # 4. Optional expected_outcome verification once the LLM
        # claims done. Mismatch downgrades the result AND opens a
        # defect (PER-129) — when the agent insists "done" but the
        # screen doesn't match the spec, that's a regression worth
        # surfacing, not just a silent failure.
        if done and expected:
            ok, why = await self._check_screen_with_retries(
                expected, scenario_id, node_id,
            )
            if not ok:
                done = False
                last_reason = f"expected_outcome_mismatch: {why}"
                try:
                    if self.defect_callback and self.run_id:
                        # Priority/severity stay at "high"/"major" by
                        # default — operator triages from there. Note
                        # we use *code* values per PER-120 contract,
                        # not the old P1/P2 strings.
                        await self.defect_callback({
                            "run_id": self.run_id,
                            "step_idx": step_idx,
                            "screen_name": (description or "")[:500] or None,
                            "priority_code": "high",
                            "severity_code": "major",
                            "kind": "spec_mismatch",
                            "title": (
                                f"Цель «{description[:60]}…» не достигнута"
                                if len(description) > 60
                                else f"Цель «{description}» не достигнута"
                            ),
                            "description": (
                                f"Ожидание: {expected}\n"
                                f"LLM-вердикт: {why}"
                            ),
                            "llm_analysis_json": {
                                "node_id": node_id,
                                "mode": mode,
                                "history_tail": history[-8:],
                            },
                        })
                except Exception:
                    logger.exception("[goal] defect callback failed")

        # PER-111 v2: explore-mode goals never set done=true by design
        # (the LLM is told not to, and the worker would override it
        # anyway). Exhausting max_steps in explore mode is the *designed*
        # exit, not a failure — report it as completed so the timeline
        # doesn't show a red mark on every free-exploration node.
        if mode == "explore" and not done:
            done = True
            last_reason = (
                last_reason
                or f"explore mode finished after {max_steps} steps"
            )

        evt: dict[str, Any] = {
            "type": "scenario.step_completed" if done else "scenario.step_failed",
            "scenario_id": scenario_id,
            "step_idx": step_idx,
            "action": "goal",
            "element_label": (description or "(explore)")[:200],
            "reason": last_reason or (None if done else "max_steps_exceeded"),
        }
        if node_id is not None:
            evt["node_id"] = node_id
        await self._emit(evt)
        return done, evt.get("reason")

    def _tap_at_hint_for_prompt(self) -> str:
        """PER-145 L1: per-model coord-space guidance for ``tap_at``.

        Different VLMs were trained on different coordinate spaces, and
        the worker scales their output into AXe screen points (which is
        what the device actually understands). We tell the model what
        space to emit so it doesn't fight its own training:

          * ``points``           — Gemma family. Output in raw screen
            points already (≤ device width × height).
          * ``normalized_1000``  — Qwen2.5/3-VL. The ``{"point_2d":
            [x, y]}`` convention with both axes in 0–1000.
          * ``pixels``           — Nemotron-style. Raw retina pixels;
            worker divides by device scale.

        Returns an empty string for default ``points`` (no extra
        instruction needed — the model just looks at the screen).
        """
        space = self._tap_at_coord_space
        screen_w = int(getattr(self.controller, "_width", 0) or 0)
        screen_h = int(getattr(self.controller, "_height", 0) or 0)
        if space == "normalized_1000":
            return (
                "ВАЖНО про tap_at: координаты (x, y) задавай в "
                "НОРМАЛИЗОВАННОМ пространстве 0–1000 по обеим осям "
                "(стандартный grounding-формат Qwen-VL). Worker сам "
                "пересчитает в реальные точки экрана. Пример: центр "
                "экрана = (500, 500); правый нижний угол = (1000, 1000)."
            )
        if space == "pixels":
            scale = float(getattr(self.controller, "_scale", 1.0) or 1.0)
            if screen_w > 0 and screen_h > 0:
                px_w = int(round(screen_w * scale))
                px_h = int(round(screen_h * scale))
                return (
                    f"ВАЖНО про tap_at: координаты (x, y) задавай в "
                    f"raw пикселях экрана: x ∈ [0, {px_w}], "
                    f"y ∈ [0, {px_h}]. Worker масштабирует в "
                    f"point-пространство контроллера."
                )
            return (
                "ВАЖНО про tap_at: координаты (x, y) задавай в raw "
                "пикселях экрана. Worker масштабирует обратно."
            )
        if space == "image_pixels":
            # PER-170 retry (Codex QA #5): the constructor accepts
            # ``image_pixels`` in ``allowed_spaces`` but until now the
            # prompt-hint method silently fell through to «emit in
            # points». A chat-LLM mis-configured (or new chat-LLM
            # whose author picked this space) would then receive a
            # contradictory hint vs the dispatcher's actual
            # interpretation. Wire it up: image_pixels means
            # "coordinates of the input image we sent you", and the
            # max image dim is owned by screenshot_max_dim (or the
            # raw screenshot when no resize). We don't have the
            # exact image dims at prompt-build time (they depend on
            # the screenshot of the moment), so we emit a generic
            # hint and trust the model to bound by what it sees.
            return (
                "ВАЖНО про tap_at: координаты (x, y) задавай в "
                "ПИКСЕЛЯХ ИЗОБРАЖЕНИЯ, которое тебе показано (как "
                "UI-TARS/Qwen-VL-grounder). Не пересчитывай в реальный "
                "экран — это сделает worker. Не превышай размер картинки."
            )
        # points — пишем явный диапазон, чтобы модель не отправляла
        # пиксели «на всякий случай».
        if screen_w > 0 and screen_h > 0:
            return (
                f"ВАЖНО про tap_at: координаты (x, y) задавай в "
                f"point-пространстве этого устройства: x ∈ "
                f"[0, {screen_w}], y ∈ [0, {screen_h}]. НЕ пиксели."
            )
        return ""

    async def _goal_think_pass(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        screenshot_b64: str | None,
    ) -> str:
        """First pass of the PER-131-lite two-step decode for
        thinking-capable models.

        Prepends ``self._thinking_activation`` to the system prompt
        to switch the model into reasoning mode, sends the prompt
        WITHOUT ``response_format`` (so the model can emit free
        text before its final answer), then extracts the final
        answer using ``self._thinking_extract_regex``. The returned
        string is fed back into the second (constrained-JSON) pass
        so the model can lean on its own reasoning when filling the
        schema.

        Returns "" if the model errors out, returns no useful
        thinking, or the extract regex is missing / fails to match.
        The caller treats an empty return as "fall back to
        single-pass".
        """
        import re as _re
        sysprompt_with_think = (
            f"{self._thinking_activation}\n{system_prompt}"
            if self._thinking_activation else system_prompt
        )
        try:
            raw = await self.llm_client.chat(
                system=sysprompt_with_think,
                user=user_prompt,
                # PER-144: raised 1500 → 4000. Nemotron-style reasoning
                # models on real goal-decide prompts (history + elements
                # + screenshot) need >1500 tokens just to fit the
                # <think> block; with the old cap the response hits
                # finish_reason=length before the model closes
                # </think> and content arrives empty. 4000 gives the
                # tail enough breathing room to also write a JSON
                # answer after the think block — measured on Nemotron
                # 3 Nano Omni 30B-A3B and Qwen-3.6 with sane budgets.
                # If you trim this back, also lower
                # ``json_max_tokens`` in _goal_decide for symmetry.
                max_tokens=4000,
                screenshot_b64=screenshot_b64,
            )
        except Exception as exc:
            logger.warning("[scenario] thinking pass failed: %s", exc)
            return ""
        if not raw:
            return ""
        if not self._thinking_extract_regex:
            # Nothing to extract — treat the whole response as the
            # model's thinking and pass it through.
            return raw.strip()
        try:
            m = _re.search(self._thinking_extract_regex, raw, _re.DOTALL)
        except _re.error as exc:
            logger.warning(
                "[scenario] thinking_extract_regex invalid (%s); "
                "passing through raw response",
                exc,
            )
            return raw.strip()
        if not m:
            # Pattern didn't match — model probably didn't emit the
            # expected wrapper. Better to forward the whole text than
            # to discard everything.
            return raw.strip()
        # Group 1 holds the final answer; some patterns might capture
        # the thinking part instead, but that's a passport bug we
        # surface upstream rather than mask here.
        return (m.group(1) or "").strip()

    @staticmethod
    def _format_visited_actions(
        visited: set[tuple[str, str | None]],
    ) -> str:
        """PER-161: render visited_actions for the current screen as
        a one-line «уже пробовали» note. Empty string when nothing
        has been tried yet — caller skips the section then.

        Sorted output keeps the prompt diff-stable run-to-run (cache
        hits in llama.cpp), avoiding spurious re-encodes when the set
        order shuffles.
        """
        if not visited:
            return ""
        parts: list[str] = []
        for action, element_id in sorted(visited):
            if element_id:
                parts.append(f"{action} «{element_id}»")
            else:
                parts.append(action)
        return ", ".join(parts)

    async def _context_pin_hint(
        self, elements_block: str, history: list[str] | None
    ) -> str | None:
        """PER-198: classify the screen and, on a PIN/secret-code screen
        with >=4 digit taps already in history, return a hard rule for
        the Planner. None when no context agent, classification fails,
        screen isn't PIN, or fewer than 4 digits entered.

        Counting heuristic: a «digit tap» is any history entry that
        mentions tap/tap_at plus a single 0-9 token (the keypad presses
        our own logs render as «tap_at … цифра 8 …» / «input … pin»).
        Deliberately loose — over-counting only makes the submit hint
        fire one step earlier, which is harmless; under-counting is the
        failure we're fixing.
        """
        if self.context_agent is None:
            self._context_is_pin_last = False
            return None
        try:
            result = await self.context_agent.classify(elements_block)
        except Exception as exc:  # never break the decision loop
            logger.debug("context classify failed: %s", exc)
            self._context_is_pin_last = False
            return None
        # PER-204: cache the PIN verdict so _goal_decide can stamp the
        # decision and the submit-macro fires in _run_goal_node.
        self._context_is_pin_last = bool(result and result.is_pin_entry)
        if result is None or not result.is_pin_entry:
            return None

        # PER-203 Phase 3: counting + hint text now live in the shared
        # planning.hints module (reused by the bus planner-runner).
        digit_taps = _count_digit_taps(history)
        logger.info(
            "[PER-198 context] screen=%s conf=%.2f digit_taps=%d",
            result.label, result.confidence, digit_taps,
        )
        # >=4 digits → submit; else → keypad strategy (tap digit buttons,
        # never enter_text on a canvas keypad).
        return _pin_submit_hint(digit_taps) or _pin_keypad_hint()

    def _credential_routing_hint(self) -> str | None:
        """PER-200: tell the Planner which test_data key maps to which
        code-entry screen, so it stops grabbing the wrong credential.

        The PER-198 smoke showed the model entering ``sms_code`` (0000)
        on a PIN screen instead of ``pin_code`` (8520) — the goal text
        crams four similar credentials together and the 4B model can't
        disambiguate. This static routing table is injected on every
        goal decision (cheap, no LLM) and only references keys that
        actually exist in test_data, with values MASKED (we never put
        real secrets in the prompt beyond what value_source already
        does).
        """
        # PER-203 Phase 3: delegated to shared planning.hints.
        return _credential_routing_hint_fn(set(self.test_data.keys()))

    async def _bus_goal_decide(
        self,
        *,
        goal_text: str,
        mode: str,
        success_criteria: str,
        elements: list,
        history: list[str],
        step_idx: int,
        max_steps: int,
        system_prompt: str,
        user_template: str,
        actions: list[dict],
        actions_block: str,
        test_data_block: str,
        screenshot_b64: str | None = None,
        visited_summary: str = "",
        memory_block: str = "",
    ) -> dict[str, Any] | None:
        """PER-203 Phase 4: bus-mode replacement for ``_goal_decide``.

        Publishes ``screen.captured`` with the full screen state and
        awaits ``ground.produced`` from the runner chain (context →
        planner → grounder). Returns the canonical
        ``{done, reason, expected_next_screen, actions}`` decision with
        the grounder's coords merged into each action's ``action_args``
        (x, y) — so the EXISTING inner dispatch loop taps pre-resolved
        coordinates and runs the whole batch atomically, no per-action
        grounding, no mid-batch re-planning. None on timeout.
        """
        from explorer.bus import BusClient, Envelope, MsgType

        if self._bus is None:
            self._bus = BusClient(consumer_name=f"worker-{self.run_id}")
            await self._bus.connect()
            # PER-175 full chain: the worker awaits the TERMINAL stage,
            # ground.verified (Grounding Verifier), not ground.produced —
            # so the whole 13-module pipeline runs before the worker acts.
            await self._bus.ensure_group(MsgType.GROUND_VERIFIED, "g.worker")

        payload = {
            "goal_text": goal_text,
            "mode": mode,
            "success_criteria": success_criteria,
            "elements": elements,
            "elements_block": build_elements_block(elements)[0],
            "history": history,
            "step_idx": step_idx,
            "max_steps": max_steps,
            "system_prompt": system_prompt,
            "user_template": user_template,
            "actions": actions,
            "actions_block": actions_block,
            "test_data_block": test_data_block,
            "test_data_keys": list(self.test_data.keys()),
            "visited_summary": visited_summary,
            "memory_block": memory_block,
            "screenshot_b64": screenshot_b64,
            # PER-175 Phase C: screen pixel dims so the Context Identifier
            # runner can scale OmniParser's normalized bboxes into the
            # affordance map's pixel space. Secrets (test_data values) are
            # deliberately NOT put on the bus — they stay in the worker.
            "screen_w": int(getattr(self.controller, "_width", 0) or 0),
            "screen_h": int(getattr(self.controller, "_height", 0) or 0),
        }
        await self._bus.publish(Envelope(
            run_id=str(self.run_id), step_id=step_idx,
            type=MsgType.SCREEN_CAPTURED, payload=payload,
        ))
        logger.info("[bus] published screen.captured step=%d — awaiting ground.verified", step_idx)

        import time as _time
        deadline = _time.time() + 180.0
        while _time.time() < deadline:
            got = await self._bus.consume(MsgType.GROUND_VERIFIED, "g.worker", count=5, block_ms=5000)
            for entry_id, env in got:
                await self._bus.ack(MsgType.GROUND_VERIFIED, "g.worker", entry_id)
                if env.run_id != str(self.run_id) or env.step_id != step_idx:
                    continue
                ga = env.payload.get("grounded_actions") or env.payload.get("actions") or []
                # Merge grounder coords into action_args.x/y so the
                # existing tap_at dispatch uses them directly.
                for a in ga:
                    coords = a.get("coords")
                    if coords and len(coords) == 2:
                        aa = a.get("action_args") if isinstance(a.get("action_args"), dict) else {}
                        aa["x"], aa["y"] = coords[0], coords[1]
                        a["action_args"] = aa
                logger.info("[bus] ground.verified step=%d actions=%d", step_idx, len(ga))
                # PER-175 Phase C: deterministic keypad macro. The vision
                # Context Identifier put an affordance_map on the bus; if it
                # says this is a PIN screen and we hold the code, drive the
                # canvas keypad ourselves (tap digits in order + submit)
                # instead of trusting the planner's mechanism choice — the
                # exact fix for the cccc3333 canvas no-op loop. Secret is
                # read here worker-side; never rode the bus.
                amap_dict = env.payload.get("affordance_map") or {}
                from explorer.affordances import AffordanceMap
                from explorer.platform_adapter import (
                    keypad_macro, should_fire_keypad_macro,
                )
                amap_obj = AffordanceMap.from_dict(amap_dict)
                # PER-175 Phase C SAFETY (smoke dddd4444): fire ONLY on
                # corroborated evidence — a real keypad grid (OmniParser
                # >=6 digit keys) or a high-confidence PIN screen_type. The
                # SigLIP label alone (uniform ~0.09) fired the macro on the
                # LOGIN screen, nearly typing the PIN there. Otherwise fall
                # through to the planner — never type a secret on an
                # unverified screen.
                macro = None
                if should_fire_keypad_macro(amap_obj):
                    macro = keypad_macro(self.test_data)
                if macro:
                    logger.info(
                        "[bus] keypad macro fired (screen_type=%s conf=%.2f "
                        "keypad_keys=%d) → %d actions (digit taps + submit)",
                        amap_obj.screen_type, amap_obj.screen_confidence,
                        len(amap_obj.keypad_keys), len(macro),
                    )
                    return {
                        "done": False,
                        "reason": env.payload.get("reason"),
                        "expected_next_screen": env.payload.get("expected_next_screen"),
                        "actions": macro,
                        "context_is_pin": True,
                    }
                return {
                    "done": bool(env.payload.get("done", False)),
                    "reason": env.payload.get("reason"),
                    "expected_next_screen": env.payload.get("expected_next_screen"),
                    "actions": ga,
                    # PER-204: carry the context runner's PIN verdict so
                    # the submit-macro in _run_goal_node knows the screen
                    # type (the bus runner already appended+grounded a
                    # submit, so this is the idempotent safety net).
                    "context_is_pin": bool(env.payload.get("context_is_pin")),
                }
        logger.warning("[bus] timeout awaiting ground.produced step=%d", step_idx)
        return None

    async def _goal_decide(
        self,
        *,
        goal_text: str,
        mode: str,
        success_criteria: str,
        elements: list,
        history: list[str],
        step_idx: int,
        max_steps: int,
        system_prompt: str,
        user_template: str,
        actions: list[dict],
        actions_block: str,
        test_data_block: str,
        screenshot_b64: str | None = None,
        visited_summary: str = "",
        # PER-169: episodic memory block — already-recalled text
        # the caller wants prepended to the user prompt. Empty
        # means no memory (no graphiti / no relevant episodes).
        memory_block: str = "",
    ) -> dict[str, Any] | None:
        """One LLM round for a goal node (PER-111 v2).

        Returns the raw decoded JSON object the model produced, or
        None on transport / parse failure (caller breaks the loop).

        ``actions`` is the workspace-curated action dictionary
        (claim_next ships it); ``actions_block`` and
        ``test_data_block`` are the pre-rendered prompt fragments.
        Schema is rebuilt per call because ``element_id`` enum
        depends on what's on screen right now — that's the whole
        point of the new contract.

        PER-119: ``screenshot_b64`` is the rendered screen as base64
        PNG. When provided, the user message is sent as a multimodal
        content array (text + image) so the LLM can ground its
        decision on the visible UI. When None we fall back to
        text-only — the AXe accessibility dump still has everything
        the model needs for screens with proper labels.
        """
        # Per-step elements block (and the live id enum the schema
        # depends on).
        elements_block, element_ids = build_elements_block(elements)
        history_block = (
            "\n".join(
                f"  {i + 1}. {h}" for i, h in enumerate(history[-8:])
            )
            if history else "  (пока ничего)"
        )
        # PER-161: prepend visited_actions when present. The LLM
        # already reads history; tucking the note in at the top of
        # the same block keeps the prompt template stable (no new
        # placeholder) and the model picks it up without retraining.
        if visited_summary:
            history_block = (
                f"  (на этом экране уже пробовали: {visited_summary})\n"
                + history_block
            )

        test_data_keys = list(self.test_data.keys())
        schema = build_goal_schema(
            test_data_keys=test_data_keys,
            actions=actions,
            element_ids=element_ids,
        )

        user_prompt = _render_template(
            user_template,
            {
                "mode": mode,
                "goal": goal_text,
                "step_idx": str(step_idx + 1),
                "max_steps": str(max_steps),
                "success_criteria": success_criteria,
                "elements_block": elements_block,
                "actions_block": actions_block,
                "test_data_block": test_data_block,
                "history_block": history_block,
            },
        )
        # PER-198: Context Identifier hint. Classify the current screen
        # (DeBERTa zero-shot microservice); on a PIN/secret-code screen
        # count how many digit taps already happened this goal and, once
        # >=4, inject a hard rule that the submit button is mandatory.
        # This is the behavioural lever for the PER-172 PIN bug: the
        # Planner now *knows* the screen type and the digit count, so it
        # stops re-tapping the keypad. Best-effort — failure / unassigned
        # role just skips the hint (legacy behaviour).
        ctx_hint = await self._context_pin_hint(elements_block, history)
        if ctx_hint:
            user_prompt = ctx_hint + "\n\n" + user_prompt

        # PER-200 (loop-breaker, pure-Python slice): if the last 3
        # history entries are near-identical the agent is stuck in a
        # cycle (the «tap 8 ×5» pattern). Inject a hard «this isn't
        # working, do something DIFFERENT» rule. Cheap — no LLM call —
        # and complements the Context Identifier hint for the general
        # case (non-PIN loops too).
        loop_hint = _loop_breaker_hint(history)
        if loop_hint:
            # PER-200: when stuck, escalate to the Reflection agent for a
            # specific «try this instead» recommendation (LLM, but only
            # fires on the rare stuck-event, not every step). Falls back
            # to the static hint when reflection is unavailable / fails.
            if self.reflection_agent is not None and goal_text:
                try:
                    note = await self.reflection_agent.review(goal_text, history or [])
                except Exception as exc:
                    logger.debug("reflection review failed: %s", exc)
                    note = None
                if note is not None and note.recommendation:
                    logger.info(
                        "[PER-200 reflection] stuck=%s rec=%s",
                        note.stuck, note.recommendation[:120],
                    )
                    loop_hint = (
                        loop_hint
                        + f"\n🧭 РЕКОМЕНДАЦИЯ РЕФЛЕКСИИ: {note.recommendation}"
                    )
            user_prompt = loop_hint + "\n\n" + user_prompt

        # PER-200: credential→screen routing. Always-on (cheap) so the
        # model picks the right test_data key per screen type — the
        # direct fix for the «entered sms_code on the PIN screen» bug.
        cred_hint = self._credential_routing_hint()
        if cred_hint:
            user_prompt = cred_hint + "\n\n" + user_prompt

        # PER-169: episodic memory recall as a prepended block.
        # We prepend instead of using a template placeholder so the
        # legacy DB system_prompts.user template (which has no
        # {{memory_block}} marker) keeps working — agents on old
        # workspaces get the same behaviour, opt-in is "wire
        # EpisodicMemory into ScenarioRunner".
        if memory_block:
            user_prompt = (
                "Память о ранее выполненных действиях в этой цели "
                "(используй чтобы НЕ повторять уже сделанное):\n"
                f"{memory_block}\n\n"
                + user_prompt
            )

        # PER-131-lite: if the active model carries a thinking
        # passport, we split the call into two passes — first a
        # free-form reasoning pass with the activation token in the
        # system prompt and NO response_format (the model would
        # otherwise be forced to emit JSON from token #1, leaving no
        # room for the reasoning block); then a constrained JSON
        # pass with the reasoning fed back as context. Without the
        # passport (legacy / non-thinking model) we collapse to the
        # single constrained call the worker has done from day one.
        try:
            if self._supports_thinking and self._thinking_activation:
                thinking_text = await self._goal_think_pass(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    screenshot_b64=screenshot_b64,
                )
                # On a thinking-pass failure (timeout / parse error)
                # fall through to single-pass with the original
                # prompt so the run keeps making progress instead of
                # stalling on a brittle preamble.
                effective_user = (
                    user_prompt
                    if not thinking_text
                    else (
                        f"{user_prompt}\n\n"
                        f"Твои предварительные размышления:\n{thinking_text}\n\n"
                        f"Теперь сформируй окончательный JSON-ответ по схеме."
                    )
                )
            else:
                effective_user = user_prompt

            # PER-138: only ask for json_schema constrained decoding
            # when the backend can compile a grammar from it. For
            # endpoints that can't (some OpenAI-compat, Anthropic
            # API, custom adapters), drop response_format and rely
            # on parse-recovery (raw_decode tolerates trailing
            # chatter and markdown fences) plus the existing
            # anti-loop guard for shape-violating outputs.
            # PER-138 follow-up: reasoning models (Nemotron Reasoning,
            # Qwen3-Thinking) can sneak <think>…</think> into the JSON
            # pass too — 400 tokens cuts the JSON in half (real bug:
            # `"value_l` truncation on Nemotron smoke). 1500 gives
            # enough headroom both for residual reasoning and for the
            # 9-field JSON shape, including long value_literal /
            # reasoning fields.
            # PER-144 fix: thinking-capable models (Nemotron Reasoning,
            # Qwen3-Thinking) often emit `<think>…</think>` even on the
            # JSON pass that already saw the reasoning from pass #1, and
            # 1500 tokens isn't enough — the model burns through them
            # inside the second think-block and hits finish_reason=length
            # before any visible JSON. Doubled budget gives reasoning
            # models the headroom; non-thinking models almost never use
            # more than ~200 tokens for the JSON shape anyway, so the
            # ceiling-bump costs nothing in the dense-instruct case.
            json_max_tokens = 4000 if self._supports_thinking else 1500
            chat_kwargs = {
                "system": system_prompt,
                "user": effective_user,
                "max_tokens": json_max_tokens,
                "screenshot_b64": screenshot_b64,
                # PER-164 followup: per-model sampling. Without this,
                # llm_client.chat() defaulted to T=0.2 for every
                # model — root cause of Gemma 4's "impulsive back-tap
                # after one wait" on loading screens. Passport now
                # forwards Gemma 4's recommended T=0.65/top_p=0.95/
                # top_k=64/min_p=0.05, Qwen-family's T=0.7/top_p=0.8/
                # top_k=20/min_p=0, etc.
                "temperature": self._sampling_temperature,
                "top_p": self._sampling_top_p,
                "top_k": self._sampling_top_k,
                "min_p": self._sampling_min_p,
            }
            if self._supports_json_schema:
                chat_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "goal_decision",
                        "schema": schema,
                    },
                }
            resp = await self.llm_client.chat(**chat_kwargs)
        except Exception as exc:
            logger.warning("[scenario] goal LLM call failed: %s", exc)
            return None
        if not resp:
            # PER-143 follow-up: thinking-capable models (Nemotron
            # Reasoning, Qwen3-Thinking) sometimes consume the entire
            # max_tokens budget inside <think>…</think> on the JSON
            # pass and emit no visible answer before EOS. Without this
            # log the goal silently fails with ``llm_no_decision``,
            # making the issue invisible. The empty string is the
            # signal — see PER-144 for the structural fix.
            logger.warning(
                "[scenario] goal LLM returned empty response "
                "(likely all max_tokens consumed by thinking); "
                "thinking_passport=%s, json_schema=%s",
                self._supports_thinking, self._supports_json_schema,
            )
            return None

        # Even with constrained decoding the chat-template can wrap the
        # JSON in markdown fences or a leading "Answer:" prefix — strip
        # those, then use raw_decode to read exactly one balanced
        # object (immune to trailing chatter the model might add when
        # the grammar is not enforced — graceful-fallback path).
        text = resp.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
        start = text.find("{")
        if start < 0:
            logger.warning(
                "[scenario] goal LLM returned no JSON: %r", text[:200]
            )
            return None
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            logger.warning(
                "[scenario] goal LLM JSON parse failed: %s; raw=%r",
                exc, text[start:start + 300],
            )
            return None

        # PER-163 retry #2: post-decode validation.
        #
        # Background: llama-server's JSON-Schema → GBNF compiler does
        # not reliably enforce per-branch ``oneOf`` constraints. The
        # PER-163 fix made base ``element_id`` nullable (so coord-only
        # ``tap_at`` can return null per the prompt), and per-branch
        # oneOf for element-targeted actions tries to pin a non-null
        # string from the enum — but the grammar compiler often
        # ignores the branch constraint. Result: model can return
        # ``action=tap`` (or input/long_press/assert) with
        # ``element_id=null`` — which then crashes ``_find_element``
        # or worse, falls through to a fake-id retry that wastes
        # max_steps.
        #
        # We catch it here, log loudly, and return None so the goal
        # loop counts the step as ``llm_no_decision`` and moves on.
        # The dispatcher will not see this malformed shape; the LLM
        # gets a fresh prompt next iteration where visited_actions
        # already covers any partial state.
        try:
            from explorer.goal_schema import _ELEMENT_TARGETED_ACTIONS
        except ImportError:
            _ELEMENT_TARGETED_ACTIONS = frozenset()
        decoded_action = (obj.get("action") or "").strip().lower()
        # done=true short-circuits the goal loop regardless of action;
        # validation only matters when the loop will actually
        # dispatch.
        if (
            not obj.get("done")
            and decoded_action in _ELEMENT_TARGETED_ACTIONS
            and obj.get("element_id") in (None, "", "null")
        ):
            logger.warning(
                "[scenario] schema-leak rejected: action=%r requires "
                "element_id, model returned null; treating as no_decision",
                decoded_action,
            )
            return None
        # PER-204: stamp the cached PIN verdict (set by _context_pin_hint
        # above) so _run_goal_node can fire the submit-macro.
        obj["context_is_pin"] = self._context_is_pin_last
        return obj

    async def _dispatch(
        self,
        action: str,
        label: str,
        value: str,
        element_id: str | None = None,
        action_args: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None]:
        """Map a scenario action onto a controller call.

        PER-111 v2:
            * ``element_id`` (when provided) takes priority over
              ``label`` for element lookup. Goal-node decisions always
              carry an id; legacy linear scenarios only carry a label,
              hence the optional parameter.
            * ``action_args`` carries the structured args the LLM
              produced for direction-based / timing-based actions
              (``swipe.direction``, ``scroll.direction``, ``wait.ms``,
              ``long_press.duration_ms``). For tap / input / back /
              assert the dict is empty.
            * Every action seeded by migration 20260518_per111_v2
              (tap / input / swipe / wait / assert / long_press /
              scroll / back) has a real implementation here — no
              silent fall-through to tap, which previously made
              ``scroll`` from the LLM look like a successful step
              while actually doing the wrong thing.
        """
        args = action_args or {}

        if action == "back":
            ok = await self.controller.go_back()
            return ok, None if ok else "go_back returned False"

        # PER-137: coordinate tap and free-form text entry. Both let
        # the LLM act on UIs whose elements aren't in the accessibility
        # tree (PIN keypads, canvas-painted buttons, RN screens without
        # accessibilityIdentifier). Neither requires element_id — the
        # action argues directly with the screen, not via the AXe enum.
        if action == "tap_at":
            # PER-164: when the chat-LLM provides a
            # ``target_description``, route the coordinate decision
            # through the dedicated UI-grounder (UI-TARS et al)
            # instead of trusting the chat-LLM's own (x, y) — dense
            # general-purpose VLMs are unreliable at canvas-keypad
            # localisation (see PER-163 retry #2 smoke comparison).
            # The grounder returns its own (x, y) tagged with its
            # own coord_space (UI-TARS = ``pixels``, Molmo =
            # ``normalized_1000``, etc.), so the space used for
            # scaling depends on which model produced the number.
            target_description = (args.get("target_description") or "").strip()
            grounder_used = False
            raw_x: int
            raw_y: int
            space: str = self._tap_at_coord_space
            # Image dims used by the grounder — needed when its
            # coord_space is image_pixels so we can scale back to
            # phone logical points. Zero = unknown / not applicable.
            grounder_image_w: int = 0
            grounder_image_h: int = 0
            if target_description and self._grounder is not None:
                shot: tuple[bytes, int, int] | None = None
                try:
                    shot = await self._take_screenshot_bytes()
                except Exception:
                    logger.exception(
                        "tap_at: failed to capture screenshot for grounder "
                        "— falling back to chat-LLM coords"
                    )
                if shot is not None:
                    shot_bytes, grounder_image_w, grounder_image_h = shot
                    try:
                        located = await self._grounder.locate(shot_bytes, target_description)
                    except Exception:
                        logger.exception(
                            "tap_at: grounder.locate raised — falling back to chat-LLM coords"
                        )
                        located = None
                    if located is not None:
                        raw_x = int(located.x)
                        raw_y = int(located.y)
                        space = located.coord_space
                        grounder_used = True
                        logger.info(
                            "tap_at: grounder %s → (%d,%d) space=%s image=(%d,%d) target=%r",
                            getattr(located, "raw_text", "")[:60] or "?",
                            raw_x, raw_y, space,
                            grounder_image_w, grounder_image_h,
                            target_description,
                        )
            if not grounder_used:
                try:
                    raw_x = int(args.get("x"))
                    raw_y = int(args.get("y"))
                except (TypeError, ValueError):
                    return False, (
                        "tap_at requires integer x, y "
                        "(or args.target_description when a grounder is active)"
                    )
            # PER-145 L1 / PER-164: scale into screen points by space.
            #   ``points``           → pass through (Gemma family)
            #   ``normalized_1000``  → 0–1000 normalized (Qwen2.5/3-VL)
            #   ``pixels``           → raw retina pixels (Nemotron)
            #   ``image_pixels``     → PER-164: pixels of the sent image,
            #                          which was resized to
            #                          screenshot_max_dim BEFORE going to
            #                          the grounder. Scale by
            #                          screen_logical / image_dim, NOT
            #                          retina factor — model never saw
            #                          retina-native pixels.
            # Screen dimensions live on the AXe controller, captured
            # at connect-time. Fall back to (raw_x, raw_y) if dims are
            # missing — better to try than to crash.
            screen_w = int(getattr(self.controller, "_width", 0) or 0)
            screen_h = int(getattr(self.controller, "_height", 0) or 0)
            scale = float(getattr(self.controller, "_scale", 1.0) or 1.0)
            if space == "normalized_1000" and screen_w > 0 and screen_h > 0:
                x = int(round(raw_x / 1000.0 * screen_w))
                y = int(round(raw_y / 1000.0 * screen_h))
            elif space == "image_pixels" and (
                grounder_image_w > 0 and grounder_image_h > 0
                and screen_w > 0 and screen_h > 0
            ):
                x = int(round(raw_x * screen_w / grounder_image_w))
                y = int(round(raw_y * screen_h / grounder_image_h))
            elif space == "pixels" and scale > 0:
                x = int(round(raw_x / scale))
                y = int(round(raw_y / scale))
            else:
                x, y = raw_x, raw_y
            tap_at_fn = getattr(self.controller, "tap_at", None)
            if not callable(tap_at_fn):
                return False, "controller has no tap_at"
            result = await tap_at_fn(x, y)
            ok = bool(result and getattr(result, "ok", True))
            return ok, None if ok else (
                f"tap_at(raw={raw_x},{raw_y} space={space} → "
                f"pts={x},{y}) failed"
            )

        if action == "enter_text":
            # PER-143: prefer the value already resolved by the caller
            # via :func:`goal_schema.resolve_value` (which honours
            # value_source ∈ {test_data, improvised, literal}). Fall
            # back to a literal ``args.text`` only if the LLM emitted
            # the text inline without setting value_source. This keeps
            # enter_text consistent with ``input``: the model can pick
            # either contract and the runtime still gets the right
            # string — including secret-aware lookups like
            # test_data.phone → "+79051543055" (with leading prefix)
            # rather than the raw digits the LLM tends to inline.
            text = value if value else args.get("text")
            if not isinstance(text, str) or not text:
                return False, (
                    "enter_text requires non-empty text "
                    "(set value_source or args.text)"
                )
            # PER-160: enter_text was the original "free-form text
            # into the currently focused field" action, but
            # ``controller.type_text`` still routes through CDP first
            # if available — which silently bypasses native input
            # events the same way set_text_in_field did. Prefer the
            # HID-only path when the controller exposes one (real AXe
            # controller does; FakeController in tests does not, hence
            # the getattr fallback).
            hid_fn = getattr(self.controller, "type_text_via_hid", None)
            if callable(hid_fn):
                typed = await hid_fn(text)
            else:
                type_fn = getattr(self.controller, "type_text", None)
                if not callable(type_fn):
                    return False, "controller has no type_text or type_text_via_hid"
                typed = await type_fn(text)
            return bool(typed), None if typed else "type_text returned False"

        if action == "wait":
            ms = args.get("ms")
            # Controller may not implement wait_ms on every backend —
            # fall back to plain asyncio sleep so the action still
            # behaves correctly.
            wait_ms_fn = getattr(self.controller, "wait_ms", None)
            try:
                ms_int = int(ms) if ms is not None else 500
            except (TypeError, ValueError):
                ms_int = 500
            if callable(wait_ms_fn):
                ok = await wait_ms_fn(ms_int)
            else:
                await asyncio.sleep(max(0.1, min(ms_int / 1000.0, 60.0)))
                ok = True
            return bool(ok), None if ok else "wait failed"

        if action == "swipe":
            direction = (args.get("direction") or "").strip().lower()
            # Direction wins when supplied (v2 contract); coordinate
            # form is the legacy fallback for old linear scenarios
            # that encoded "x1,y1,x2,y2" in value.
            if direction:
                swipe_dir_fn = getattr(self.controller, "swipe_direction", None)
                if callable(swipe_dir_fn):
                    ok = await swipe_dir_fn(direction)
                    return bool(ok), None if ok else f"swipe {direction} failed"
                # Controller lacks swipe_direction — fall back to a
                # coordinate swipe derived from screen size if we can
                # introspect it; otherwise it's a configuration bug.
                ok = await self._fallback_directional_swipe(direction)
                return ok, None if ok else f"swipe {direction} unsupported"
            coords = self._parse_swipe(value)
            ok = await self.controller.swipe(*coords)
            return bool(ok), None if ok else "swipe returned False"

        if action == "scroll":
            direction = (args.get("direction") or "").strip().lower()
            scroll_fn = getattr(self.controller, "scroll", None)
            if callable(scroll_fn) and direction:
                ok = await scroll_fn(direction)
                return bool(ok), None if ok else f"scroll {direction} failed"
            # Fall back through swipe_direction with the same arg —
            # iOS treats them the same at the gesture layer.
            swipe_dir_fn = getattr(self.controller, "swipe_direction", None)
            if callable(swipe_dir_fn) and direction:
                ok = await swipe_dir_fn(direction)
                return bool(ok), None if ok else f"scroll {direction} failed"
            return False, "scroll unsupported (no scroll/swipe_direction)"

        if action == "assert":
            # Soft check: present in current elements? Lookup uses
            # the same retry loop as actions so async screens settle.
            element = await self._find_element(label, element_id=element_id)
            target = element_id or label
            return (element is not None,
                    None if element is not None else f"element {target!r} not found")

        # Below: actions that target a specific element on screen.
        element = await self._find_element(label, element_id=element_id)
        if element is None:
            target = element_id or label
            return False, (
                f"element {target!r} not found after {_LOOKUP_RETRIES} tries"
            )

        if action == "long_press":
            duration_ms = args.get("duration_ms") or 800
            long_press_fn = getattr(self.controller, "long_press", None)
            frame = element.get("frame") or {}
            try:
                cx = int(frame.get("x", 0)) + int(frame.get("width", 0)) // 2
                cy = int(frame.get("y", 0)) + int(frame.get("height", 0)) // 2
            except (TypeError, ValueError):
                cx, cy = None, None
            if callable(long_press_fn):
                ok = await long_press_fn(cx, cy, duration_ms)
                return bool(ok), None if ok else "long_press failed"
            return False, "long_press unsupported by controller"

        if action == "input":
            test_id = element.get("test_id") or element.get("identifier")
            elem_label = element.get("label") or label or ""
            # PER-160: the user-facing path. Tap the field so iOS brings
            # up the keyboard, wait until it is on-screen, then send
            # HID keystrokes — same code path a real fingertip takes.
            # This is the only way to find keyboard-related bugs
            # (overlay covering field, broken onChange handler, missing
            # custom keyboard, anti-fraud rejecting non-native input)
            # and the only way React/iOS state updates correctly on
            # apps that listen to UIControlEventEditingChanged rather
            # than reading the field value at submit-time.
            #
            # Strict opt-out: ``action_args.bypass_keyboard=true`` keeps
            # the old set_text_in_field path for the rare cases where
            # the keyboard genuinely cannot be brought up (custom IME,
            # canvas-rendered field, headless flow).
            bypass = bool(args.get("bypass_keyboard"))
            kbd_fn = getattr(
                self.controller, "tap_field_and_type_via_keyboard", None
            )
            if not bypass and callable(kbd_fn) and (test_id or elem_label):
                ok, reason = await kbd_fn(test_id, elem_label, value)
                return bool(ok), None if ok else (
                    reason or "keyboard input returned False"
                )
            # Bypass / legacy path — write the value through CDP or
            # AXe set-text. Faster, but does NOT trigger native input
            # events. See PER-160 for why this is opt-in now.
            #
            # The real AXe controller's signature is
            # set_text_in_field(test_id, label, text) — three args.
            # The test FakeController uses two (test_id, text). Resolve
            # at call time by introspecting the param count so both
            # work without forking the dispatcher.
            from inspect import signature
            try:
                params = signature(self.controller.set_text_in_field).parameters
                arity = len([
                    p for p in params.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ])
            except (TypeError, ValueError):
                arity = 2  # err on the side of the simpler signature
            if test_id or elem_label:
                if arity >= 3:
                    ok = await self.controller.set_text_in_field(
                        test_id or "", elem_label, value
                    )
                else:
                    ok = await self.controller.set_text_in_field(
                        test_id or elem_label, value
                    )
                return bool(ok), (
                    None if ok else "set_text_in_field returned False"
                )
            # No usable identifier — fall back to tap-then-type. Uses
            # the resolved element's label as the tap target so the
            # AXe controller can find it the same way.
            tapped = await self.controller.tap_by_label(elem_label)
            if not tapped or not getattr(tapped, "ok", True):
                return False, "tap before type failed"
            typed = await self.controller.type_text(value)
            return typed, None if typed else "type_text returned False"

        if action == "tap":
            # Prefer the resolved element's label (what the AXe
            # controller keys on) over the LLM's echo — handles cases
            # where the model passed only element_id.
            tap_target = (element.get("label") or label or "")
            result = await self.controller.tap_by_label(tap_target)
            ok = bool(result and getattr(result, "ok", True))
            return ok, None if ok else "tap_by_label returned not-ok"

        # Unknown action — treat as soft failure rather than silently
        # tapping. The LLM would otherwise see "OK" for an action the
        # worker can't actually perform, which corrupts its history.
        return False, f"unsupported action {action!r}"

    async def _fallback_directional_swipe(self, direction: str) -> bool:
        """Compute a vertical/horizontal swipe from screen dimensions
        when the controller lacks a direction-aware swipe. Falls back
        on a fixed 300px swipe in the screen centre if we don't know
        the size."""
        w = getattr(self.controller, "_width", None) or 390
        h = getattr(self.controller, "_height", None) or 844
        cx, cy = w // 2, h // 2
        delta = min(w, h) // 3
        coords_by_dir = {
            "up": (cx, cy + delta, cx, cy - delta),
            "down": (cx, cy - delta, cx, cy + delta),
            "left": (cx + delta, cy, cx - delta, cy),
            "right": (cx - delta, cy, cx + delta, cy),
        }
        coords = coords_by_dir.get(direction.lower())
        if not coords:
            return False
        try:
            ok = await self.controller.swipe(*coords)
        except AttributeError:
            return False
        return bool(ok)

    async def _find_element(
        self, label: str, element_id: str | None = None
    ) -> dict | None:
        """Look the element up on the current screen, with retries to
        absorb settle-after-navigation timing.

        PER-111 v2: when ``element_id`` is provided, it takes priority
        — the LLM picks element_id from a constrained enum of stable
        AXe identifiers, so an exact id match is the most reliable
        lookup path. ``label`` is the human-readable fallback used by
        legacy scenarios that only carry text labels and by free
        exploration.
        """
        if not label and not element_id:
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
            # Pass 1: id-first match (stable across re-renders).
            if element_id:
                for el in elements:
                    if (
                        el.get("id") == element_id
                        or el.get("test_id") == element_id
                        or el.get("identifier") == element_id
                    ):
                        return el
            # Pass 2: label fallback for legacy scenarios.
            if label:
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
        except asyncio.CancelledError:
            # Task cancellation from above must propagate so the
            # asyncio runtime can unwind cleanly.
            raise
        except Exception as exc:
            # PER-110 follow-up: when the backend tells us the run is
            # in a terminal state (409 → RunCancelled), the worker's
            # outer loop is responsible for tearing down. If we swallow
            # the exception here, the goal loop and the rest of the
            # scenario keep spinning forever — that's what produced the
            # "stuck after cancel" hang during the demo. We don't
            # import RunCancelled to avoid a circular dep with worker.py,
            # so detect by class name.
            if type(exc).__name__ == "RunCancelled":
                raise
            logger.exception(
                "[scenario] event callback failed (event=%s)",
                event.get("type"),
            )
