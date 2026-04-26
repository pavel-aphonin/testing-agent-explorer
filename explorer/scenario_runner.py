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
    ) -> None:
        self.controller = controller
        self.scenarios = scenarios or []
        self.test_data = test_data or {}
        self.event_callback = event_callback

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
        sid = scenario.get("id", "?")
        title = scenario.get("title") or "(без названия)"
        steps = scenario.get("steps") or []
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

    async def _run_step(
        self, scenario_id: str, step_idx: int, step: dict[str, Any]
    ) -> tuple[bool, str | None]:
        action = (step.get("action") or "tap").lower()
        label = step.get("element_label") or ""
        value = _substitute_test_data(step.get("value") or "", self.test_data)
        await self._emit({
            "type": "scenario.step_started",
            "scenario_id": scenario_id,
            "step_idx": step_idx,
            "action": action,
            "element_label": label,
        })
        try:
            ok, reason = await self._dispatch(action, label, value)
        except Exception as exc:
            logger.exception("[scenario] step crashed")
            ok, reason = False, f"crash: {exc}"
        await self._emit({
            "type": "scenario.step_completed" if ok else "scenario.step_failed",
            "scenario_id": scenario_id,
            "step_idx": step_idx,
            "action": action,
            "element_label": label,
            "reason": reason,
        })
        return ok, reason

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
