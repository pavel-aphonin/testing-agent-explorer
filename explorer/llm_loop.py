"""LLM-driven exploration loop.

The LLM is the brain: it sees the screenshot + element list on every
step and decides what to do next. No PUCT formulas, no heuristics.

Flow:
    1. Capture screenshot + elements
    2. Send to LLM: "What do you see? What should we do next?"
    3. LLM returns: {action, element_index, value, reasoning}
    4. Execute the action
    5. Record the transition
    6. Repeat
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from explorer.models import (
    ActionDetail,
    ActionType,
    ElementKind,
    ElementSnapshot,
    GraphEdge,
    ScreenNode,
)

logger = logging.getLogger("explorer.llm_loop")

SETTLE_DELAY = 3.0
AXE_BIN = "/opt/homebrew/bin/axe"

EventCallback = Callable[[dict], None | Awaitable[None]]

SYSTEM_PROMPT = """You are an AI agent exploring a mobile app. Your goal is to discover ALL screens and features of the app by interacting with UI elements.

On each step you see:
1. A list of interactive elements on the current screen (buttons, text fields, switches, etc.)
2. A history of your previous actions

You must respond with a JSON object:
{
    "reasoning": "Brief explanation of what you see and why you chose this action",
    "action": "tap" or "input",
    "element_index": 0,
    "value": "text to type (only for input action, null for tap)"
}

Rules:
- If you see a login/registration form, fill in ALL fields with appropriate data and tap the submit button
- For email fields, use: test@test.com
- For password fields, use: password123
- For name fields, use: Test User
- For phone fields, use: +7 900 000-00-00
- After logging in successfully, explore ALL buttons and navigation elements
- If you see a system popup (Save Password, Allow Notifications), dismiss it by tapping "Not Now" / "Cancel" / "OK"
- If you're stuck on the same screen, try different elements or go back
- Prefer untapped buttons over already-explored ones
- Go DEEPER into the app — don't stay on one screen

IMPORTANT: Respond with ONLY the JSON object, no other text."""


class LLMExplorationLoop:
    """LLM-driven exploration: the model decides every action."""

    def __init__(
        self,
        controller: Any,
        app_bundle_id: str,
        llm_base_url: str = "http://localhost:8080",
        llm_model: str = "embeddings",
        max_steps: int = 200,
        event_callback: EventCallback | None = None,
        capture_retries: int = 5,
        capture_retry_delay: float = 2.0,
        rag_enabled: bool = False,
        rag_base_url: str = "http://localhost:8000",
        rag_token: str = "",
        test_data: dict[str, str] | None = None,
        scenarios: list[dict] | None = None,
        max_steps_per_screen: int = 20,
        defect_detection_enabled: bool = True,
        defect_llm_base_url: str | None = None,
        defect_callback: "Callable[[dict], Awaitable[None]] | None" = None,
        run_id: str | None = None,
        pbt_enabled: bool = False,
    ) -> None:
        self.controller = controller
        self.app_bundle_id = app_bundle_id
        self.llm_url = llm_base_url.rstrip("/")
        self.llm_model = llm_model
        self.max_steps = max_steps
        self.event_callback = event_callback
        self.capture_retries = capture_retries
        self.capture_retry_delay = capture_retry_delay
        self.rag_enabled = rag_enabled
        self.rag_base_url = rag_base_url.rstrip("/")
        self.rag_token = rag_token
        # Test data entries — {key: value}. The agent is told about these in
        # the system prompt and should prefer them over generic placeholders
        # when filling form fields. Example: {"email": "test@example.com"}.
        self.test_data: dict[str, str] = test_data or {}
        # Pre-scripted scenarios to execute before free exploration. Each is
        # {id, title, steps: [{screen_name, action, element_label, value?, ...}]}.
        # The agent walks them in order (with {{test_data.key}} substitution),
        # then falls back to free exploration for remaining steps.
        self.scenarios: list[dict] = scenarios or []
        # Budget cap: if the agent spends more than this many consecutive steps
        # on the same screen_id, we force-move it away. Prevents "LLM burns all
        # tokens typing 1, 2, 3, ... into a number field" (Garri's concern at
        # the demo). Default 20 ~ enough for typical form fill + submit.
        self.max_steps_per_screen = max_steps_per_screen
        self._current_screen_step_count = 0

        # Defect detector: after each action, ask a separate LLM call whether
        # what we observed is a real defect (vs infra noise) and rank it.
        # Defects are posted to the backend and surface in the Defects tab.
        self.defect_callback = defect_callback
        self.run_id = run_id
        self._defect_detector = None
        if defect_detection_enabled and defect_llm_base_url:
            from explorer.defect_detector import DefectDetector
            self._defect_detector = DefectDetector(
                llm_base_url=defect_llm_base_url,
                model="rag-chat",
            )

        # Property-based testing: when enabled, the agent's system prompt
        # includes per-field-type variants (empty / overflow / XSS / SQL
        # injection / unicode) so the LLM systematically probes form
        # validation. The DefectDetector then catches "app accepted invalid
        # input" as a validation defect.
        self.pbt_enabled = pbt_enabled

        self.screens: dict[str, ScreenNode] = {}
        self.edges: list[GraphEdge] = []
        self.action_log: list[dict] = []
        self._history: list[str] = []  # human-readable action history for LLM context
        self._tried_fallbacks: set[str] = set()  # elements tried when LLM unavailable

        self._step = 0
        self._current_screen_id = ""
        self._consecutive_no_new = 0  # count steps with no new discoveries

    def _substitute_test_data(self, text: str) -> str:
        """Replace {{test_data.KEY}} placeholders with configured values."""
        import re
        if not text or "{{" not in text:
            return text

        def _repl(match: "re.Match[str]") -> str:
            key = match.group(1).strip()
            return self.test_data.get(key, match.group(0))

        # Supports {{test_data.email}} and shorthand {{email}}
        text = re.sub(r"\{\{\s*test_data\.(\w+)\s*\}\}", _repl, text)
        text = re.sub(r"\{\{\s*(\w+)\s*\}\}", _repl, text)
        return text

    @staticmethod
    def _pbt_prompt_section() -> str:
        """Block of PBT instructions appended to SYSTEM_PROMPT when enabled.

        Lists the variant categories built into PBTInputGenerator so the LLM
        knows what to try and what counts as a validation bug.
        """
        return """
## PBT (property-based testing) MODE — ENABLED

For every text field you encounter, you should systematically probe its
validation. After entering a valid value and successfully moving forward,
COME BACK to the form (use the back button) and try the next variant.
Repeat until you've covered the categories below.

Per-field-type variants to try:
- Email: empty, "not-an-email", "<script>alert(1)</script>@test.com",
  "' OR 1=1 --", very long string (a*500 + @test.com), unicode "ТЕСТ@тест.рф"
- Password: empty, "ab" (too short), "a"*1000, "<script>alert(1)</script>",
  "' OR 1=1 --"
- Phone: empty, "abc", "+7 900 000-00-0" repeated 20×
- Name: empty, "A" (too short), "А"*1000, "<script>alert(1)</script>"
- Generic text: empty, "a"*1000, "<script>alert(1)</script>"

After entering each variant, click submit/proceed and observe.
DEFECTS to flag (the DefectDetector will pick these up):
- App ACCEPTED clearly invalid input (validation broken) — P1 validation defect
- App CRASHED on long input or special chars — P0 crash
- Form looks broken / fields disappear — UI defect"""

    def _build_system_prompt(self) -> str:
        """Build the system prompt, injecting test_data and scenario plans.

        When test_data is configured we tell the model about keys so it prefers
        them over generic placeholders. When scenarios are configured we list
        the upcoming scripted steps so the model anchors its actions to them
        before exploring freely. When PBT is enabled, validation-probing
        instructions are appended.
        """
        extras: list[str] = []

        if self.test_data:
            extras.append("")
            extras.append("## Available test data (prefer these over generic defaults)")
            extras.append("")
            extras.append("When filling form fields, use these values whenever the field matches:")
            extras.append("")
            for key, value in self.test_data.items():
                extras.append(f"- `{key}` = `{value}`")
            extras.append("")
            extras.append(
                "For example, if you see an email field and `email` is listed above, "
                "use THAT value instead of the fallback `test@test.com`."
            )

        if self.scenarios:
            extras.append("")
            extras.append("## Scenarios to follow FIRST (before free exploration)")
            extras.append("")
            extras.append(
                "The user has configured scripted scenarios. Execute each step in "
                "order. After all scenarios complete, continue with free exploration."
            )
            for sc in self.scenarios:
                extras.append("")
                extras.append(f"### Scenario: {sc.get('title', '(untitled)')}")
                for i, step in enumerate(sc.get("steps", []), 1):
                    screen = step.get("screen_name", "")
                    action = step.get("action", "tap")
                    element = step.get("element_label", "")
                    value = self._substitute_test_data(step.get("value", "") or "")
                    expected = step.get("expected_result", "")
                    parts = [f"{i}."]
                    if screen:
                        parts.append(f"[{screen}]")
                    parts.append(f"{action}")
                    if element:
                        parts.append(f'"{element}"')
                    if value:
                        parts.append(f'with value "{value}"')
                    if expected:
                        parts.append(f"→ expect: {expected}")
                    extras.append("  " + " ".join(parts))

        if self.pbt_enabled:
            extras.append(self._pbt_prompt_section())

        if not extras:
            return SYSTEM_PROMPT
        return SYSTEM_PROMPT + "\n".join(extras)

    async def run(self) -> dict:
        await self._emit({"type": "status_change", "new_status": "running"})

        print(">>> Waiting for app to settle (2s)...", flush=True)
        await asyncio.sleep(2.0)

        screen = await self._capture()
        self._current_screen_id = screen.screen_id
        self.screens[screen.screen_id] = screen
        await self._emit_screen(screen, step=0)

        print(f">>> Initial: {screen.name!r} ({len(screen.interactive_elements)} elements)", flush=True)

        prev_screen_id = self._current_screen_id
        while self._step < self.max_steps:
            self._step += 1
            step = self._step

            screen = await self._capture()
            self.screens[screen.screen_id] = screen

            # Per-screen budget: if we've spent too many steps on the same
            # screen, force a "go back" to unstick the agent. Prevents LLM
            # from burning tokens on pathological loops (e.g. "let me try
            # entering numbers 1..1000 in this number field").
            if screen.screen_id == prev_screen_id:
                self._current_screen_step_count += 1
            else:
                self._current_screen_step_count = 1
            prev_screen_id = screen.screen_id

            if self._current_screen_step_count > self.max_steps_per_screen:
                msg = (
                    f"Лимит {self.max_steps_per_screen} шагов на одном экране "
                    f"исчерпан — принудительно возвращаемся назад"
                )
                print(f">>> [Step {step}] {msg}", flush=True)
                await self._emit({"type": "log", "step_idx": step, "message": msg})
                await self._go_back()
                self._current_screen_step_count = 0
                continue

            self._current_screen_id = screen.screen_id
            elements = screen.interactive_elements

            if not elements:
                # Screen may still be loading OR a system popup is blocking.
                # Strategy:
                #   1. Wait + retry several times
                #   2. If still 0 elements — send screenshot to LLM and ask
                #      what it sees and where to tap
                for retry in range(self.capture_retries):
                    print(
                        f"    [wait] No elements (attempt {retry + 1}/{self.capture_retries}), "
                        f"waiting {self.capture_retry_delay}s...",
                        flush=True,
                    )
                    await asyncio.sleep(self.capture_retry_delay)
                    screen = await self._capture()
                    self.screens[screen.screen_id] = screen
                    self._current_screen_id = screen.screen_id
                    elements = screen.interactive_elements
                    if elements:
                        print(f"    [wait] Found {len(elements)} elements on retry {retry + 1}", flush=True)
                        break

                if not elements:
                    # Last resort: ask LLM what it sees on the screenshot
                    print(f"    [vision] Asking LLM about blank screen...", flush=True)
                    tap_coords = await self._ask_llm_about_blank_screen()
                    if tap_coords:
                        tx, ty = tap_coords
                        print(f"    [vision] LLM says tap ({tx}, {ty})", flush=True)
                        try:
                            await asyncio.wait_for(
                                self.controller.tap_at(tx, ty), timeout=5
                            )
                            await asyncio.sleep(2)
                            screen = await self._capture()
                            self.screens[screen.screen_id] = screen
                            self._current_screen_id = screen.screen_id
                            elements = screen.interactive_elements
                            if elements:
                                print(f"    [vision] After tap: {len(elements)} elements!", flush=True)
                        except Exception:
                            pass

                if not elements:
                    print(f">>> [Step {step}] Still no elements, swiping back", flush=True)
                    await self._swipe_back()
                    continue

            # Ask LLM what to do
            decision = await self._ask_llm(screen, elements)
            if decision is None:
                # Fallback: tap first untried element, SKIPPING back buttons
                _BACK_LABELS = {"Вход", "Back", "Назад", "<", "‹", "Close", "Закрыть"}
                for i, el in enumerate(elements):
                    label = (el.label or "").strip()
                    # Skip back/navigation buttons in fallback
                    if label in _BACK_LABELS:
                        continue
                    tkey = f"{screen.screen_id}|{el.kind}|{label}"
                    if tkey not in self._tried_fallbacks:
                        self._tried_fallbacks.add(tkey)
                        decision = {
                            "action": "input" if el.kind == ElementKind.TEXT_FIELD else "tap",
                            "element_index": i,
                            "value": "test@test.com" if "email" in (el.test_id or "").lower() else
                                     "password123" if "password" in (el.test_id or "").lower() or
                                     (hasattr(el, 'element_type') and el.element_type == "SecureTextField") else
                                     None,
                            "reasoning": "LLM unavailable, trying next element (skip back buttons)",
                        }
                        break
                if decision is None:
                    self._consecutive_no_new += 1
                    if self._consecutive_no_new >= 15:
                        print(f">>> [Step {step}] All screens explored ({len(self.screens)} screens). Stopping.", flush=True)
                        await self._emit({
                            "type": "log", "step_idx": step,
                            "message": f"Исследование завершено: все найденные экраны ({len(self.screens)}) полностью обследованы.",
                        })
                        break
                    print(f">>> [Step {step}] No untried elements, going back", flush=True)
                    await self._go_back()
                    continue

            action_type = decision.get("action", "tap")
            elem_idx = decision.get("element_index", 0)
            value = decision.get("value")
            reasoning = decision.get("reasoning", "")

            # Clamp index
            if elem_idx < 0 or elem_idx >= len(elements):
                elem_idx = 0

            element = elements[elem_idx]
            before = self._current_screen_id

            print(
                f"\n>>> [Step {step}/{self.max_steps}] "
                f"{screen.name!r}: {action_type} on [{elem_idx}] {element.label!r}"
                + (f" = {value!r}" if value else "")
                + f"\n    Reasoning: {reasoning}",
                flush=True,
            )

            # Execute
            if action_type == "input" and value is not None:
                await self._input_text(element, value)
            else:
                await self._tap(element)

            await asyncio.sleep(SETTLE_DELAY)

            # Observe result
            new_screen = await self._capture()
            is_new = new_screen.screen_id not in self.screens
            self.screens[new_screen.screen_id] = new_screen
            self._current_screen_id = new_screen.screen_id

            # Record
            edge = GraphEdge(
                source_screen_id=before,
                target_screen_id=new_screen.screen_id,
                action=ActionDetail(
                    action_type=ActionType.INPUT if action_type == "input" else ActionType.TAP,
                    target_label=element.label,
                    target_test_id=element.test_id,
                    input_text=value,
                ),
            )
            self.edges.append(edge)

            # RAG verification (if enabled)
            rag_result = None
            if self.rag_enabled:
                rag_result = await self._check_rag(
                    source_screen=screen.name,
                    action=action_type,
                    element_label=element.label,
                    value=value,
                    target_screen=new_screen.name,
                )

            self.action_log.append({
                "step": step,
                "source_screen": before,
                "target_screen": new_screen.screen_id,
                "element": element.label,
                "action": action_type,
                "value": value,
                "reasoning": reasoning,
                "is_new_screen": is_new,
                "rag_result": rag_result,
            })

            # Update history for LLM context
            result_desc = f"→ NEW screen '{new_screen.name}'" if is_new else \
                          f"→ screen '{new_screen.name}'" if new_screen.screen_id != before else \
                          "→ same screen (no change)"
            self._history.append(
                f"Step {step}: {action_type} '{element.label}'"
                + (f" value='{value}'" if value else "")
                + f" {result_desc}"
            )
            # Keep last 20 history items
            if len(self._history) > 20:
                self._history = self._history[-20:]

            # Emit screen event (backend uses it to track visit_count)
            await self._emit_screen(new_screen, step=step, is_new=is_new)

            if is_new:
                print(f"    → NEW: {new_screen.name!r}", flush=True)
                self._consecutive_no_new = 0  # reset — found something new
            elif new_screen.screen_id != before:
                print(f"    → Known: {new_screen.name!r}", flush=True)
                self._consecutive_no_new = 0
            else:
                print(f"    → Same screen", flush=True)

            await self._emit_edge(edge, step=step)

            # Defect detection: ask a separate LLM whether what we observed
            # is a real defect. Runs async — if the detector LLM is slow we
            # don't block the main exploration loop.
            if self._defect_detector is not None and self.defect_callback is not None:
                try:
                    verdict = await self._defect_detector.classify(
                        action=action_type,
                        element_label=element.label or "",
                        value=value,
                        screen_name_before=screen.name,
                        screen_name_after=new_screen.name,
                        element_count_before=len(elements),
                        element_count_after=len(new_screen.interactive_elements),
                        spec_snippet=rag_result if self.rag_enabled else None,
                    )
                    if verdict and verdict.is_defect and not verdict.is_infra:
                        print(
                            f"    ⚠ DEFECT [{verdict.priority}] {verdict.title}",
                            flush=True,
                        )
                        await self.defect_callback({
                            "run_id": self.run_id,
                            "step_idx": step,
                            "screen_id_hash": new_screen.screen_id,
                            "screen_name": new_screen.name,
                            "priority": verdict.priority,
                            "kind": verdict.kind,
                            "title": verdict.title,
                            "description": verdict.description,
                            "screenshot_path": getattr(new_screen, "screenshot_path", None),
                            "llm_analysis_json": {
                                "action": action_type,
                                "element": element.label,
                                "value": value,
                                "before": screen.name,
                                "after": new_screen.name,
                            },
                        })
                except Exception:
                    logger.exception("Defect detection crashed, ignoring")

            if step % 3 == 0 or is_new:
                await self._emit_stats()

        await self._emit_stats()
        print(f"\n>>> Done: {len(self.screens)} screens, {len(self.edges)} edges", flush=True)
        return {"screens": len(self.screens), "edges": len(self.edges), "steps": self._step}

    # ─────────────────── LLM ───────────────────

    async def _ask_llm(
        self, screen: ScreenNode, elements: list[ElementSnapshot]
    ) -> dict | None:
        """Send current screen state to LLM, get back a decision."""
        # Build element list for the prompt
        elem_lines = []
        for i, el in enumerate(elements):
            kind = el.kind.value if hasattr(el.kind, "value") else str(el.kind)
            label = el.label or "(no label)"
            test_id = f" [id={el.test_id}]" if el.test_id else ""
            value = f" value='{el.value}'" if el.value else ""
            elem_lines.append(f"  {i}. [{kind}] '{label}'{test_id}{value}")

        elements_text = "\n".join(elem_lines)

        history_text = "\n".join(self._history[-10:]) if self._history else "No previous actions"

        user_prompt = f"""Current screen: '{screen.name or 'unnamed'}'
Screens discovered so far: {len(self.screens)}

Interactive elements:
{elements_text}

Recent history:
{history_text}

What should I do next? Respond with JSON only."""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.llm_url}/v1/chat/completions",
                    json={
                        "model": self.llm_model,
                        "messages": [
                            {"role": "system", "content": self._build_system_prompt()},
                            {"role": "user", "content": user_prompt},
                        ],
                        "max_tokens": 256,
                        "temperature": 0.3,
                        # Force pure JSON output — no markdown fences,
                        # no thinking tags, just the JSON object
                        "response_format": {"type": "json_object"},
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                print(f"    [LLM raw] {content[:200]}", flush=True)

                # Strip thinking tags if present (Gemma 4 thinking mode)
                if "<think>" in content and "</think>" in content:
                    content = content[content.index("</think>") + len("</think>"):]

                # Parse JSON from response (handle markdown code fences)
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1] if "\n" in content else content
                    content = content.rsplit("```", 1)[0]
                if "{" in content:
                    start = content.index("{")
                    # Handle truncated JSON (max_tokens cut off the closing brace)
                    try:
                        end = content.rindex("}") + 1
                    except ValueError:
                        # Force-close the JSON
                        content += '"}'
                        end = len(content)
                    json_str = content[start:end]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try minimal extraction: just action + element_index
                        import re
                        action_m = re.search(r'"action"\s*:\s*"(\w+)"', json_str)
                        idx_m = re.search(r'"element_index"\s*:\s*(\d+)', json_str)
                        val_m = re.search(r'"value"\s*:\s*"([^"]*)"', json_str)
                        if action_m:
                            return {
                                "action": action_m.group(1),
                                "element_index": int(idx_m.group(1)) if idx_m else 0,
                                "value": val_m.group(1) if val_m else None,
                                "reasoning": "(parsed from truncated response)",
                            }
                        print(f"    [LLM] Unparseable JSON: {json_str[:100]}", flush=True)
                else:
                    print(f"    [LLM] No JSON in response: {content[:100]}", flush=True)

        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            # Fallback: tap first untapped button
            return {
                "action": "tap",
                "element_index": 0,
                "reasoning": f"LLM unavailable ({e}), tapping first element",
            }

        return None

    # ─────────────────── RAG verification ───────────────────

    async def _check_rag(
        self,
        source_screen: str,
        action: str,
        element_label: str | None,
        value: str | None,
        target_screen: str,
    ) -> dict | None:
        """Query the knowledge base to verify if the action result matches spec.

        Returns {matches: [...], violation: bool, message: str} or None on error.
        """
        query = (
            f"User performed '{action}' on '{element_label or 'element'}' "
            f"on screen '{source_screen}'. "
            f"{'Entered: ' + repr(value) + '. ' if value else ''}"
            f"Result: navigated to '{target_screen}'. "
            f"Is this expected behavior?"
        )
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{self.rag_base_url}/api/admin/knowledge/query",
                    headers={"Authorization": f"Bearer {self.rag_token}"},
                    json={"query": query, "top_k": 3},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    matches = data.get("matches", [])
                    if matches:
                        # Found relevant docs — emit as log
                        top = matches[0]
                        await self._emit({
                            "type": "log",
                            "step_idx": self._step,
                            "message": f"RAG: найдено {len(matches)} совпадений в базе знаний. "
                                       f"Ближайшее: \"{top.get('text', '')[:80]}...\" "
                                       f"(расстояние: {top.get('distance', '?'):.3f})",
                        })
                        return data
        except Exception as e:
            logger.debug("RAG check failed: %s", e)
        return None

    # ─────────────────── Vision fallback ───────────────────

    async def _ask_llm_about_blank_screen(self) -> tuple[int, int] | None:
        """When AXe sees 0 elements, send screenshot to LLM and ask what to do.

        The LLM may see a system popup (Save Password, Allow Notifications)
        that's invisible to the accessibility tree. It responds with
        coordinates to tap.
        """
        # Get screenshot as base64
        screenshot_b64 = None
        try:
            screenshot_b64 = await asyncio.wait_for(
                self.controller.take_screenshot(), timeout=10
            )
        except Exception:
            pass

        prompt = (
            "The screen has 0 interactive elements detected by accessibility tools. "
            "This usually means a SYSTEM POPUP is blocking the screen. "
            "Common iOS system popups:\n"
            "- 'Save Password?' with buttons 'Not Now' (left) and 'Save' (right)\n"
            "- 'Allow Notifications?' with 'Don't Allow' and 'Allow'\n"
            "- Permission dialogs with 'OK' or 'Cancel'\n\n"
            "Based on the screenshot, tell me where to tap to dismiss the popup.\n"
            "Respond with JSON: {\"x\": 100, \"y\": 300, \"reasoning\": \"I see Save Password popup, tapping Not Now\"}\n"
            "If you don't see a popup, respond: {\"x\": 0, \"y\": 0, \"reasoning\": \"No popup visible\"}"
        )

        messages = [
            {"role": "system", "content": "You analyze mobile app screenshots. Respond with JSON only."},
            {"role": "user", "content": []}
        ]

        if screenshot_b64:
            messages[1]["content"] = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                {"type": "text", "text": prompt},
            ]
        else:
            messages[1]["content"] = prompt

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.llm_url}/v1/chat/completions",
                    json={
                        "model": self.llm_model,
                        "messages": messages,
                        "max_tokens": 300,
                        "temperature": 0.1,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                # Strip thinking tags
                if "<think>" in content and "</think>" in content:
                    content = content[content.index("</think>") + len("</think>"):]
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1] if "\n" in content else content
                    content = content.rsplit("```", 1)[0]
                print(f"    [vision] LLM response: {content[:300]}", flush=True)

                if "{" in content:
                    start = content.index("{")
                    end = content.rindex("}") + 1 if "}" in content else len(content)
                    json_str = content[start:end]
                    if not json_str.endswith("}"):
                        json_str += "}"
                    data = json.loads(json_str)
                    x = int(data.get("x", 0))
                    y = int(data.get("y", 0))
                    reasoning = data.get("reasoning", "")
                    print(f"    [vision] Reasoning: {reasoning}", flush=True)
                    if x > 0 and y > 0:
                        return (x, y)
        except Exception as e:
            print(f"    [vision] LLM call failed: {e}", flush=True)

        return None

    # ─────────────────── Actions ───────────────────

    async def _tap(self, element: ElementSnapshot) -> None:
        center = element.get_center()
        if center:
            x, y = center
            try:
                await asyncio.wait_for(self.controller.tap_at(x, y), timeout=10)
            except Exception as e:
                print(f"    ! Tap failed: {e}", flush=True)

    async def _input_text(self, element: ElementSnapshot, text: str) -> None:
        """Clear field and paste new text via clipboard.

        React Native controlled components DON'T see AXe HID keyboard
        input in their state — visually text appears but onChange never
        fires, so on submit the field is "empty". Clipboard paste (Cmd+V)
        DOES fire onChange, so we use: Cmd+A → pbcopy → Cmd+V.
        """
        center = element.get_center()
        if not center:
            return
        x, y = center
        udid = getattr(self.controller, '_udid', '')
        try:
            # 1. Tap to focus the field
            await asyncio.wait_for(self.controller.tap_at(x, y), timeout=5)
            await asyncio.sleep(0.5)

            # 2. Select all existing text (Cmd+A)
            if udid:
                await _axe_cmd_a(udid)
                await asyncio.sleep(0.3)

            if text:
                # 3. Copy new value to simulator clipboard
                proc = await asyncio.create_subprocess_exec(
                    "xcrun", "simctl", "pbcopy", udid,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.communicate(input=text.encode())
                await asyncio.sleep(0.2)

                # 4. Paste (Cmd+V) — this fires React's onChange
                await _axe_cmd_v(udid)
                await asyncio.sleep(0.3)
            else:
                # Empty string = test empty field: delete the selection
                if udid:
                    await _axe_key(udid, "42")  # Backspace
        except Exception as e:
            print(f"    ! Input failed: {e}", flush=True)

    async def _swipe_back(self) -> None:
        """iOS back gesture: swipe from left edge to right."""
        try:
            await asyncio.wait_for(
                self.controller.swipe(10, 400, 250, 400), timeout=5
            )
            await asyncio.sleep(SETTLE_DELAY)
            screen = await self._capture()
            self.screens[screen.screen_id] = screen
            self._current_screen_id = screen.screen_id
            print(f"    ← Swiped back to: {screen.name!r}", flush=True)
        except Exception as e:
            print(f"    ! Swipe back failed: {e}", flush=True)

    async def _go_back(self) -> None:
        """Try swipe-back gesture. Only relaunch as absolute last resort."""
        # Try swipe back 3 times
        for _ in range(3):
            before = self._current_screen_id
            await self._swipe_back()
            if self._current_screen_id != before:
                return  # successfully went back

        # Swipe didn't change screen — relaunch as last resort
        print("    ← Swipe didn't work, relaunching...", flush=True)
        try:
            await self.controller.terminate_app(self.app_bundle_id)
            await asyncio.sleep(1)
            await self.controller.launch_app(self.app_bundle_id)
            await asyncio.sleep(3)
            screen = await self._capture()
            self.screens[screen.screen_id] = screen
            self._current_screen_id = screen.screen_id
        except Exception:
            pass

    # ─────────────────── Capture ───────────────────

    async def _capture(self) -> ScreenNode:
        from explorer.analyzer import analyze_screen
        try:
            elements = await asyncio.wait_for(
                self.controller.get_ui_elements(), timeout=10
            )
            screenshot = None
            try:
                screenshot = await asyncio.wait_for(
                    self.controller.take_screenshot(), timeout=15
                )
            except Exception:
                pass
            return analyze_screen(elements=elements, screenshot_b64=screenshot)
        except Exception as e:
            return ScreenNode(screen_id=f"error_{hash(str(e))}")

    # ─────────────────── Events ───────────────────

    async def _emit(self, event: dict) -> None:
        if self.event_callback is None:
            return
        try:
            result = self.event_callback(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass

    async def _emit_screen(self, node: ScreenNode, *, step: int, is_new: bool = True) -> None:
        # Never use the raw hash as a fallback name — it leaks into the UI as
        # gibberish (e.g. "Обнаружил «4f53cda1»"). analyze_screen now always
        # returns a meaningful name (app_label or "Главный экран"), but if a
        # caller bypasses it we still mask the hash here.
        name = (node.name or "").strip() or "Главный экран"
        await self._emit({
            "type": "screen_discovered",
            "step_idx": step,
            "screen_id_hash": node.screen_id,
            "screen_name": name,
            "is_new": is_new,
            "screenshot_b64": node.screenshot_b64 if is_new else None,
        })

    async def _emit_edge(self, edge: GraphEdge, *, step: int) -> None:
        # Bundle the per-action details into a single dict that maps
        # 1:1 to the backend's edge.action_details_json column. The UI
        # uses these fields when rendering the steps list and the
        # PathFinder result.
        details = {
            "element": edge.action.target_label or edge.action.target_test_id or None,
            "value": edge.action.input_text,
        }
        await self._emit({
            "type": "edge_discovered",
            "step_idx": step,
            "source_screen_hash": edge.source_screen_id,
            "target_screen_hash": edge.target_screen_id,
            "action_type": str(edge.action.action_type),
            "success": edge.source_screen_id != edge.target_screen_id,
            "action_details": details,
        })

    async def _emit_stats(self) -> None:
        await self._emit({
            "type": "stats_update",
            "stats": {
                "screens": len(self.screens),
                "edges": len(self.edges),
                "step": self._step,
                "max_steps": self.max_steps,
            },
        })


async def _axe_key(udid: str, key: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        AXE_BIN, "key", key, "--udid", udid,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await asyncio.wait_for(proc.communicate(), timeout=5)


async def _axe_cmd_a(udid: str) -> None:
    """Press Cmd+A (select all). Modifier 227 = Left Command, Key 4 = 'a'."""
    proc = await asyncio.create_subprocess_exec(
        AXE_BIN, "key-combo",
        "--modifiers", "227", "--key", "4", "--udid", udid,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    await asyncio.wait_for(proc.communicate(), timeout=5)


async def _axe_cmd_v(udid: str) -> None:
    """Press Cmd+V (paste). Modifier 227 = Left Command, Key 25 = 'v'."""
    proc = await asyncio.create_subprocess_exec(
        AXE_BIN, "key-combo",
        "--modifiers", "227", "--key", "25", "--udid", udid,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    await asyncio.wait_for(proc.communicate(), timeout=5)
