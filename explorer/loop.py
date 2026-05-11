"""Clean exploration loop (Engine V2).

This replaces the monolithic ExplorationEngine with a simple state machine:

    observe → decide → dismiss popup → execute → observe → record → learn → verify

No special-casing for text fields, no form filling heuristics, no navigator.
Each step is one action on one element. The strategy decides what to do;
the loop just executes and records.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from explorer.models import (
    ActionDetail,
    ActionType,
    ElementKind,
    ElementSnapshot,
    GraphEdge,
    ScreenNode,
)
from explorer.strategy import (
    ExplorationStrategy,
    PBTInputGenerator,
    _element_key,
    _transition_key,
)

logger = logging.getLogger("explorer.loop")

# How long to wait for UI to settle after an action
SETTLE_DELAY = 1.5

# System popup dismiss labels (tap to clear)
DISMISS_LABELS = frozenset({
    "Not Now", "Don't Allow", "Cancel", "Dismiss", "Close",
    "OK", "Later", "Skip", "No Thanks", "Deny",
    "Не сейчас", "Не разрешать", "Отмена", "Закрыть",
    "Позже", "Пропустить", "Отклонить",
})

# Max back presses before relaunch
MAX_BACK_ATTEMPTS = 3

EventCallback = Callable[[dict], None | Awaitable[None]]

# AXe CLI binary path — resolved by axe_client (env var → PATH →
# Homebrew fallback). Single source so this legacy loop honours the
# same TA_AXE_BIN override as the LLM-driven path.
from explorer.axe_client import AXE as AXE_BIN  # noqa: E402


async def _axe_key_combo(udid: str, key: str, modifier: str) -> None:
    """Press a key with modifier via AXe (e.g. Cmd+A for select all)."""
    proc = await asyncio.create_subprocess_exec(
        AXE_BIN, "key-combo", key, "--modifier", modifier, "--udid", udid,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await asyncio.wait_for(proc.communicate(), timeout=5)


async def _axe_key(udid: str, key: str) -> None:
    """Press a single key via AXe (e.g. delete)."""
    proc = await asyncio.create_subprocess_exec(
        AXE_BIN, "key", key, "--udid", udid,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await asyncio.wait_for(proc.communicate(), timeout=5)


class ExplorationLoop:
    """Clean, minimal exploration loop.

    Arguments:
        controller:     AXeExplorerClient or any compatible controller
        strategy:       ExplorationStrategy (MC, PUCT, LLM, Hybrid)
        app_bundle_id:  iOS/Android bundle/package identifier
        max_steps:      stop after this many steps
        event_callback: optional callback for live progress events
    """

    def __init__(
        self,
        controller: Any,
        strategy: ExplorationStrategy,
        app_bundle_id: str,
        max_steps: int = 200,
        event_callback: EventCallback | None = None,
    ) -> None:
        self.controller = controller
        self.strategy = strategy
        self.app_bundle_id = app_bundle_id
        self.max_steps = max_steps
        self.event_callback = event_callback

        # Graph state
        self.screens: dict[str, ScreenNode] = {}
        self.edges: list[GraphEdge] = []
        self.visited_transitions: set[str] = set()

        # Data layer: full record of every action taken
        self.action_log: list[dict] = []

        # PBT input generator — tracks tried variants per field
        self.pbt = PBTInputGenerator()

        self._step = 0
        self._stuck_count = 0
        self._current_screen_id: str = ""

    async def run(self) -> dict:
        """Run the two-phase exploration loop.

        Phase A (BFS Discovery): Find all reachable screens by trying
        each button once (happy path for forms). Stop when no new screens
        are found.

        Phase B (DFS Testing): For each screen with text fields, test
        all PBT variants. Relaunch + navigate before each variant to
        get clean fields.
        """
        await self._emit({"type": "status_change", "new_status": "running"})

        print(">>> Waiting for app to settle (2s)...", flush=True)
        await asyncio.sleep(2.0)

        print(">>> Capturing initial screen...", flush=True)
        screen = await self._capture()
        self._current_screen_id = screen.screen_id
        self.screens[screen.screen_id] = screen
        self._start_screen_id = screen.screen_id

        print(f">>> Initial: {screen.name!r} ({len(screen.interactive_elements)} elements)", flush=True)
        await self._emit_screen(screen, step=0)
        await self._emit_stats()

        # ─── PHASE A: BFS Discovery (happy path) ───
        print("\n══════ PHASE A: Discovery (BFS) ══════", flush=True)
        await self._phase_discovery()

        # ─── PHASE B: DFS Testing (PBT on each screen) ───
        if self._step < self.max_steps:
            print("\n══════ PHASE B: Testing (PBT) ══════", flush=True)
            await self._phase_testing()

        # Done
        await self._emit_stats()
        print(
            f"\n>>> Complete: {len(self.screens)} screens, "
            f"{len(self.edges)} edges in {self._step} steps",
            flush=True,
        )
        return {
            "screens": len(self.screens),
            "edges": len(self.edges),
            "steps": self._step,
        }

    # ═══════════════════════════ PHASE A: Discovery ═══════════════════

    async def _phase_discovery(self) -> None:
        """DFS: on each screen, try untried elements. When an action leads
        to a NEW screen — stay there and explore it immediately (depth-first).
        When all elements on a screen are tried — go back. When back fails —
        relaunch.

        For forms (screens with text fields): fill valid data + submit as
        ONE action. If it leads to a new screen — explore that screen.

        This naturally builds a depth-first tree through the app."""
        back_stack: list[str] = []  # screens we can go back to

        # Anti-stuck guard: consecutive relaunches without any new
        # element interaction mean the loop has nothing to do but
        # cycle through "exhausted screen → relaunch → same exhausted
        # screen". Previously the while-condition only watched
        # ``self._step``, which is only incremented after a successful
        # action — so a fully-exhausted root screen made the loop
        # spin forever (audit PER-104, finding #1). Cap consecutive
        # relaunches and exit cleanly so the run posts a terminal
        # status instead of hanging.
        consecutive_relaunches = 0
        _MAX_CONSECUTIVE_RELAUNCHES = 3

        while self._step < self.max_steps:
            # Re-capture current screen (may have changed after back/relaunch)
            screen = await self._capture()
            self.screens[screen.screen_id] = screen
            self._current_screen_id = screen.screen_id
            elements = screen.interactive_elements

            # Auto-dismiss system popups
            if await self._try_dismiss_popup(elements):
                continue

            if not elements:
                # No elements — try going back
                if not await self._try_go_back():
                    break
                continue

            # Find untried elements on THIS screen
            text_fields = [e for e in elements if e.kind == ElementKind.TEXT_FIELD]
            buttons = [e for e in elements if e.kind != ElementKind.TEXT_FIELD]

            # Check if form (text fields) has been tried with valid data
            form_key = f"form_valid:{screen.screen_id}"
            form_tried = form_key in self.visited_transitions

            # Priority 1: If form not yet submitted with valid data — do it
            if text_fields and not form_tried:
                self._step += 1
                consecutive_relaunches = 0  # made progress on this screen
                self.visited_transitions.add(form_key)
                print(f"\n>>> [Step {self._step}] {screen.name!r}: form (valid data)", flush=True)
                before = self._current_screen_id

                for field in text_fields:
                    valid = self.pbt.valid_input(field)
                    await self._execute(field, ActionType.INPUT, valid)

                await self._try_submit(elements)

                new_screen = await self._capture()
                is_new = new_screen.screen_id not in self.screens
                self.screens[new_screen.screen_id] = new_screen
                self._current_screen_id = new_screen.screen_id

                await self._record(before, text_fields[0], ActionType.INPUT,
                                   self.pbt.valid_input(text_fields[0]), "valid",
                                   new_screen, is_new)

                if is_new:
                    back_stack.append(before)
                    print(f"    → NEW: {new_screen.name!r} ← exploring this now!", flush=True)
                    continue  # stay on new screen, explore it next iteration
                else:
                    print(f"    → Same/known: {new_screen.name!r}", flush=True)
                    # Form didn't lead anywhere new — continue with buttons
                    continue

            # Priority 2: Try untried buttons
            untried_buttons = [
                b for b in buttons
                if _transition_key(screen.screen_id, b) not in self.visited_transitions
            ]

            if untried_buttons:
                # Pick one (strategy decides which)
                btn = self.strategy.select_element(
                    screen, untried_buttons, self.visited_transitions
                )
                if btn is None:
                    btn = untried_buttons[0]

                self._step += 1
                consecutive_relaunches = 0  # made progress on this screen
                print(f"\n>>> [Step {self._step}] {screen.name!r}: tap {btn.label!r}", flush=True)
                before = self._current_screen_id

                await self._execute(btn, ActionType.TAP, None)

                new_screen = await self._capture()
                is_new = new_screen.screen_id not in self.screens
                self.screens[new_screen.screen_id] = new_screen
                self._current_screen_id = new_screen.screen_id

                await self._record(before, btn, ActionType.TAP, None, None,
                                   new_screen, is_new)

                if is_new:
                    back_stack.append(before)
                    print(f"    → NEW: {new_screen.name!r} ← exploring this now!", flush=True)
                    continue  # DFS: go deeper
                elif new_screen.screen_id != before:
                    print(f"    → Known: {new_screen.name!r}", flush=True)
                else:
                    print(f"    → Self-loop", flush=True)
                continue

            # Priority 3: All elements tried on this screen — go back
            print(f"\n>>> All elements tried on {screen.name!r}, going back", flush=True)
            if back_stack:
                target = back_stack[-1]
                went_back = await self._try_go_back()
                if went_back:
                    new_s = await self._capture()
                    self.screens[new_s.screen_id] = new_s
                    self._current_screen_id = new_s.screen_id
                    if new_s.screen_id == target:
                        back_stack.pop()
                    continue

            # Can't go back — relaunch
            print(">>> Relaunching app...", flush=True)
            await self._relaunch()
            back_stack.clear()
            consecutive_relaunches += 1
            if consecutive_relaunches >= _MAX_CONSECUTIVE_RELAUNCHES:
                # Relaunched several times without making forward
                # progress — the root screen has nothing untried.
                # Exit so the run posts terminal status; staying in
                # the loop would burn LLM budget on the same screen
                # forever and never reach `status_change=completed`.
                print(
                    f">>> {_MAX_CONSECUTIVE_RELAUNCHES} consecutive "
                    "relaunches without progress — discovery exhausted.",
                    flush=True,
                )
                break

        # Note: ``consecutive_relaunches`` is reset to 0 every time the
        # main loop successfully takes a new action (lines around 207
        # and 250 increment self._step). The break above only fires
        # when the loop has done nothing else but relaunch in a row.

        print(f"\n>>> Discovery complete: {len(self.screens)} screens", flush=True)

    # ═══════════════════════════ PHASE B: Testing ═══════════════════

    async def _phase_testing(self) -> None:
        """For each screen with text fields: try all PBT variants.
        Relaunch + navigate before each variant for clean fields."""
        # Collect screens that have text fields
        form_screens = [
            (sid, s) for sid, s in self.screens.items()
            if any(e.kind == ElementKind.TEXT_FIELD for e in s.interactive_elements)
        ]

        if not form_screens:
            print(">>> No forms to test", flush=True)
            return

        for screen_id, screen in form_screens:
            text_fields = [e for e in screen.interactive_elements
                           if e.kind == ElementKind.TEXT_FIELD]

            for field in text_fields:
                while self._step < self.max_steps:
                    variant = self.pbt.next_input(field)
                    if variant is None:
                        break  # all variants tried for this field

                    value, category = variant
                    if category == "valid":
                        continue  # already tested in Phase A

                    self._step += 1
                    print(
                        f"\n>>> [Step {self._step}] PBT {screen.name!r}: "
                        f"{field.label!r} [{category}]={value!r}",
                        flush=True,
                    )

                    # Relaunch for clean fields
                    await self._relaunch()

                    # Navigate to the form screen
                    if not await self._navigate_to(screen_id):
                        print(f"    ! Can't reach form, skipping", flush=True)
                        break

                    # Re-capture to get fresh elements
                    fresh = await self._capture()
                    fresh_fields = [e for e in fresh.interactive_elements
                                    if e.kind == ElementKind.TEXT_FIELD]

                    before = self._current_screen_id

                    # Fill OTHER fields with valid data
                    for f in fresh_fields:
                        if _element_key(f) != _element_key(field):
                            await self._execute(f, ActionType.INPUT,
                                                self.pbt.valid_input(f))

                    # Fill target field with PBT variant
                    target = next((f for f in fresh_fields
                                   if _element_key(f) == _element_key(field)), None)
                    if target:
                        await self._execute(target, ActionType.INPUT, value)
                    else:
                        print(f"    ! Field {field.label!r} not found after nav", flush=True)
                        break

                    # Submit
                    await self._try_submit(fresh.interactive_elements)

                    # Record result
                    result = await self._capture()
                    is_new = result.screen_id not in self.screens
                    self.screens[result.screen_id] = result
                    self._current_screen_id = result.screen_id

                    await self._record(before, field, ActionType.INPUT, value,
                                       category, result, is_new)

                    if is_new:
                        print(f"    → NEW: {result.name!r} (PBT found new path!)", flush=True)
                        await self._emit_screen(result, step=self._step)
                    else:
                        print(f"    → {result.name!r}", flush=True)

    # ═══════════════════════════ Helpers ═══════════════════

    async def _record(
        self,
        source_id: str,
        element: ElementSnapshot,
        action: ActionType,
        data: str | None,
        data_category: str | None,
        target_screen: ScreenNode,
        is_new: bool,
    ) -> None:
        """Record a transition to the graph + data layer + emit events.

        Now async because event emission is awaited (PER-104 #7) —
        callers must use `await self._record(...)`.
        """
        tkey = _transition_key(source_id, element)
        is_new_transition = tkey not in self.visited_transitions
        self.visited_transitions.add(tkey)

        edge = GraphEdge(
            source_screen_id=source_id,
            target_screen_id=target_screen.screen_id,
            action=ActionDetail(
                action_type=action,
                target_label=element.label,
                target_test_id=element.test_id,
                target_frame=element.frame,
                input_text=data,
                input_category=data_category,
            ),
        )
        self.edges.append(edge)

        self.action_log.append({
            "step": self._step,
            "source_screen": source_id,
            "target_screen": target_screen.screen_id,
            "element_kind": element.kind.value,
            "element_label": element.label,
            "element_test_id": element.test_id,
            "action": action.value,
            "data": data,
            "data_category": data_category,
            "is_new_screen": is_new,
            "is_new_transition": is_new_transition,
        })

        self.strategy.update(
            source_id, element, action,
            target_screen.screen_id, is_new,
        )

        # Emit. Audit PER-104 #7: the previous code used
        # asyncio.ensure_future and never awaited the resulting tasks
        # — callback failures vanished and the run could finish
        # before backend events flushed (race between run-end status
        # and the last edge event). Awaiting each emit blocks the
        # loop for the duration of the HTTP POST, but the events are
        # small and the backend is local; the predictability is
        # worth more than the few-ms parallelism we lose.
        if is_new:
            await self._emit_screen(target_screen, step=self._step)
        await self._emit_edge(edge, step=self._step)
        if self._step % 5 == 0 or is_new:
            await self._emit_stats()

    async def _navigate_to(self, target_id: str) -> bool:
        """Try to navigate to a specific screen. Simple strategy:
        try back, if fail — relaunch and hope we land there."""
        if self._current_screen_id == target_id:
            return True

        # Try going back (up to 3 times)
        for _ in range(3):
            if await self._try_go_back():
                if self._current_screen_id == target_id:
                    return True

        # Relaunch — most apps return to their start screen. Audit
        # PER-104 #8: returning True when we land on the start screen
        # (instead of target) silently lied to callers — Phase B then
        # ran PBT variants against the wrong screen, marking
        # validations as "passed on screen X" when they were actually
        # observed on Y. Now: report success only if we actually
        # reached the requested target.
        await self._relaunch()
        return self._current_screen_id == target_id

    async def _relaunch(self) -> None:
        """Terminate + relaunch the app for clean state."""
        try:
            await self.controller.terminate_app(self.app_bundle_id)
            await asyncio.sleep(1)
            await self.controller.launch_app(self.app_bundle_id)
            await asyncio.sleep(2)
            screen = await self._capture()
            self.screens[screen.screen_id] = screen
            self._current_screen_id = screen.screen_id
        except Exception as e:
            print(f"    ! Relaunch failed: {e}", flush=True)

    # ─────────────────────────── execution ──

    async def _execute(
        self, element: ElementSnapshot, action: ActionType, data: str | None
    ) -> None:
        """Execute a single action on an element."""
        if action == ActionType.INPUT and data is not None:
            # For text fields: tap → Cmd+A → Delete → type new value
            center = element.get_center()
            if center:
                x, y = center
                try:
                    # 1. Tap to focus the field
                    await asyncio.wait_for(
                        self.controller.tap_at(x, y), timeout=5
                    )
                    await asyncio.sleep(0.3)

                    # 2. Select all + Delete via AXe key commands
                    #    This clears any existing text in the field
                    udid = getattr(self.controller, '_udid', '')
                    if udid:
                        # Cmd+A = select all
                        await _axe_key_combo(udid, "a", "command")
                        await asyncio.sleep(0.2)
                        # Delete = remove selected
                        await _axe_key(udid, "delete")
                        await asyncio.sleep(0.2)

                    # 3. Type new value (skip for empty string — testing empty field)
                    if data:
                        await asyncio.wait_for(
                            self.controller.type_text(data), timeout=10
                        )

                except (asyncio.TimeoutError, Exception) as e:
                    print(f"    ! Input failed: {e}", flush=True)
        elif action == ActionType.TAP:
            center = element.get_center()
            if center:
                x, y = center
                try:
                    result = await asyncio.wait_for(
                        self.controller.tap_at(x, y), timeout=10
                    )
                    if hasattr(result, "error") and result.error:
                        print(f"    ! Tap failed: {result.error}", flush=True)
                except (asyncio.TimeoutError, Exception) as e:
                    print(f"    ! Tap failed: {e}", flush=True)

        await asyncio.sleep(SETTLE_DELAY)

    # ─────────────────────────── form submit ──

    SUBMIT_KEYWORDS = (
        "войти", "вход", "login", "log in", "sign in",
        "submit", "send", "ok", "continue", "next", "далее",
        "регистрация", "зарегистрировать", "register", "sign up",
        "сохранить", "save", "apply", "отправить",
    )

    async def _try_submit(self, elements: list[ElementSnapshot]) -> None:
        """After filling a form, try to tap the submit button."""
        # Re-capture to see current state (keyboard may have changed layout)
        try:
            fresh = await self._capture()
            search_elements = fresh.interactive_elements
        except Exception:
            search_elements = elements

        for el in search_elements:
            if el.kind != ElementKind.BUTTON:
                continue
            label_lc = (el.label or "").lower().strip()
            test_id_lc = (el.test_id or "").lower()
            if any(kw in label_lc for kw in self.SUBMIT_KEYWORDS) or \
               any(kw in test_id_lc for kw in ("login", "submit", "signin")):
                center = el.get_center()
                if center:
                    x, y = center
                    print(f"    [submit] Tapping '{el.label}' at ({x}, {y})", flush=True)
                    try:
                        await asyncio.wait_for(
                            self.controller.tap_at(x, y), timeout=5
                        )
                        await asyncio.sleep(SETTLE_DELAY)
                    except Exception:
                        pass
                    return

    # ─────────────────────────── popup dismiss ──

    async def _try_dismiss_popup(self, elements: list[ElementSnapshot]) -> bool:
        """If a system popup is visible, dismiss it and return True."""
        for el in elements:
            label = (el.label or "").strip()
            if label in DISMISS_LABELS:
                center = el.get_center()
                if center:
                    x, y = center
                    print(f"    [auto-dismiss] Tapping '{label}' at ({x}, {y})", flush=True)
                    try:
                        await asyncio.wait_for(
                            self.controller.tap_at(x, y), timeout=5
                        )
                        await asyncio.sleep(1.0)
                        new_screen = await self._capture()
                        self.screens[new_screen.screen_id] = new_screen
                        self._current_screen_id = new_screen.screen_id
                    except Exception as e:
                        print(f"    [auto-dismiss] Failed: {e}", flush=True)
                    return True
        return False

    # ─────────────────────────── navigation ──

    async def _try_go_back(self) -> bool:
        """Try pressing back (swipe or button). If fails, relaunch."""
        for attempt in range(MAX_BACK_ATTEMPTS):
            try:
                # Swipe right from left edge (iOS back gesture)
                await asyncio.wait_for(
                    self.controller.swipe(0, 400, 200, 400), timeout=5
                )
                await asyncio.sleep(SETTLE_DELAY)

                new_screen = await self._capture()
                if new_screen.screen_id != self._current_screen_id:
                    self.screens[new_screen.screen_id] = new_screen
                    self._current_screen_id = new_screen.screen_id
                    print(f"    ← Back to: {new_screen.name!r}", flush=True)
                    return True
            except Exception:
                pass

        # Relaunch as last resort
        print("    ← Relaunching app...", flush=True)
        try:
            await self.controller.terminate_app(self.app_bundle_id)
            await asyncio.sleep(1)
            await self.controller.launch_app(self.app_bundle_id)
            await asyncio.sleep(3)

            new_screen = await self._capture()
            self.screens[new_screen.screen_id] = new_screen
            self._current_screen_id = new_screen.screen_id
            print(f"    ← Relaunched to: {new_screen.name!r}", flush=True)
            return True
        except Exception as e:
            print(f"    ! Relaunch failed: {e}", flush=True)
            return False

    # ─────────────────────────── capture ──

    async def _capture(self) -> ScreenNode:
        """Capture current screen and return a ScreenNode."""
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

            return analyze_screen(
                elements=elements,
                screenshot_b64=screenshot,
            )
        except Exception as e:
            logger.error("Capture failed: %s", e)
            return ScreenNode(screen_id=f"error_{hash(str(e))}")

    # ─────────────────────────── events ──

    async def _emit(self, event: dict) -> None:
        if self.event_callback is None:
            return
        try:
            result = self.event_callback(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass

    async def _emit_screen(self, node: ScreenNode, *, step: int) -> None:
        await self._emit({
            "type": "screen_discovered",
            "step_idx": step,
            "screen_id_hash": node.screen_id,
            "screen_name": node.name or node.screen_id[:8],
            "screenshot_path": None,
        })

    async def _emit_edge(self, edge: GraphEdge, *, step: int) -> None:
        await self._emit({
            "type": "edge_discovered",
            "step_idx": step,
            "source_screen_hash": edge.source_screen_id,
            "target_screen_hash": edge.target_screen_id,
            "action_type": str(edge.action.action_type),
            "success": edge.source_screen_id != edge.target_screen_id,
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
