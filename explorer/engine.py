"""Main exploration engine: random walk with memory."""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime
from pathlib import Path

from explorer.analyzer import analyze_screen
from explorer.form_filler import FormFiller
from explorer.models import (
    ActionDetail,
    ActionType,
    AppGraph,
    ElementKind,
    ElementSnapshot,
    GraphEdge,
    ScreenNode,
)
from explorer.navigator import (
    SETTLE_DELAY,
    find_nearest_unexplored,
    go_back_to_screen,
)
from explorer.screen_id import compute_screen_id

logger = logging.getLogger("explorer.engine")

# Max consecutive same-screen detections before declaring stuck
MAX_STUCK_COUNT = 10


class ExplorationEngine:
    """
    Systematic app explorer that builds a state machine graph.

    Uses random walk with memory: picks random unexplored interactive
    elements, executes actions, records transitions.
    """

    def __init__(
        self,
        controller,
        app_bundle_id: str,
        output_dir: str = "explorer_output",
        resume_from: str | None = None,
    ):
        self.controller = controller
        self.app_bundle_id = app_bundle_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.form_filler = FormFiller()
        self.current_screen_id: str | None = None
        self._stuck_count = 0

        if resume_from:
            self.graph = AppGraph.load(resume_from)
            logger.info(f"Resumed graph: {self.graph.stats()}")
        else:
            self.graph = AppGraph(app_bundle_id=app_bundle_id)

    async def run(self) -> AppGraph:
        """Run the exploration loop until all screens are fully explored."""
        print(f"\n>>> Starting exploration of {self.app_bundle_id}")
        logger.info(f"Starting exploration of {self.app_bundle_id}")

        # App is already launched by Appium session. Just wait for UI to settle.
        print(">>> Waiting for app to settle (2s)...")
        await asyncio.sleep(2.0)

        print(">>> Capturing initial screen...")
        initial_node = await self._capture_screen()
        self.graph.add_node(initial_node)
        self.current_screen_id = initial_node.screen_id
        self._save_checkpoint()

        print(
            f">>> Initial screen: {initial_node.name!r} "
            f"({len(initial_node.interactive_elements)} interactive elements)"
        )
        for el in initial_node.interactive_elements:
            print(f"    - [{el.kind}] {el.label!r} (test_id={el.test_id})")
        logger.info(
            f"Initial screen: {initial_node.name!r} "
            f"({len(initial_node.interactive_elements)} interactive elements)"
        )

        step = 0
        while True:
            step += 1
            unexplored_screens = self.graph.get_screens_with_unexplored()

            if not unexplored_screens:
                logger.info("All screens fully explored!")
                break

            # Check if current screen has unexplored actions
            unexplored = self.graph.get_unexplored_actions(self.current_screen_id)

            if not unexplored:
                # Navigate to nearest screen with unexplored actions
                target_id = find_nearest_unexplored(
                    self.graph, self.current_screen_id
                )
                if not target_id:
                    logger.info("No reachable unexplored screens. Done.")
                    break

                logger.info(
                    f"[Step {step}] Navigating to screen with unexplored actions: "
                    f"{target_id[:8]}"
                )
                result = await go_back_to_screen(
                    self.controller,
                    self.graph,
                    self.current_screen_id,
                    target_id,
                    self._capture_screen_id,
                )
                if result:
                    self.current_screen_id = result
                else:
                    logger.warning("Navigation failed, trying next screen")
                    continue
                continue

            # Prioritize buttons/switches over text fields
            # (text fields need form-filling logic, buttons are simpler to explore)
            buttons = [e for e in unexplored if e.kind != ElementKind.TEXT_FIELD]
            if buttons:
                element = random.choice(buttons)
            else:
                element = random.choice(unexplored)
            print(
                f"\n>>> [Step {step}] Screen: {self._screen_name()}, "
                f"Action: {element.kind} on {element.label!r}, "
                f"Remaining: {len(unexplored)-1} unexplored on this screen"
            )

            if element.kind == ElementKind.TEXT_FIELD:
                await self._explore_text_field(element)
            else:
                await self._explore_tap(element)

            # Stuck detection
            if self._stuck_count >= MAX_STUCK_COUNT:
                logger.warning(
                    f"Stuck on screen {self.current_screen_id[:8]} "
                    f"for {MAX_STUCK_COUNT} steps. Marking remaining as explored."
                )
                self._mark_screen_fully_explored(self.current_screen_id)
                self._stuck_count = 0

            self._save_checkpoint()

        # Finalize
        self.graph.exploration_end = datetime.now()
        self._save_checkpoint()
        logger.info(f"Exploration complete. {self.graph.stats()}")
        return self.graph

    async def _explore_tap(self, element: ElementSnapshot) -> None:
        """Execute a tap action and record the transition."""
        center = element.get_center()
        if not center:
            print(f"    ! No coordinates for {element.label!r}, skipping")
            self._record_self_loop(element, ActionType.TAP)
            return

        x, y = center
        before_screen = self.current_screen_id

        print(f"    Tapping {element.label!r} at ({x}, {y})...", flush=True)
        try:
            result = await asyncio.wait_for(
                self.controller.tap_at(x, y), timeout=10
            )
            if result.error:
                print(f"    ! Tap failed: {result.error}", flush=True)
                self._record_self_loop(element, ActionType.TAP)
                return
        except asyncio.TimeoutError:
            print(f"    ! Tap timed out", flush=True)
            self._record_self_loop(element, ActionType.TAP)
            return
        except Exception as e:
            print(f"    ! Tap exception: {e}", flush=True)
            self._record_self_loop(element, ActionType.TAP)
            return

        print(f"    Waiting {SETTLE_DELAY}s for UI to settle...")
        await asyncio.sleep(SETTLE_DELAY)

        print(f"    Capturing screen...")
        new_node = await self._capture_screen()
        is_new = self.graph.add_node(new_node)

        # Record edge
        action = ActionDetail(
            action_type=ActionType.TAP,
            target_label=element.label,
            target_test_id=element.test_id,
            target_frame=element.frame,
        )
        self.graph.add_edge(GraphEdge(
            source_screen_id=before_screen,
            target_screen_id=new_node.screen_id,
            action=action,
        ))

        if new_node.screen_id == before_screen:
            self._stuck_count += 1
            print(f"    -> Same screen (self-loop)")
        else:
            self._stuck_count = 0
            print(
                f"    -> {'NEW' if is_new else 'Known'} screen: "
                f"{new_node.name!r} ({new_node.screen_id[:8]})"
            )

        self.current_screen_id = new_node.screen_id

    async def _explore_text_field(self, element: ElementSnapshot) -> None:
        """Explore a text field with PBT variants."""
        before_screen = self.current_screen_id

        # Get all text fields on this screen (for form group detection)
        node = self.graph.nodes[self.current_screen_id]
        text_fields = [
            el for el in node.interactive_elements
            if el.kind == ElementKind.TEXT_FIELD
        ]

        # For form groups: fill all fields with valid data first (happy path)
        if len(text_fields) > 1:
            await self._fill_form_happy_path(text_fields, element)
        else:
            await self._fill_single_field(element)

    async def _fill_form_happy_path(
        self,
        all_fields: list[ElementSnapshot],
        trigger_field: ElementSnapshot,
    ) -> None:
        """Fill all form fields with valid data, then try variations on trigger field."""
        before_screen = self.current_screen_id

        # Relaunch app to get clean empty fields
        print("      Relaunching app for clean form...", flush=True)
        try:
            await self.controller.terminate_app(self.app_bundle_id)
            await asyncio.sleep(1)
            await self.controller.launch_app(self.app_bundle_id)
            await asyncio.sleep(2)
        except Exception as e:
            print(f"      ! Relaunch failed: {e}", flush=True)

        # First: fill ALL fields with valid values
        for field in all_fields:
            valid_value = self.form_filler.get_valid_value_for(field)
            await self._type_into_field(field, valid_value)

        # Dismiss keyboard
        try:
            await self.controller.press_enter()
            await asyncio.sleep(0.5)
        except Exception:
            pass

        # After filling, capture screen to find submit button
        post_fill_node = await self._capture_screen()
        # Find a likely submit button on the post-fill screen
        submit_button = self._find_submit_button(post_fill_node)
        if submit_button:
            print(f"      Tapping submit button: {submit_button.label!r}", flush=True)
            await self._tap_element_directly(submit_button)
            await asyncio.sleep(2)  # Wait for navigation

        # Now record the actual result state
        new_node = await self._capture_screen()
        self.graph.add_node(new_node)

        # Record edge for the trigger field with valid input
        action = ActionDetail(
            action_type=ActionType.INPUT,
            target_label=trigger_field.label,
            target_test_id=trigger_field.test_id,
            target_frame=trigger_field.frame,
            input_text=self.form_filler.get_valid_value_for(trigger_field),
            input_category="valid",
        )
        self.graph.add_edge(GraphEdge(
            source_screen_id=before_screen,
            target_screen_id=new_node.screen_id,
            action=action,
        ))

        self.current_screen_id = new_node.screen_id

        # If we're still on the same screen, now also mark the trigger field
        # "valid" variant as explored. The remaining variants will be explored
        # in subsequent iterations.
        self.form_filler._tried_variants.setdefault(trigger_field.uid(), set())
        self.form_filler._tried_variants[trigger_field.uid()].add("valid")

    async def _fill_single_field(self, element: ElementSnapshot) -> None:
        """Fill a single field with the next PBT variant."""
        variant = self.form_filler.get_next_variant(element)
        if not variant:
            logger.info(f"All variants exhausted for {element.label}")
            return

        value, category = variant
        before_screen = self.current_screen_id

        await self._type_into_field(element, value)

        # Dismiss keyboard
        try:
            await self.controller.press_enter()
            await asyncio.sleep(0.5)
        except Exception:
            pass

        await asyncio.sleep(SETTLE_DELAY)
        new_node = await self._capture_screen()
        self.graph.add_node(new_node)

        action = ActionDetail(
            action_type=ActionType.INPUT,
            target_label=element.label,
            target_test_id=element.test_id,
            target_frame=element.frame,
            input_text=value,
            input_category=category,
        )
        self.graph.add_edge(GraphEdge(
            source_screen_id=before_screen,
            target_screen_id=new_node.screen_id,
            action=action,
        ))

        self.current_screen_id = new_node.screen_id

    SUBMIT_KEYWORDS = (
        "войти", "вход", "login", "log in", "sign in",
        "submit", "send", "ok", "continue", "next", "далее",
        "регистрация", "зарегистрировать", "register", "sign up",
        "сохранить", "save", "apply",
    )

    def _find_submit_button(self, node: ScreenNode) -> ElementSnapshot | None:
        """Find a likely submit/login button on the screen."""
        for el in node.interactive_elements:
            if el.kind != ElementKind.BUTTON:
                continue
            label_lc = (el.label or "").lower().strip()
            if any(kw in label_lc for kw in self.SUBMIT_KEYWORDS):
                return el
            test_id_lc = (el.test_id or "").lower()
            if any(kw in test_id_lc for kw in ("login", "submit", "signin", "войти")):
                return el
        return None

    async def _tap_element_directly(self, element: ElementSnapshot) -> bool:
        """Tap an element using the best available method (testID > coords)."""
        try:
            if element.test_id and hasattr(self.controller, 'tap_by_id'):
                result = await self.controller.tap_by_id(element.test_id)
                if not result.error:
                    return True
            center = element.get_center()
            if center:
                x, y = center
                result = await self.controller.tap_at(x, y)
                return result.error is None
        except Exception as e:
            print(f"      ! Tap element failed: {e}", flush=True)
        return False

    async def _type_into_field(
        self, element: ElementSnapshot, text: str
    ) -> None:
        """Set text into a TextInput field. Uses CDP for RN apps when available."""
        try:
            print(f"      Typing into {element.test_id or element.label!r}: {text!r:.40}", flush=True)

            # If controller supports set_text_in_field (AXe + CDP), use it
            if hasattr(self.controller, 'set_text_in_field'):
                ok = await asyncio.wait_for(
                    self.controller.set_text_in_field(
                        test_id=element.test_id,
                        label=element.label,
                        text=text,
                    ),
                    timeout=10,
                )
                if ok:
                    await asyncio.sleep(0.2)
                    return

            # Fallback: tap + type via coordinates
            center = element.get_center()
            if not center:
                return
            x, y = center
            await asyncio.wait_for(self.controller.tap_at(x, y), timeout=5)
            await asyncio.sleep(0.5)
            if text:
                await asyncio.wait_for(
                    self.controller.type_text(text), timeout=10
                )
                await asyncio.sleep(0.3)
        except asyncio.TimeoutError:
            print(f"      ! Timed out typing into {element.label}", flush=True)
        except Exception as e:
            print(f"      ! Failed to type into {element.label}: {e}", flush=True)

    async def _capture_screen(self) -> ScreenNode:
        """Capture the current screen state from the device."""
        import time
        try:
            t0 = time.monotonic()

            print(f"      [capture] Getting UI elements...", flush=True)
            elements = await self.controller.get_ui_elements()
            t1 = time.monotonic()
            print(f"      [capture] Found {len(elements)} elements ({t1-t0:.1f}s)", flush=True)

            # Use cached screenshot from vision client if available
            cached = getattr(self.controller, '_last_screenshot_b64', None)
            if cached:
                screenshot = cached
                self.controller._last_screenshot_b64 = None
                t2 = time.monotonic()
                print(f"      [capture] Using cached screenshot", flush=True)
            else:
                print(f"      [capture] Taking screenshot...", flush=True)
                screenshot = await self.controller.take_screenshot()
                t2 = time.monotonic()
                print(f"      [capture] Screenshot done ({t2-t1:.1f}s)", flush=True)

            node = analyze_screen(
                elements=elements,
                screenshot_b64=screenshot,
            )
            print(f"      [capture] Analyzed: {node.name!r}, {len(node.interactive_elements)} interactive ({t2-t0:.1f}s total)", flush=True)
            return node
        except Exception as e:
            import traceback
            print(f"      [capture] ERROR: {e}", flush=True)
            print(f"      [capture] Traceback: {traceback.format_exc()}", flush=True)
            logger.error(f"Failed to capture screen: {e}")
            return ScreenNode(screen_id="error_" + str(hash(str(e)))[:8])

    async def _capture_screen_id(self) -> tuple[str, list[dict]]:
        """Capture current screen and return (screen_id, raw_elements)."""
        try:
            node = await self._capture_screen()
            self.graph.add_node(node)
            self.current_screen_id = node.screen_id
            return (node.screen_id, [])
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return (self.current_screen_id or "unknown", [])

    def _record_self_loop(
        self, element: ElementSnapshot, action_type: ActionType
    ) -> None:
        """Record a self-loop edge (action that didn't change the screen)."""
        if not self.current_screen_id:
            return
        action = ActionDetail(
            action_type=action_type,
            target_label=element.label,
            target_test_id=element.test_id,
            target_frame=element.frame,
        )
        self.graph.add_edge(GraphEdge(
            source_screen_id=self.current_screen_id,
            target_screen_id=self.current_screen_id,
            action=action,
            success=False,
            notes="Action failed or did not change screen",
        ))

    def _mark_screen_fully_explored(self, screen_id: str) -> None:
        """Mark all remaining elements on a screen as explored via self-loops."""
        unexplored = self.graph.get_unexplored_actions(screen_id)
        for el in unexplored:
            self._record_self_loop(el, ActionType.TAP)

    def _screen_name(self) -> str:
        """Get the name of the current screen."""
        if self.current_screen_id and self.current_screen_id in self.graph.nodes:
            return self.graph.nodes[self.current_screen_id].name or self.current_screen_id[:8]
        return "unknown"

    def _save_checkpoint(self) -> None:
        """Save current graph state to checkpoint file."""
        checkpoint_path = self.output_dir / "checkpoint.json"
        self.graph.save(checkpoint_path)
