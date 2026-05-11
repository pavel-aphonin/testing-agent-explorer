"""Main exploration engine: PUCT-driven local selection + Go-Explore frontier.

The engine plays the role of an "online MCTS player" against the simulator.
On every step it:

    1. Asks the PUCT selector which unexplored element on the current screen
       has the highest score (Q + c_puct * prior * sqrt(N) / (1 + N(s,a))).
    2. Executes that action via the controller.
    3. Captures the resulting screen and updates the graph.
    4. Backs up the PUCT statistics with reward = 1 if the action discovered
       a brand-new screen, else 0.
    5. Tells the Go-Explore archive about the new (visit_count, unexplored)
       state of both the source and target screens.

When the current screen has no unexplored actions left, the engine queries
the archive for the highest-scoring *frontier* state anywhere in the app
and uses the navigator to walk back there. This is the global counterpart
to PUCT's local greediness.

The mode (MC, AI, Hybrid) is just a different ModeConfig — the loop body
is identical. AI/Hybrid runs may pass a `prior_provider` callable that the
engine consults whenever a new screen is registered, to seed PUCT priors
from an LLM. MC runs leave it None and PUCT falls back to uniform priors.

The engine also accepts an `event_callback` for live progress streaming.
The worker passes its own callback that POSTs each event to the backend's
internal API; CLI / tests can pass any callable or None.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

from explorer.analyzer import analyze_screen
from explorer.form_filler import FormFiller
from explorer.go_explore import GoExploreArchive
from explorer.mcts import PUCTSelector
from explorer.models import (
    ActionDetail,
    ActionType,
    AppGraph,
    ElementKind,
    ElementSnapshot,
    GraphEdge,
    ScreenNode,
)
from explorer.modes import ExplorationMode, ModeConfig, get_mode_config
from explorer.navigator import (
    SETTLE_DELAY,
    find_nearest_unexplored,
    go_back_to_screen,
)
from explorer.screen_id import compute_screen_id

logger = logging.getLogger("explorer.engine")

# Max consecutive same-screen detections before declaring stuck.
# Was 10, but that wastes too many steps on self-loops. 3 is enough:
# by the third self-loop, the action is clearly not leading anywhere.
MAX_STUCK_COUNT = 3

# Type aliases for the optional engine callbacks.
PriorProvider = Callable[[ScreenNode], "dict[str, float] | Awaitable[dict[str, float]]"]
EventCallback = Callable[[dict], "None | Awaitable[None]"]


def action_id_for(element: ElementSnapshot) -> str:
    """Stable PUCT action identifier for an element on a screen.

    Matches the signature format that AppGraph.get_unexplored_actions uses
    internally so that "is this action explored?" agrees between the graph
    edge log and the PUCT visit table.
    """
    frame_key = ""
    if element.frame:
        frame_key = (
            f"{int(element.frame.get('x', 0))},"
            f"{int(element.frame.get('y', 0))}"
        )
    return f"tap|{element.label or ''}|{element.test_id or ''}|{frame_key}"


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
        mode: ExplorationMode | str = ExplorationMode.HYBRID,
        max_steps: int = 200,
        prior_provider: PriorProvider | None = None,
        event_callback: EventCallback | None = None,
        test_data: dict[str, str] | None = None,
    ):
        self.controller = controller
        self.app_bundle_id = app_bundle_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mode_config: ModeConfig = get_mode_config(mode)
        self.max_steps = max_steps
        self.prior_provider = prior_provider
        self.event_callback = event_callback

        # ``test_data`` flows through the form filler as the highest-
        # priority source for happy-path values — see PER-20. Empty
        # dict keeps the legacy "use BUILTIN_VARIANTS" behaviour.
        self.form_filler = FormFiller(test_data=test_data)
        self.current_screen_id: str | None = None
        self._stuck_count = 0
        self.step_idx = 0

        # PUCT selector and Go-Explore archive are the brains of the loop.
        # Both are stateful and live for the duration of one engine run.
        self.puct = PUCTSelector(c_puct=self.mode_config.c_puct)
        self.archive = GoExploreArchive()

        # action_id -> ElementSnapshot for the *current* screen, rebuilt
        # each time we land on a new screen. PUCT only knows about IDs.
        self._action_id_to_element: dict[str, ElementSnapshot] = {}

        if resume_from:
            self.graph = AppGraph.load(resume_from)
            logger.info(f"Resumed graph: {self.graph.stats()}")
        else:
            self.graph = AppGraph(app_bundle_id=app_bundle_id)

    async def run(self) -> AppGraph:
        """Run the PUCT + Go-Explore loop until all frontiers are exhausted.

        Stops when one of the following holds:
            - max_steps reached;
            - the current screen has no unexplored actions AND
              the Go-Explore archive has no remaining frontier states.
        """
        print(f"\n>>> Starting exploration of {self.app_bundle_id}")
        logger.info(
            f"Starting exploration of {self.app_bundle_id} "
            f"(mode={self.mode_config.mode}, c_puct={self.mode_config.c_puct}, "
            f"max_steps={self.max_steps})"
        )
        await self._emit_status("running")

        # App is already launched by Appium session. Just wait for UI to settle.
        print(">>> Waiting for app to settle (2s)...")
        await asyncio.sleep(2.0)

        print(">>> Capturing initial screen...")
        initial_node = await self._capture_screen()
        is_new = self.graph.add_node(initial_node)
        self.current_screen_id = initial_node.screen_id
        await self._on_screen_landed(initial_node, is_new=is_new, step=0)
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

        try:
            while self.step_idx < self.max_steps:
                self.step_idx += 1
                step = self.step_idx

                # Dismiss system popups (Save Password, Allow Notifications,
                # etc.) BEFORE doing anything else. These create spurious
                # self-loops and inflate stuck count.
                await self._dismiss_system_popup()

                unexplored = self.graph.get_unexplored_actions(self.current_screen_id)

                if not unexplored:
                    # No local frontier — ask Go-Explore for the global best.
                    frontier = self.archive.best_frontier(
                        exclude_current=self.current_screen_id
                    )
                    if frontier is None:
                        logger.info("No frontier remaining anywhere. Done.")
                        break
                    logger.info(
                        f"[Step {step}] Local frontier exhausted; "
                        f"navigating to {frontier.state_id[:8]} "
                        f"(unexplored={frontier.unexplored_count}, "
                        f"visits={frontier.visit_count})"
                    )
                    result = await go_back_to_screen(
                        self.controller,
                        self.graph,
                        self.current_screen_id,
                        frontier.state_id,
                        self._capture_screen_id,
                    )
                    if result:
                        self.current_screen_id = result
                        # Re-register and refresh archive after the jump.
                        if result in self.graph.nodes:
                            await self._on_screen_landed(
                                self.graph.nodes[result], is_new=False, step=step
                            )
                    else:
                        logger.warning(
                            f"[Step {step}] Navigation to "
                            f"{frontier.state_id[:8]} failed; will retry"
                        )
                    continue

                # Local frontier exists — pick an action via PUCT, restricted
                # to the still-unexplored elements on this screen.
                action_id = self._select_action(self.current_screen_id, unexplored)
                if action_id is None:
                    # Defensive: PUCT doesn't know about this screen for some
                    # reason. Re-register and retry on the next iteration.
                    logger.warning(
                        f"[Step {step}] PUCT returned no action despite "
                        f"{len(unexplored)} unexplored — re-registering"
                    )
                    self._register_screen_with_puct(self.graph.nodes[self.current_screen_id])
                    continue

                element = self._action_id_to_element.get(action_id)
                if element is None:
                    # Out-of-sync: rebuild the element map and retry.
                    self._rebuild_action_map(self.graph.nodes[self.current_screen_id])
                    element = self._action_id_to_element.get(action_id)
                    if element is None:
                        logger.error(
                            f"[Step {step}] Cannot resolve action {action_id}"
                        )
                        break

                print(
                    f"\n>>> [Step {step}/{self.max_steps}] "
                    f"Screen: {self._screen_name()}, "
                    f"Action: {element.kind} on {element.label!r}, "
                    f"Remaining unexplored: {len(unexplored) - 1}"
                )

                before_screen = self.current_screen_id
                known_before: set[str] = set(self.graph.nodes.keys())

                if element.kind == ElementKind.TEXT_FIELD:
                    await self._explore_text_field(element)
                else:
                    await self._explore_tap(element)

                after_screen = self.current_screen_id
                is_new_screen = (
                    after_screen != before_screen and after_screen not in known_before
                )

                # PUCT backup:
                #   +1.0  discovered a brand-new screen (excellent)
                #   +0.2  reached a known but different screen (some value)
                #   -0.5  self-loop: action brought us back to the same screen
                #         (strong penalty to discourage re-trying)
                if is_new_screen:
                    reward = 1.0
                elif after_screen != before_screen:
                    reward = 0.2
                else:
                    reward = -0.5
                self.puct.backup(before_screen, action_id, reward)

                # Re-register the new screen with PUCT and update the archive
                # for both endpoints. The source screen now has one fewer
                # unexplored action; the target screen may be brand new.
                if after_screen in self.graph.nodes:
                    target_node = self.graph.nodes[after_screen]
                    await self._on_screen_landed(target_node, is_new=is_new_screen, step=step)
                self._update_archive_for(before_screen)

                # Edge event: the worker writes this to Postgres + Redis.
                await self._emit_edge(
                    step=step,
                    source=before_screen,
                    target=after_screen,
                    action_type=ActionType.TAP if element.kind != ElementKind.TEXT_FIELD else ActionType.INPUT,
                    success=after_screen != before_screen or reward > 0,
                )

                # Periodic stats update so the UI doesn't have to count events.
                if step % 5 == 0 or is_new_screen:
                    await self._emit_stats(step=step)

                # Stuck detection
                if self._stuck_count >= MAX_STUCK_COUNT:
                    logger.warning(
                        f"Stuck on screen {self.current_screen_id[:8]} "
                        f"for {MAX_STUCK_COUNT} steps. Marking remaining as explored."
                    )
                    self._mark_screen_fully_explored(self.current_screen_id)
                    self._stuck_count = 0

                self._save_checkpoint()

        finally:
            # Finalize
            self.graph.exploration_end = datetime.now()
            self._save_checkpoint()
            await self._emit_stats(step=self.step_idx)
            logger.info(f"Exploration complete. {self.graph.stats()}")

        return self.graph

    # ------------------------------------------------------------------ PUCT

    def _select_action(
        self,
        state_id: str,
        unexplored: list[ElementSnapshot],
    ) -> str | None:
        """Pick the best PUCT action restricted to the still-unexplored set."""
        ss = self.puct.stats_for(state_id)
        if ss is None:
            # Should have been registered when we landed; do it now as a fallback.
            self._register_screen_with_puct(self.graph.nodes[state_id])
            ss = self.puct.stats_for(state_id)
            if ss is None:
                return None

        unexplored_ids = {action_id_for(el) for el in unexplored}
        # Exclude actions PUCT knows about that are no longer in the unexplored set.
        excluded = {aid for aid in ss.actions.keys() if aid not in unexplored_ids}
        return self.puct.select(state_id, exclude=excluded)

    def _register_screen_with_puct(self, node: ScreenNode) -> None:
        """Register a screen's actions with the PUCT selector (idempotent).

        Also rebuilds the action_id → element map and queries the optional
        prior_provider when the mode requests LLM priors.
        """
        if node.screen_id in self.puct.known_states():
            # Already registered; just refresh the action map.
            self._rebuild_action_map(node)
            return

        action_ids: list[str] = []
        for el in node.interactive_elements:
            action_ids.append(action_id_for(el))

        priors: dict[str, float] | None = None
        if self.mode_config.use_llm_priors and self.prior_provider is not None:
            try:
                raw = self.prior_provider(node)
                if asyncio.iscoroutine(raw):
                    # The provider is async; we cannot await here. The caller
                    # should pass an async provider via _on_screen_landed instead.
                    logger.warning(
                        "prior_provider returned a coroutine in sync path; "
                        "ignoring priors. Use _on_screen_landed for async."
                    )
                elif isinstance(raw, dict):
                    priors = raw
            except Exception:
                logger.exception("prior_provider failed; falling back to uniform")

        self.puct.register_state(node.screen_id, action_ids, priors=priors)
        self._rebuild_action_map(node)

    def _rebuild_action_map(self, node: ScreenNode) -> None:
        """Refresh the local action_id → element table for the current screen."""
        self._action_id_to_element = {
            action_id_for(el): el for el in node.interactive_elements
        }

    async def _on_screen_landed(
        self,
        node: ScreenNode,
        *,
        is_new: bool,
        step: int,
    ) -> None:
        """Hook called whenever the engine lands on a screen.

        Handles PUCT registration, async LLM prior fetching, archive update,
        and the screen_discovered event for live progress streaming.
        """
        # Async prior fetching path: only when prior_provider is set and the
        # screen hasn't been registered yet. We do this *before* registration.
        priors: dict[str, float] | None = None
        if (
            is_new
            and self.mode_config.use_llm_priors
            and self.prior_provider is not None
            and node.screen_id not in self.puct.known_states()
        ):
            try:
                raw = self.prior_provider(node)
                if asyncio.iscoroutine(raw):
                    raw = await raw
                if isinstance(raw, dict):
                    priors = raw
            except Exception:
                logger.exception("async prior_provider failed; uniform priors")

        if node.screen_id not in self.puct.known_states():
            action_ids = [action_id_for(el) for el in node.interactive_elements]
            self.puct.register_state(node.screen_id, action_ids, priors=priors)

        self._rebuild_action_map(node)
        self._update_archive_for(node.screen_id)

        if is_new:
            await self._emit_screen(node, step=step)

    def _update_archive_for(self, screen_id: str) -> None:
        """Refresh the Go-Explore record for a screen with current PUCT counts."""
        node = self.graph.nodes.get(screen_id)
        if node is None:
            return
        ss = self.puct.stats_for(screen_id)
        visit_count = ss.visit_count if ss is not None else 0
        # Total actions = all interactive elements; explored = those with at
        # least one outgoing edge in the graph (matches the unexplored query).
        total = len(node.interactive_elements)
        unexplored = len(self.graph.get_unexplored_actions(screen_id))
        explored = max(total - unexplored, 0)
        self.archive.update(
            state_id=screen_id,
            total_actions=total,
            explored_actions=explored,
            visit_count=visit_count,
        )

    # ------------------------------------------------------------------ events

    async def _emit(self, event: dict) -> None:
        """Send one event to the optional callback. Tolerates sync or async callables."""
        if self.event_callback is None:
            return
        try:
            result = self.event_callback(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("event_callback failed for event=%r", event)

    async def _emit_status(self, new_status: str) -> None:
        await self._emit({"type": "status_change", "new_status": new_status})

    async def _emit_screen(self, node: ScreenNode, *, step: int) -> None:
        await self._emit(
            {
                "type": "screen_discovered",
                "step_idx": step,
                "screen_id_hash": node.screen_id,
                "screen_name": node.name or node.screen_id[:8],
                "screenshot_path": None,
            }
        )

    async def _emit_edge(
        self,
        *,
        step: int,
        source: str,
        target: str,
        action_type: ActionType,
        success: bool,
    ) -> None:
        await self._emit(
            {
                "type": "edge_discovered",
                "step_idx": step,
                "source_screen_hash": source,
                "target_screen_hash": target,
                "action_type": str(action_type),
                "success": success,
            }
        )

    async def _emit_stats(self, *, step: int) -> None:
        await self._emit(
            {
                "type": "stats_update",
                "stats": {
                    "screens": len(self.graph.nodes),
                    "edges": len(self.graph.edges),
                    "step": step,
                    "max_steps": self.max_steps,
                },
            }
        )

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
        """Explore a text field: fill form + submit once, then mark all
        text fields on this screen as explored.

        The old approach tried every PBT variant for every field, which
        consumed 10-15 steps per form. The new approach:
        1. Fill all visible fields with valid data (happy path)
        2. Tap submit
        3. Mark ALL text fields on the source screen as explored
        4. Move on to new screens instead of re-filling the same form
        """
        before_screen = self.current_screen_id

        node = self.graph.nodes[self.current_screen_id]
        text_fields = [
            el for el in node.interactive_elements
            if el.kind == ElementKind.TEXT_FIELD
        ]

        # Fill form with valid data and submit
        if len(text_fields) > 1:
            await self._fill_form_happy_path(text_fields, element)
        else:
            await self._fill_single_field(element)

        # Mark ALL text fields on the source screen as explored so PUCT
        # doesn't keep selecting them. One valid fill + submit is enough
        # to discover where the form leads. PBT variants (empty, invalid,
        # overflow) are useful for testing but waste exploration budget.
        for field in text_fields:
            aid = action_id_for(field)
            if before_screen in self.graph.nodes:
                # Record a self-loop edge so get_unexplored_actions() skips it
                self.graph.add_edge(GraphEdge(
                    source_screen_id=before_screen,
                    target_screen_id=before_screen,
                    action=ActionDetail(
                        action_type=ActionType.INPUT,
                        target_label=field.label,
                        target_test_id=field.test_id,
                        input_category="explored",
                    ),
                ))

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

        # Record edge for the trigger field with valid input.
        # The persisted ``input_text`` is redacted for sensitive
        # fields (password / token / secret) so neither the graph
        # checkpoint on disk nor any future backend persistence
        # carries the raw secret. Non-sensitive values (ФИО / phone
        # / search) stay verbatim — diagnosability matters for them.
        from explorer.form_filler import redact_value
        valid_value = self.form_filler.get_valid_value_for(trigger_field)
        action = ActionDetail(
            action_type=ActionType.INPUT,
            target_label=trigger_field.label,
            target_test_id=trigger_field.test_id,
            target_frame=trigger_field.frame,
            input_text=redact_value(trigger_field, valid_value),
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

        # Redact stored input_text for sensitive fields — same policy
        # as _fill_form_happy_path above.
        from explorer.form_filler import redact_value
        action = ActionDetail(
            action_type=ActionType.INPUT,
            target_label=element.label,
            target_test_id=element.test_id,
            target_frame=element.frame,
            input_text=redact_value(element, value),
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
        """Set text into a TextInput field. Uses CDP for RN apps when available.

        Stdout shows a redacted view of ``text`` for sensitive fields
        (password / token / secret / SecureTextField) — see
        form_filler.redact_value. The controller still receives the
        real value because the typing call below uses ``text``
        directly; only the log line is masked.
        """
        from explorer.form_filler import redact_value
        log_text = redact_value(element, text)
        try:
            print(
                f"      Typing into {element.test_id or element.label!r}: {log_text!r:.40}",
                flush=True,
            )

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

    # Labels that identify system/OS popup buttons to auto-dismiss.
    # These bypass the normal exploration loop — the agent taps them
    # immediately to clear the dialog and continue with the real app.
    _DISMISS_LABELS = frozenset({
        # English
        "Not Now", "Don't Allow", "Cancel", "Dismiss", "Close",
        "OK", "Later", "Skip", "No Thanks", "Deny",
        # Russian
        "Не сейчас", "Не разрешать", "Отмена", "Закрыть",
        "Позже", "Пропустить", "Отклонить",
    })
    # Alert titles that mark system popups (the agent should not explore these)
    _SYSTEM_ALERT_TITLES = frozenset({
        "Save Password?", "Сохранить пароль?",
        "Allow Notifications", "Разрешить уведомления",
        "Would Like to Send You Notifications",
        "Would Like to Access Your Location",
        "Would Like to Use Your Current Location",
    })

    async def _dismiss_system_popup(self) -> None:
        """If a system alert (Save Password, Allow Notifications, etc.) is
        on screen, tap the dismiss button and re-capture."""
        try:
            elements = await asyncio.wait_for(
                self.controller.get_ui_elements(), timeout=5
            )
        except Exception:
            return

        # Collect all labels from the raw element dicts
        all_labels: list[str] = []
        for e in elements:
            label = e.get("label") or e.get("title") or ""
            all_labels.append(label)

        # Check for dismiss-like buttons regardless of alert type.
        # System popups on iOS have buttons like "Not Now", "Save",
        # "Don't Allow" etc. If we see any of these, tap the dismiss one.
        dismiss_candidates: list[dict] = []
        for e in elements:
            label = (e.get("label") or e.get("title") or "").strip()
            if label in self._DISMISS_LABELS:
                dismiss_candidates.append(e)

        if not dismiss_candidates:
            return

        # Prefer "Not Now" / "Не сейчас" / "Cancel" over other options
        target = dismiss_candidates[0]
        for c in dismiss_candidates:
            lbl = (c.get("label") or c.get("title") or "").strip()
            if lbl in ("Not Now", "Не сейчас", "Cancel", "Отмена", "Don't Allow", "Не разрешать"):
                target = c
                break

        frame = target.get("frame")
        label = (target.get("label") or target.get("title") or "")
        if frame:
            x = int(frame.get("x", 0) + frame.get("width", 0) / 2)
            y = int(frame.get("y", 0) + frame.get("height", 0) / 2)
            print(f"    [auto-dismiss] Tapping '{label}' at ({x}, {y})", flush=True)
            try:
                await asyncio.wait_for(self.controller.tap_at(x, y), timeout=5)
                await asyncio.sleep(1.0)
                new_node = await self._capture_screen()
                self.current_screen_id = new_node.screen_id
                self.graph.add_node(new_node)
            except Exception as exc:
                print(f"    [auto-dismiss] Failed: {exc}", flush=True)
            return

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
