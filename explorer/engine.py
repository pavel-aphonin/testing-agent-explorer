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
import re
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

from explorer.analyzer import TAB_BAR_TYPES, action_key, analyze_screen, is_variant
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
    UNREACHABLE_NOTE,
    WRITTEN_OFF_NOTE,
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

# PER-207: consecutive steps with ZERO graph growth before we declare a
# dead-end and stop the run. The «Local frontier exhausted → navigate →
# still exhausted» branch otherwise loops until max_steps, circling the
# same screens (observed in run bbbb2222: ~20 identical iterations after
# the scenario goal failed). 25 leaves ample room to legitimately
# exhaust one complex screen's actions (which may all self-loop) before
# tripping, while saving the bulk of a 200-step budget.
MAX_NO_PROGRESS_STEPS = 25

# A running app never has an empty accessibility tree, so zero elements
# means the read failed (e.g. AXe «Timed out creating the simulator
# remote automation session»). Testbed run baseline-1 recorded such a
# read as a new screen and kept «navigating» back to it 21 times.
# Retry with these delays (seconds) before giving up.
EMPTY_CAPTURE_RETRY_DELAYS = (1.0, 2.0)

# Budget for one controller.set_text_in_field call. The AXe client
# verifies the typed value and retries up to 3 times (tap, wait for the
# keyboard, clear, type, read back — up to ~12 s each); the old 10 s cap
# cut the first attempt short and testbed run 2 never got past login.
SET_TEXT_TIMEOUT_S = 60

# A screen not yet in the graph is re-captured after this delay and
# accepted once two captures agree (see _capture_settled_screen).
NEW_SCREEN_RECHECKS = 3
NEW_SCREEN_RECHECK_DELAY_S = 1.0
SPARSE_SCREEN_RECHECK_DELAY_S = 2.5

# PIN keypads: digits are never explored one by one — a few random codes
# can lock a test account. The PIN from test data («pin») is entered as a
# whole; after this many rejected entries in a row the agent stops trying.
MAX_PIN_FAILURES = 2
PIN_DIGIT_DELAY_S = 0.3
KEYPAD_DIGITS = frozenset("0123456789")

# SpringBoard alerts cleared on launch before the first capture.
MAX_LAUNCH_ALERTS = 3

# Failed navigations to one frontier screen before its remaining actions
# are written off (see _note_unreachable).
MAX_NAV_FAILURES = 2

# Type aliases for the optional engine callbacks.
PriorProvider = Callable[[ScreenNode], "dict[str, float] | Awaitable[dict[str, float]]"]
EventCallback = Callable[[dict], "None | Awaitable[None]"]


# Actions that open system UI running in its own process — photo picker,
# camera, share sheet, document picker. It is drawn over the app but is
# missing from the app's AX tree, so the agent keeps «seeing» the screen
# underneath and taps blindly into the system UI (aweassist run 3: after
# «Выбрать из галереи» the taps meant for «Справки и выписки», «Договоры»…
# landed in the Photos picker).
_SYSTEM_UI_TRIGGER_RE = re.compile(
    r"галере|сделать фото|снять фото|\bкамер|выбрать (фото|файл|документ)|поделиться"
    r"|photo library|choose (photo|from library|file)|take (a )?photo|\bcamera\b|\bshare\b",
    re.IGNORECASE,
)


# Without priors PUCT breaks ties in AX-tree order — top to bottom — so the
# tab bar, last in the tree, came last: aweassist run 2 reached «Платежи» at
# step 50 and «Чаты» at 166, behind a hundred cards on the main screen.
# Tabs open whole sections; trying them first maps the app breadth-first.
#
# A tab leads to the same section from every screen, though, so once it has
# been tapped it drops below ordinary actions everywhere (TAB_VISITED_WEIGHT):
# with the boost kept, every screen inside a section re-tried all five tabs
# first (aweassist run 3, steps 63–69: nothing but tab hops).
TAB_BAR_PRIOR_WEIGHT = 10.0
TAB_VISITED_WEIGHT = 0.1


def structural_priors(
    node: ScreenNode, visited_tabs: frozenset[str] | set[str] = frozenset()
) -> dict[str, float] | None:
    """Priors from the screen's structure alone (no LLM): tab bar items
    weigh TAB_BAR_PRIOR_WEIGHT until tapped, then TAB_VISITED_WEIGHT;
    everything else 1. None when there is nothing to prefer, which keeps
    PUCT's uniform default."""

    def weight(el: ElementSnapshot) -> float:
        if el.element_type not in TAB_BAR_TYPES:
            return 1.0
        return TAB_VISITED_WEIGHT if el.label in visited_tabs else TAB_BAR_PRIOR_WEIGHT

    weights = {action_id_for(el): weight(el) for el in node.interactive_elements}
    if all(w == 1.0 for w in weights.values()):
        return None
    return weights


def opens_system_ui(label: str | None) -> bool:
    return bool(label) and bool(_SYSTEM_UI_TRIGGER_RE.search(label))


def _root_label(elements: list[dict]) -> str | None:
    """Label of the AX tree's Application root: «» for SpringBoard (system
    alerts, home screen), None when the tree has no Application element."""
    for e in elements:
        if e.get("type") == "Application":
            return (e.get("label") or "").strip()
    return None


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
        forbidden: list[str] | None = None,
        system_ui_guard: bool = True,
    ):
        self.controller = controller
        # Skip actions that open system UI living outside the app's AX tree
        # (see opens_system_ui).
        self.system_ui_guard = system_ui_guard
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
        # Labels / accessibility ids the agent must never act on — e.g. an
        # app's debug environment switcher, where exploring would select
        # «Production» and log in to a live backend (aweassist).
        self.forbidden: frozenset[str] = frozenset(forbidden or ())
        self.current_screen_id: str | None = None
        self._stuck_count = 0
        self.step_idx = 0
        # PER-207: dead-end watchdog. ``_last_node_count`` is the graph
        # size at the last step that grew it; ``_no_progress_steps``
        # counts consecutive steps since then. Abort when it crosses
        # MAX_NO_PROGRESS_STEPS.
        self._last_node_count = 0
        self._last_edge_count = 0
        self._no_progress_steps = 0
        # Screen to leave on the next step (see MAX_STUCK_COUNT).
        self._leave_screen: str | None = None
        # AX root label of our own app (its display name), taken from the
        # first in-app capture. A different non-blank root means an action
        # took us into another app — see _capture_screen.
        self._app_label: str | None = None
        self._dismissing_popup = False
        # PIN keypad screens (all ten digit buttons) and their digit frames.
        self._keypad_screens: set[str] = set()
        self._keypad_frames: dict[str, dict] = {}
        self._pin_failures = 0
        # Screens the agent has scrolled: their captured frames may be stale
        # (apps often keep the scroll offset when you come back).
        self._scrolled_screens: set[str] = set()
        # Tab bar items already tapped somewhere (see TAB_VISITED_WEIGHT).
        self._visited_tabs: set[str] = set()
        # Failed navigations per target screen; see MAX_NAV_FAILURES.
        self._nav_failures: dict[str, int] = {}

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

        await self._clear_system_ui_at_start()

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

                # PER-207: dead-end watchdog. Runs at the TOP of every
                # iteration so it also counts the frontier-exhausted
                # branch (which ``continue``s past the end of the loop
                # body). If the graph hasn't grown for
                # MAX_NO_PROGRESS_STEPS consecutive steps we're circling
                # — stop with a clear reason instead of burning the rest
                # of max_steps.
                # Progress = a new screen or a newly explored action. Only
                # circling between exhausted screens adds neither.
                cur_node_count = len(self.graph.nodes)
                cur_edge_count = len(self.graph.edges)
                if cur_node_count > self._last_node_count or cur_edge_count > self._last_edge_count:
                    self._last_node_count = cur_node_count
                    self._last_edge_count = cur_edge_count
                    self._no_progress_steps = 0
                else:
                    self._no_progress_steps += 1
                if self._no_progress_steps >= MAX_NO_PROGRESS_STEPS:
                    logger.warning(
                        "[Step %d] Dead-end: no new screen in %d consecutive "
                        "steps (graph=%d nodes) — aborting exploration early.",
                        step, self._no_progress_steps, cur_node_count,
                    )
                    await self._emit({
                        "type": "log",
                        "step_idx": step,
                        "message": (
                            f"Тупик: за {self._no_progress_steps} шагов не "
                            f"открылось ни одного нового экрана — "
                            f"останавливаю исследование."
                        ),
                    })
                    break

                # Dismiss system popups (Save Password, Allow Notifications,
                # etc.) BEFORE doing anything else. These create spurious
                # self-loops and inflate stuck count.
                await self._dismiss_system_popup()

                if self.current_screen_id in self._keypad_screens:
                    if await self._enter_pin_here(step):
                        continue

                unexplored = self.graph.get_unexplored_actions(self.current_screen_id)
                leaving = self._leave_screen == self.current_screen_id
                self._leave_screen = None

                frontier = None
                if not unexplored or leaving:
                    # No local frontier (or this screen keeps going nowhere)
                    # — ask Go-Explore for the global best elsewhere.
                    frontier = self.archive.best_frontier(
                        exclude_current=self.current_screen_id
                    )
                    if frontier is None and not unexplored:
                        logger.info("No frontier remaining anywhere. Done.")
                        break
                if frontier is not None:
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
                        replay_input=self._replay_form,
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
                    if result and result != frontier.state_id and self._same_screen(
                        result, frontier.state_id
                    ):
                        # Arrived at a variant of the target (e.g. the same
                        # screen with a «reload» banner): work on the target.
                        logger.info(
                            "[Step %d] %s is a variant of %s; treating as reached",
                            step, result[:8], frontier.state_id[:8],
                        )
                        self.current_screen_id = frontier.state_id
                    elif result != frontier.state_id:
                        self._note_unreachable(frontier.state_id, step)
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

                # Stuck detection: this screen's recent actions lead nowhere.
                # Explore elsewhere first and come back to the rest later —
                # writing them off (the old behaviour) dropped 99 of 111
                # actions on aweassist «Счета», most of them below the fold.
                if self._stuck_count >= MAX_STUCK_COUNT:
                    logger.info(
                        f"Stuck on screen {self.current_screen_id[:8]} "
                        f"for {MAX_STUCK_COUNT} steps; exploring elsewhere first."
                    )
                    self._leave_screen = self.current_screen_id
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

        self.puct.register_state(
            node.screen_id, action_ids, priors=priors or structural_priors(node, self._visited_tabs)
        )
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
            self.puct.register_state(
                node.screen_id, action_ids, priors=priors or structural_priors(node, self._visited_tabs)
            )

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

        # Off-screen elements (and every element once this screen has been
        # scrolled, since the captured frames are then stale) are looked
        # up live and scrolled into view first.
        scroll = getattr(self.controller, "scroll_into_view", None)
        needs_scroll = getattr(self.controller, "needs_scroll", None)
        if scroll and needs_scroll and (
            before_screen in self._scrolled_screens
            or needs_scroll(element.element_type, element.frame)
        ):
            frame = await scroll(element.element_type, element.label, element.test_id, element.frame or {})
            self._scrolled_screens.add(before_screen)
            stuck = getattr(self.controller, "scroll_stuck", False)
            if frame is None:
                print(f"    ! Could not bring {element.label!r} on screen, skipping", flush=True)
                self._record_self_loop(element, ActionType.TAP)
            if stuck:
                self._write_off_below(element, before_screen)
            if frame is None:
                return
            x = int(frame.get("x", 0) + frame.get("width", 0) / 2)
            y = int(frame.get("y", 0) + frame.get("height", 0) / 2)

        print(f"    Tapping {element.label!r} at ({x}, {y})...", flush=True)
        try:
            result = None
            # Switches ignore a plain coordinate tap — not even on the knob
            # (testbed run 5: the notification toggle never flipped). AXe's
            # id-targeted tap hits the switch's activation point with a
            # physical touch, which does flip it.
            tap_switch = getattr(self.controller, "tap_switch", None) or getattr(
                self.controller, "tap_by_id", None
            )
            if element.kind == ElementKind.SWITCH and element.test_id and tap_switch:
                # tap_switch verifies the flip and retries (up to ~3 × 3 s).
                result = await asyncio.wait_for(
                    tap_switch(element.test_id), timeout=30
                )
                print(
                    f"    Switch tapped by id {element.test_id!r}"
                    + (f"; failed: {result.error}" if result.error else ""),
                    flush=True,
                )
            if result is None or result.error:
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

        if element.element_type in TAB_BAR_TYPES and element.label not in self._visited_tabs:
            self._visited_tabs.add(element.label)
            prefix = f"tap|{element.label}|"
            self.puct.scale_priors(
                lambda aid: aid.startswith(prefix),
                TAB_VISITED_WEIGHT / TAB_BAR_PRIOR_WEIGHT,
            )

        print(f"    Waiting {SETTLE_DELAY}s for UI to settle...")
        await asyncio.sleep(SETTLE_DELAY)

        print(f"    Capturing screen...")
        new_node = await self._capture_settled_screen(before=before_screen)
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
            # A switch flipping in place is its normal behaviour, not a
            # sign of being stuck. Counting it (testbed run 5) tripped
            # MAX_STUCK_COUNT on Settings and wrote off «Выйти» and
            # «Удалить аккаунт» untried.
            if element.kind != ElementKind.SWITCH:
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
                        target_frame=field.frame,
                        input_category="explored",
                    ),
                ))

    async def _fill_and_submit(self, fields: list[ElementSnapshot]) -> None:
        """Fill every field with its valid value, then tap the submit button."""
        for field in fields:
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
        submit_button = self._find_submit_button(post_fill_node)
        if submit_button:
            print(f"      Tapping submit button: {submit_button.label!r}", flush=True)
            await self._tap_element_directly(submit_button)
            await asyncio.sleep(2)  # Wait for navigation

    async def _replay_form(self, action: ActionDetail) -> bool:
        """Navigator hook for ``input`` edges: refill the whole form on the
        current screen from test data and submit, as exploration did."""
        node = await self._capture_screen()
        if node.screen_id in self._keypad_screens:
            return await self._type_pin()
        fields = [
            el for el in node.interactive_elements
            if el.kind == ElementKind.TEXT_FIELD
        ]
        if not fields:
            return False
        print(f"      Replaying form ({len(fields)} fields) on {node.name!r}", flush=True)
        await self._fill_and_submit(fields)
        return True

    # ------------------------------------------------------------------ PIN

    def _pin(self) -> str | None:
        return self.form_filler.test_data.get("pin")

    async def _type_pin(self) -> bool:
        """Tap the PIN digits from test data on the current keypad screen."""
        pin = self._pin()
        if not pin or self._pin_failures >= MAX_PIN_FAILURES:
            return False
        for digit in pin:
            if hasattr(self.controller, "tap_by_id"):
                result = await self.controller.tap_by_id(digit)
                if result.error is None:
                    await asyncio.sleep(PIN_DIGIT_DELAY_S)
                    continue
            frame = self._keypad_frames.get(digit)
            if not frame:
                return False
            await self.controller.tap_at(
                int(frame["x"] + frame["width"] / 2), int(frame["y"] + frame["height"] / 2)
            )
            await asyncio.sleep(PIN_DIGIT_DELAY_S)
        await asyncio.sleep(SETTLE_DELAY)
        return True

    async def _enter_pin_here(self, step: int) -> bool:
        """Enter the PIN on the current keypad screen and record the
        transition as one input action. Returns False when no PIN is known
        or the failure budget is spent — the digits are never explored one
        by one (random codes would lock the test account)."""
        before = self.current_screen_id
        print(f"    PIN keypad on {self._screen_name()!r}; entering PIN from test data", flush=True)
        if not await self._type_pin():
            return False
        new_node = await self._capture_settled_screen(before=before)
        is_new = self.graph.add_node(new_node)
        self.graph.add_edge(GraphEdge(
            source_screen_id=before,
            target_screen_id=new_node.screen_id,
            action=ActionDetail(
                action_type=ActionType.INPUT,
                target_label="PIN",
                input_text="***",
                input_category="valid",
            ),
        ))
        self.current_screen_id = new_node.screen_id
        if new_node.screen_id == before:
            self._pin_failures += 1
            logger.warning(
                "[Step %d] PIN not accepted (%d/%d)",
                step, self._pin_failures, MAX_PIN_FAILURES,
            )
            return self._pin_failures < MAX_PIN_FAILURES
        self._pin_failures = 0
        await self._on_screen_landed(new_node, is_new=is_new, step=step)
        return True

    async def _fill_form_happy_path(
        self,
        all_fields: list[ElementSnapshot],
        trigger_field: ElementSnapshot,
    ) -> None:
        """Fill all form fields with valid data, then try variations on trigger field."""
        before_screen = self.current_screen_id

        # No relaunch «for clean fields» here: it threw away the session,
        # so any form behind login (testbed run 4: profile edit) became
        # unreachable and the fill landed on the login screen. The
        # controller clears each field before typing instead.

        await self._fill_and_submit(all_fields)

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
                    timeout=SET_TEXT_TIMEOUT_S,
                )
                if ok:
                    await asyncio.sleep(0.2)
                    return
                if element.test_id:
                    # The controller already retried with read-back; typing
                    # again unverified would append to whatever it left.
                    print(f"      ! Could not set text in {element.test_id!r}", flush=True)
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
        "Not Now", "Don't Allow", "Don’t Allow", "Cancel", "Dismiss", "Close",
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

    async def _clear_system_ui_at_start(self) -> None:
        """Clear SpringBoard alerts shown on launch, before the first capture.

        aweassist opens with two: the notification permission prompt and
        an icon-change notice that is missing from the AX tree. Otherwise
        the first «screen» is an alert and the app's own name (used to
        tell our app from others) is never learned.
        """
        for _ in range(MAX_LAUNCH_ALERTS):
            elements = await self.controller.get_ui_elements()
            if not elements:
                return
            root = _root_label(elements)
            if root is None or root:
                return
            buttons = [
                e for e in elements
                if (e.get("label") or "").strip() in self._DISMISS_LABELS and e.get("frame")
            ]
            if buttons:
                frame = buttons[0]["frame"]
                print(f"    [launch] Dismissing system alert: {buttons[0].get('label')!r}", flush=True)
                await self.controller.tap_at(
                    int(frame.get("x", 0) + frame.get("width", 0) / 2),
                    int(frame.get("y", 0) + frame.get("height", 0) / 2),
                )
            elif hasattr(self.controller, "press_return"):
                print("    [launch] System alert without elements; pressing Return", flush=True)
                await self.controller.press_return()
            else:
                return
            await asyncio.sleep(1.5)

    async def _recover_from_springboard(self) -> None:
        """SpringBoard is in front but shows nothing we can tap.

        Either an alert missing from the AX tree (aweassist: «You have
        changed the icon for “Деньги”» has no elements; Return confirms
        it), or our app is gone — crashed or sent to the home screen — in
        which case relaunch it.
        """
        if hasattr(self.controller, "press_return"):
            print("    [auto-dismiss] SpringBoard with nothing to tap; pressing Return", flush=True)
            await self.controller.press_return()
            await asyncio.sleep(1.0)
            elements = await self.controller.get_ui_elements()
            root = _root_label(elements)
        else:
            root = ""
        if root != self._app_label:
            print("    -> App not in front; relaunching it", flush=True)
            await self.controller.launch_app(self.app_bundle_id)
            await asyncio.sleep(SETTLE_DELAY)
        outer, self._dismissing_popup = self._dismissing_popup, True
        try:
            node = await self._capture_screen()
        finally:
            self._dismissing_popup = outer
        self.current_screen_id = node.screen_id
        self.graph.add_node(node)

    async def _dismiss_system_popup(self) -> None:
        """If a system alert (Save Password, Allow Notifications, etc.) is
        on screen, tap the dismiss button and re-capture."""
        try:
            elements = await asyncio.wait_for(
                self.controller.get_ui_elements(), timeout=5
            )
        except Exception:
            return

        # System alerts live in SpringBoard (blank AX root label). Inside
        # our own app «Отмена»/«Cancel» belong to the app's sheets and
        # alerts — testbed run 3 closed the profile-edit sheet 25 times in
        # a row right after navigating to it.
        root_label = _root_label(elements)
        if self._app_label and root_label == self._app_label:
            return
        if self._app_label and root_label:
            # Another app is in front. An app switch can land after the
            # post-action capture (testbed run 4: «Открыть настройки iOS»
            # looked like a self-loop, then «Выйти» was tapped inside iOS
            # Settings), so check before acting, not only after.
            print(f"    -> Another app in front ({root_label!r}); returning", flush=True)
            await self.controller.launch_app(self.app_bundle_id)
            await asyncio.sleep(SETTLE_DELAY)
            new_node = await self._capture_screen()
            self.current_screen_id = new_node.screen_id
            self.graph.add_node(new_node)
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
            # Only a blank root is SpringBoard; no root at all means the
            # controller does not report one — nothing to recover from.
            if root_label is None or root_label:
                return
            await self._recover_from_springboard()
            return

        # Prefer "Not Now" / "Не сейчас" / "Cancel" over other options
        target = dismiss_candidates[0]
        for c in dismiss_candidates:
            lbl = (c.get("label") or c.get("title") or "").strip()
            if lbl in ("Not Now", "Не сейчас", "Cancel", "Отмена", "Don't Allow", "Don’t Allow", "Не разрешать"):
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
                # This capture must not re-enter the handler if the alert
                # is still up; the next step will try again.
                outer, self._dismissing_popup = self._dismissing_popup, True
                try:
                    new_node = await self._capture_screen()
                finally:
                    self._dismissing_popup = outer
                self.current_screen_id = new_node.screen_id
                self.graph.add_node(new_node)
            except Exception as exc:
                print(f"    [auto-dismiss] Failed: {exc}", flush=True)
            return

    async def _capture_settled_screen(self, before: str | None = None) -> ScreenNode:
        """Capture after an action, re-checking results that may be early.

        The app often answers later than SETTLE_DELAY. A known *other*
        screen is trusted as is. Re-checked until two captures agree:
        - a new screen — may still be loading (testbed run 7: «Счета»
          captured before its table rendered, account rows never seen);
        - no change (``before``) — the transition may not have started yet
          (run 12: a back tap and the switch to iOS Settings both landed
          after the capture; «Выйти» was then tapped inside iOS Settings).
        A new screen offering only one action is given longer to load.
        """
        node = await self._capture_screen()
        for _ in range(NEW_SCREEN_RECHECKS):
            if node.screen_id in self.graph.nodes and node.screen_id != before:
                return node
            sparse = node.screen_id not in self.graph.nodes and len(node.interactive_elements) <= 1
            await asyncio.sleep(SPARSE_SCREEN_RECHECK_DELAY_S if sparse else NEW_SCREEN_RECHECK_DELAY_S)
            again = await self._capture_screen()
            if again.screen_id == node.screen_id:
                return again
            print(f"      [capture] Screen still changing, re-capturing", flush=True)
            node = again
        return node

    async def _capture_screen(self, return_to_app: bool = True) -> ScreenNode:
        """Capture the current screen state from the device.

        If an action took us into another app (testbed run 2: «Открыть
        настройки iOS» → 20 steps exploring iOS Settings), bring our app
        back to the foreground and capture that instead. ``launch_app``
        on a running app only foregrounds it, so the session survives.
        """
        node = await self._capture_screen_once()
        label = (node.app_label or "").strip()
        if self._app_label is None:
            self._app_label = label or None
            return node
        if not return_to_app or label == self._app_label:
            return node
        if not label:
            # A blank root is SpringBoard: a system alert is on top. It is
            # not a screen of the app — testbed run 9 recorded the
            # notification prompt as one, and once answered it never came
            # back, so the agent spent 50 min trying to navigate to it.
            # Dismiss it, then capture the app underneath.
            if self._dismissing_popup:
                return node
            print("    -> System alert on screen; dismissing before capture", flush=True)
            self._dismissing_popup = True
            try:
                await self._dismiss_system_popup()
            finally:
                self._dismissing_popup = False
            return await self._capture_screen(return_to_app=False)
        print(
            f"    -> Left the app (now in {label!r}); returning to {self._app_label!r}",
            flush=True,
        )
        logger.warning("Action left the app for %r; returning", label)
        await self.controller.launch_app(self.app_bundle_id)
        await asyncio.sleep(SETTLE_DELAY)
        return await self._capture_screen(return_to_app=False)

    async def _capture_screen_once(self) -> ScreenNode:
        import time
        try:
            t0 = time.monotonic()

            print(f"      [capture] Getting UI elements...", flush=True)
            elements = await self.controller.get_ui_elements()
            for delay in EMPTY_CAPTURE_RETRY_DELAYS:
                if elements:
                    break
                print(f"      [capture] Empty UI tree, retrying in {delay:.0f}s...", flush=True)
                await asyncio.sleep(delay)
                elements = await self.controller.get_ui_elements()
            if not elements:
                logger.error("UI tree still empty after %d retries", len(EMPTY_CAPTURE_RETRY_DELAYS))
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
            if self.forbidden:
                # Dropped here, at the source, so nothing downstream — PUCT,
                # frontier navigation, form filling — can ever pick them.
                node.interactive_elements = [
                    el for el in node.interactive_elements
                    if el.label not in self.forbidden and el.test_id not in self.forbidden
                ]
            if self.system_ui_guard:
                node.interactive_elements = [
                    el for el in node.interactive_elements
                    if not opens_system_ui(el.label)
                ]
            digits = {
                el.label: el for el in node.interactive_elements
                if el.element_type == "Button" and el.label in KEYPAD_DIGITS
            }
            if set(digits) == KEYPAD_DIGITS:
                self._keypad_screens.add(node.screen_id)
                self._keypad_frames = {d: el.frame for d, el in digits.items() if el.frame}
                node.interactive_elements = [
                    el for el in node.interactive_elements
                    if not (el.element_type == "Button" and el.label in KEYPAD_DIGITS)
                ]
            print(f"      [capture] Analyzed: {node.name!r}, {len(node.interactive_elements)} interactive ({t2-t0:.1f}s total)", flush=True)
            return self._merge_into_known_variant(node)
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
        self,
        element: ElementSnapshot,
        action_type: ActionType,
        screen_id: str | None = None,
        notes: str = "Action failed or did not change screen",
    ) -> None:
        """Record a self-loop edge (action that didn't change the screen)
        on ``screen_id``, the current screen by default."""
        screen_id = screen_id or self.current_screen_id
        if not screen_id:
            return
        action = ActionDetail(
            action_type=action_type,
            target_label=element.label,
            target_test_id=element.test_id,
            target_frame=element.frame,
        )
        self.graph.add_edge(GraphEdge(
            source_screen_id=screen_id,
            target_screen_id=screen_id,
            action=action,
            success=False,
            notes=notes,
        ))

    def _same_screen(self, a: str, b: str) -> bool:
        """Navigation accepts a variant of the target as the target."""
        na, nb = self.graph.nodes.get(a), self.graph.nodes.get(b)
        return bool(na and nb and is_variant(na, nb))

    def _merge_into_known_variant(self, node: ScreenNode) -> ScreenNode:
        """A variant of a known screen is that screen, plus any new actions.

        A fingerprint must change when the structure does, so a card that
        loaded late gives a new one. Left alone, each variant is explored
        from scratch (aweassist run 4: «Поиск» tapped 12 times, 49 of 118
        screens) and can be written off as unreachable.
        """
        if node.screen_id in self.graph.nodes:
            return node
        known = next((n for n in self.graph.nodes.values() if is_variant(node, n)), None)
        if known is None:
            return node
        print(f"      [capture] Variant of {known.name!r} ({known.screen_id[:8]})", flush=True)
        keys = {action_key(el) for el in known.interactive_elements}
        for el in node.interactive_elements:
            if action_key(el) not in keys:
                known.interactive_elements.append(el)
                self.puct.add_action_if_missing(known.screen_id, action_id_for(el))
        # Stored frames are from another layout; look elements up live.
        self._scrolled_screens.add(known.screen_id)
        return known

    def _note_unreachable(self, screen_id: str, step: int) -> None:
        """Give up on a frontier screen after MAX_NAV_FAILURES failed trips.

        Some screens cannot be reached again: a one-shot state (a permission
        prompt already answered) or a path that no longer replays. Each
        failed trip costs minutes (back × 5, relaunch, re-login), and the
        dead-end watchdog counts steps, not time.
        """
        failures = self._nav_failures.get(screen_id, 0) + 1
        self._nav_failures[screen_id] = failures
        if failures < MAX_NAV_FAILURES or screen_id not in self.graph.nodes:
            return
        logger.warning(
            "[Step %d] %s unreachable after %d tries; writing off its actions",
            step, screen_id[:8], failures,
        )
        self._mark_screen_fully_explored(screen_id)
        # The archive only refreshes on landing, which never happens for an
        # unreachable screen; without this it stays the best frontier
        # (aweassist run 2: four more 1–2 min trips after the write-off).
        self._update_archive_for(screen_id)

    def _write_off_below(self, element: ElementSnapshot, screen_id: str) -> None:
        """Scrolling ended above ``element``: what lies below it is out of
        reach too, so one failed attempt settles them all. Actions with a
        visible text of the same name stay: they are tapped by that text."""
        node = self.graph.nodes[screen_id]
        y = (element.frame or {}).get("y", 0)
        visible = {
            s.label for s in node.elements
            if s.element_type == "StaticText" and self.controller.is_on_screen(s.frame)
        }
        below = [
            el for el in self.graph.get_unexplored_actions(screen_id)
            if (el.frame or {}).get("y", 0) >= y
            and el.element_type not in TAB_BAR_TYPES
            and el.label not in visible
        ]
        for el in below:
            self._record_self_loop(el, ActionType.TAP, screen_id, notes=UNREACHABLE_NOTE)
        print(f"    ! Scrolling ends above {element.label!r}: {len(below)} actions out of reach", flush=True)

    def _mark_screen_fully_explored(self, screen_id: str) -> None:
        """Mark all remaining elements on a screen as explored via self-loops."""
        unexplored = self.graph.get_unexplored_actions(screen_id)
        for el in unexplored:
            self._record_self_loop(el, ActionType.TAP, screen_id, notes=WRITTEN_OFF_NOTE)

    def _screen_name(self) -> str:
        """Get the name of the current screen."""
        if self.current_screen_id and self.current_screen_id in self.graph.nodes:
            return self.graph.nodes[self.current_screen_id].name or self.current_screen_id[:8]
        return "unknown"

    def _save_checkpoint(self) -> None:
        """Save current graph state to checkpoint file."""
        checkpoint_path = self.output_dir / "checkpoint.json"
        self.graph.save(checkpoint_path)
