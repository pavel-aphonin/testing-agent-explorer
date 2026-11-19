"""Navigation and backtracking for the app explorer."""

from __future__ import annotations

import asyncio
import logging
import time
from explorer.analyzer import is_variant
from explorer.models import ActionDetail, ActionType, AppGraph, GraphEdge
from explorer.screen_id import native_fingerprint

logger = logging.getLogger("explorer.navigator")

# Timing constants (seconds)
SETTLE_DELAY = 1.5
# Detours tried when a step of a path no longer replays.
MAX_REROUTES = 2
LAUNCH_DELAY = 3.0
# A relaunch waited 1 + 3 s every time: 63 relaunches, 4.2 of 57.7 min on
# Wikipedia run 10. Now the app's first screen is polled for (_await_launch):
# its own tree with something to tap, the same in two reads in a row.
TERMINATE_DELAY = 0.5
LAUNCH_MIN_S = 1.0
LAUNCH_POLL_S = 0.5
LAUNCH_MAX_S = 6.0
# What a first screen offers; a launch storyboard (splash) offers none.
_LAUNCH_ACTION_TYPES = frozenset({
    "Button", "Cell", "Link", "TextField", "SecureTextField", "Switch", "RadioButton", "Tab",
})
BACK_DELAY = 1.0
# Instead of BACK_DELAY and SETTLE_DELAY the screen is polled until it has
# changed and holds (see _await_change). Fixed waits on replayed steps and
# back presses cost 386 of 3298 s on Wikipedia run 2.
CHANGE_POLL_S = 0.3
BACK_MAX_S = 1.5
STEP_MAX_S = 3.5
# Polls start every CHANGE_POLL_S and slow down by POLL_GROWTH up to
# POLL_MAX_S: a screen that does not change is not read 11 times in 3.5 s.
POLL_GROWTH = 1.5
POLL_MAX_S = 1.0
# Between replayed steps the next step's element is waited for (it is
# looked up anyway) instead of the whole screen settling; only a short
# pause lets the transition animation finish first.
STEP_ANIMATION_S = 0.4
ELEMENT_WAIT_S = 3.5
ELEMENT_TRIES = 6
MAX_BACK_ATTEMPTS = 5
# Consecutive back presses that leave the screen unchanged before giving
# up on back navigation for this trip.
MAX_BACK_STALLS = 2


async def go_back_to_screen(
    controller,
    graph: AppGraph,
    current_screen_id: str,
    target_screen_id: str,
    capture_fn,
    replay_input=None,
    diagnose=None,
    on_transition=None,
) -> str | None:
    """
    Navigate from current screen to target screen.

    Strategies (in order):
    1. Press back up to MAX_BACK_ATTEMPTS times
    2. Relaunch app + replay path from start screen

    Returns the screen_id we ended up on, or None on failure.
    capture_fn: async () -> (screen_id, elements) - captures current screen state.
    replay_input: optional async (ActionDetail) -> bool, see replay_path.
    on_transition: optional (source_id, action, landed_id) -> None, called
    for a Back or replayed step that landed somewhere the graph has no edge
    to (see add_replay_edge).
    """
    if current_screen_id == target_screen_id:
        return current_screen_id

    # Strategy 1: Try back button
    here = current_screen_id
    stalled = 0
    attempts = 0 if current_screen_id in graph.back_futile else MAX_BACK_ATTEMPTS
    if not attempts:
        logger.info(f"Back does not leave {current_screen_id[:8]} (seen before); relaunching")
    for attempt in range(attempts):
        logger.info(f"Back attempt {attempt + 1}/{MAX_BACK_ATTEMPTS}")
        before = await _fingerprint(controller, fresh=False)
        try:
            await controller.go_back()
        except Exception as e:
            logger.warning(f"go_back failed: {e}")
            break
        await _await_change(controller, before, BACK_MAX_S, BACK_DELAY)

        new_screen_id, _ = await capture_fn()
        # Back that does not move us will not start working on the next
        # try (aweassist run 1: five ~10 s attempts before every relaunch).
        if new_screen_id == here:
            stalled += 1
            if stalled >= MAX_BACK_STALLS:
                logger.info("Back does not leave this screen; relaunching")
                graph.back_futile.add(here)
                break
        else:
            stalled = 0
        if on_transition is not None and new_screen_id != here:
            on_transition(here, ActionDetail(action_type=ActionType.BACK), new_screen_id)
        here = new_screen_id
        if new_screen_id == target_screen_id:
            logger.info(f"Reached target screen via back button")
            return new_screen_id

        # If we ended up on a known screen that's not target,
        # check if we can reach target from here via the graph
        if new_screen_id in graph.nodes:
            path = graph.find_path(new_screen_id, target_screen_id)
            if path:
                logger.info(
                    f"Found path from {new_screen_id[:8]} to {target_screen_id[:8]} "
                    f"({len(path)} steps)"
                )
                result = await replay_path(
                    controller, path, capture_fn, replay_input, graph, diagnose, on_transition
                )
                if result == target_screen_id:
                    return result
                if result:
                    here = result

    # Strategy 2: Relaunch app + replay
    logger.info("Back button failed, relaunching app")
    return await relaunch_and_navigate(
        controller, graph, target_screen_id, capture_fn, replay_input, diagnose, on_transition
    )


async def relaunch_and_navigate(
    controller,
    graph: AppGraph,
    target_screen_id: str,
    capture_fn,
    replay_input=None,
    diagnose=None,
    on_transition=None,
) -> str | None:
    """Kill app, relaunch, navigate to target via known path."""
    try:
        await controller.terminate_app(graph.app_bundle_id)
        await asyncio.sleep(TERMINATE_DELAY)
        await controller.launch_app(graph.app_bundle_id)
        await _await_launch(controller)
    except Exception as e:
        logger.error(f"Relaunch failed: {e}")
        return None

    start_screen_id, _ = await capture_fn()
    if start_screen_id == target_screen_id:
        return start_screen_id

    path = graph.find_path(start_screen_id, target_screen_id)
    if not path:
        # A live feed opens on a new variant after each launch (Wikipedia:
        # «Community» with other cards), with no recorded paths from it yet.
        # It is the same place as an earlier launch screen: try their paths.
        for earlier in reversed(graph.launch_screens):
            path = graph.find_path(earlier, target_screen_id) if earlier != start_screen_id else None
            if path:
                logger.info(f"Starting from launch screen {earlier[:8]} as {start_screen_id[:8]}")
                break
    if start_screen_id not in graph.launch_screens:
        graph.launch_screens.append(start_screen_id)
    if not path:
        logger.warning(
            f"No known path from start {start_screen_id[:8]} "
            f"to target {target_screen_id[:8]}"
        )
        return start_screen_id  # Return where we are, caller decides

    return await replay_path(
        controller, path, capture_fn, replay_input, graph, diagnose, on_transition,
        start_as=start_screen_id,
    )


async def _await_launch(controller) -> None:
    """Return once the relaunched app shows its first screen: an app root
    with a label (SpringBoard's is blank), something to tap (a splash has
    nothing), and the same native fingerprint in two reads in a row. At
    most LAUNCH_MAX_S; a controller that cannot read waits LAUNCH_DELAY."""
    read = getattr(controller, "get_ui_elements", None)
    if read is None:
        await asyncio.sleep(LAUNCH_DELAY)
        return
    web_scan = getattr(controller, "scan_web_content", None) is not None
    await asyncio.sleep(LAUNCH_MIN_S)
    deadline = time.monotonic() + LAUNCH_MAX_S - LAUNCH_MIN_S
    reads_left = round((LAUNCH_MAX_S - LAUNCH_MIN_S) / LAUNCH_POLL_S)  # and a bound on reads
    previous = None
    while time.monotonic() < deadline and reads_left > 0:
        reads_left -= 1
        elements = await (read(web=False) if web_scan else read())
        up = any(
            e.get("type") == "Application" and (e.get("label") or "").strip() for e in elements
        ) and any(e.get("type") in _LAUNCH_ACTION_TYPES for e in elements)
        current = native_fingerprint(elements) if up else None
        if current is not None and current == previous:
            return
        previous = current
        await asyncio.sleep(LAUNCH_POLL_S)


async def replay_path(
    controller,
    path: list[GraphEdge],
    capture_fn,
    replay_input=None,
    graph: AppGraph | None = None,
    diagnose=None,
    on_transition=None,
    start_as: str | None = None,
) -> str | None:
    """Replay a sequence of actions to navigate along a known path.

    Before each step the live screen must be the step's source (or a variant
    of it); otherwise the replay stops without tapping and returns the live
    screen. ``start_as`` names one more screen accepted as the first step's
    source: a relaunch lands on a new variant of the launch screen whose
    recorded paths are tried (see relaunch_and_navigate).

    ``replay_input`` replaces the replay of ``input`` edges. One such edge
    stands for a whole submitted form (login: email + password + submit)
    and stores only the trigger field's text — redacted for passwords —
    so typing it back can never get past a login screen. The engine
    passes a callback that refills the form from test data and submits.

    ``on_transition(source_id, action, landed_id)`` is told when a step
    landed on a screen that is neither the edge's target nor a variant of
    it (a modal variant, an article reached through a replayed form). The
    screen would otherwise have no incoming edge: find_path never reaches
    it and its actions get written off. It costs one capture per step.
    """
    target = path[-1].target_screen_id if path else None
    here = path[0].source_screen_id if path else None
    landed_id = None
    reroutes = MAX_REROUTES
    remaining = list(path)
    previous: GraphEdge | None = None
    if graph is not None:
        graph.failed_edge = None
    while remaining:
        edge = remaining.pop(0)
        # A tap recorded for screen A must not land on screen B. The capture
        # after the previous step is reused; the first step needs its own.
        live = landed_id
        if live is None:
            live, _ = await capture_fn()
        if (
            live != edge.source_screen_id
            and not (start_as is not None and previous is None and live == start_as)
            and not _is_variant_of(graph, live, edge.source_screen_id)
        ):
            logger.warning(
                "Replay stopped: step expects %s but the live screen is %s; not tapping",
                edge.source_screen_id[:8], live[:8],
            )
            if graph is not None:
                # The step that brought us here led off the path; with no
                # previous step the start itself was wrong: nothing to blame.
                graph.failed_edge = previous
            return live
        before = await _fingerprint(controller, fresh=False)
        if edge.action.action_type == "input" and replay_input is not None:
            success = await replay_input(edge.action)
        else:
            success = await replay_action(controller, edge.action)
        if not success:
            logger.warning(f"Failed to replay action: {edge.action}")
            if diagnose is not None:
                # What was on screen instead (Wikipedia run 4: the «Places»
                # tab did not replay on a screen whose saved tree has it).
                await diagnose("replay-failed", edge)
            current_id, _ = await capture_fn()
            # The element is gone; find another way from here, once or
            # twice (Wikipedia run: «Map» and «Search» written off because
            # their only path went through «Enable location»). Twice gone
            # and the edge is dead: no route uses it again (Wikipedia run 10).
            edge.replay_failed()
            if graph is not None:
                graph.failed_edge = edge
            landed_id = current_id
            detour = graph.find_path(current_id, target) if graph is not None and target else None
            if reroutes <= 0 or not detour or any(e.stale for e in detour):
                return current_id
            reroutes -= 1
            logger.info(f"Rerouting from {current_id[:8]} to {target[:8]} ({len(detour)} steps)")
            remaining = list(detour)
            continue
        if remaining and before is not None:
            await asyncio.sleep(STEP_ANIMATION_S)
        else:
            await _await_change(controller, before, STEP_MAX_S, SETTLE_DELAY)
        landed_id = None
        if on_transition is not None:
            landed_id, _ = await capture_fn()
            if landed_id != edge.target_screen_id and not _is_variant_of(
                graph, landed_id, edge.target_screen_id
            ):
                on_transition(here, edge.action, landed_id)
            else:
                edge.replay_worked()  # it worked after all
            here = landed_id
        previous = edge

    if landed_id is not None:
        return landed_id
    current_id, _ = await capture_fn()
    return current_id


def _is_variant_of(graph: AppGraph | None, a: str, b: str) -> bool:
    na, nb = (graph.nodes.get(a), graph.nodes.get(b)) if graph is not None else (None, None)
    return bool(na and nb and is_variant(na, nb))


def add_replay_edge(graph: AppGraph, source_id: str, action: ActionDetail, landed_id: str) -> bool:
    """Record a transition seen while navigating, so ``landed_id`` is reachable.

    Navigation creates nodes without recording how they were reached. A
    landing on a near-duplicate of the source or of a known target is not
    a new place (the screen-family rule), and an existing edge needs no twin.
    """
    if source_id not in graph.nodes or landed_id not in graph.nodes or landed_id == source_id:
        return False
    if _is_variant_of(graph, landed_id, source_id):
        return False
    signature = action.signature()
    for e in graph.edges:
        if e.source_screen_id == source_id and e.action.signature() == signature and (
            e.target_screen_id == landed_id or _is_variant_of(graph, landed_id, e.target_screen_id)
        ):
            e.replay_worked()  # seen working again
            return False
    graph.add_edge(GraphEdge(
        source_screen_id=source_id, target_screen_id=landed_id, action=action, notes="replay",
    ))
    return True


async def _fingerprint(controller, fresh: bool = True) -> str | None:
    """Native fingerprint of the screen, or None for a controller that
    cannot read it (then fixed waits are used)."""
    read = getattr(controller, "get_ui_elements", None)
    if read is None:
        return None
    if getattr(controller, "scan_web_content", None) is not None:
        return native_fingerprint(await read(reuse=not fresh, web=False))
    return native_fingerprint(await read())


async def _await_change(controller, before: str | None, max_s: float, fallback_s: float) -> None:
    """Return once the screen differs from ``before`` and two polls agree,
    or after ``max_s``."""
    if before is None:
        await asyncio.sleep(fallback_s)
        return
    deadline = time.monotonic() + max_s
    reads_left = round(max_s / CHANGE_POLL_S)
    previous = before
    delay = CHANGE_POLL_S
    while reads_left > 0 and time.monotonic() < deadline:
        await asyncio.sleep(delay)
        current = await _fingerprint(controller)
        reads_left -= 1
        if current != before and current == previous:
            return
        # Changing: confirm soon. Still the same: look again less often.
        delay = CHANGE_POLL_S if current != previous else min(delay * POLL_GROWTH, POLL_MAX_S)
        previous = current


async def _await_element(scroll, action: ActionDetail) -> dict | None:
    """The element of a replayed tap, once the screen shows it."""
    deadline = time.monotonic() + ELEMENT_WAIT_S
    delay = CHANGE_POLL_S
    for _ in range(ELEMENT_TRIES):
        frame = await scroll(None, action.target_label, action.target_test_id, action.target_frame)
        if frame is not None or time.monotonic() >= deadline:
            return frame
        await asyncio.sleep(delay)
        delay = min(delay * POLL_GROWTH, POLL_MAX_S)
    return None


async def replay_action(controller, action: ActionDetail) -> bool:
    """Replay a single action."""
    try:
        match action.action_type:
            case "tap":
                scroll = getattr(controller, "scroll_into_view", None)
                if action.target_frame and scroll:
                    # The recorded frame may be below the fold or stale
                    # (the screen was scrolled when it was recorded).
                    frame = await _await_element(scroll, action)
                    if frame is None:
                        return False
                    result = await controller.tap_at(
                        int(frame.get("x", 0) + frame.get("width", 0) / 2),
                        int(frame.get("y", 0) + frame.get("height", 0) / 2),
                    )
                    return result.error is None
                if action.target_frame:
                    x = int(
                        action.target_frame.get("x", 0)
                        + action.target_frame.get("width", 0) / 2
                    )
                    y = int(
                        action.target_frame.get("y", 0)
                        + action.target_frame.get("height", 0) / 2
                    )
                    result = await controller.tap_at(x, y)
                    return result.error is None
                # No frame — fall back to test_id (most stable) then
                # label (last resort). The AXe controller exposes
                # tap_by_id / tap_by_label, not tap_element. Old code
                # called tap_element(text=...) which crashed every
                # replay with AttributeError as soon as a recorded
                # action arrived without target_frame.
                elif action.target_test_id and hasattr(controller, "tap_by_id"):
                    result = await controller.tap_by_id(action.target_test_id)
                    return result.error is None
                elif action.target_label and hasattr(controller, "tap_by_label"):
                    result = await controller.tap_by_label(action.target_label)
                    return result.error is None
                # Controller has neither (e.g. Appium) — caller will
                # need to feed coordinates via target_frame; we can't
                # synthesise them here.
                logger.warning(
                    "replay: cannot tap %r — no target_frame and "
                    "controller has no tap_by_id/tap_by_label",
                    action.target_label or action.target_test_id,
                )
                return False

            case "input":
                # Focus field first
                if action.target_frame:
                    x = int(
                        action.target_frame.get("x", 0)
                        + action.target_frame.get("width", 0) / 2
                    )
                    y = int(
                        action.target_frame.get("y", 0)
                        + action.target_frame.get("height", 0) / 2
                    )
                    await controller.tap_at(x, y)
                    await asyncio.sleep(0.5)
                if action.input_text is not None:
                    await controller.erase_text(100)
                    await asyncio.sleep(0.3)
                    return await controller.type_text(action.input_text)
                return False

            case "back":
                return await controller.go_back()

            case "launch":
                return await controller.launch_app(
                    action.target_label or ""
                )

            case _:
                logger.warning(f"Unknown action type: {action.action_type}")
                return False

    except Exception as e:
        logger.error(f"Error replaying action {action}: {e}")
        return False


def find_nearest_unexplored(
    graph: AppGraph,
    current_screen_id: str,
) -> str | None:
    """BFS from current screen to find closest screen with unexplored actions."""
    from collections import deque

    unexplored_screens = graph.get_screens_with_unexplored()
    if not unexplored_screens:
        return None

    if current_screen_id in unexplored_screens:
        return current_screen_id

    visited = {current_screen_id}
    queue: deque[str] = deque([current_screen_id])

    while queue:
        sid = queue.popleft()
        # Get all screens reachable from sid
        for edge in graph.edges:
            if edge.source_screen_id == sid:
                target = edge.target_screen_id
                if target in visited:
                    continue
                if target in unexplored_screens:
                    return target
                visited.add(target)
                queue.append(target)

    # No reachable unexplored screen found — return any unexplored
    return next(iter(unexplored_screens), None)
