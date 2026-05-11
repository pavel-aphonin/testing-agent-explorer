"""Navigation and backtracking for the app explorer."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from explorer.models import ActionDetail, AppGraph, GraphEdge

logger = logging.getLogger("explorer.navigator")

# Timing constants (seconds)
SETTLE_DELAY = 1.5
LAUNCH_DELAY = 3.0
BACK_DELAY = 1.0
MAX_BACK_ATTEMPTS = 5


async def go_back_to_screen(
    controller,
    graph: AppGraph,
    current_screen_id: str,
    target_screen_id: str,
    capture_fn,
) -> str | None:
    """
    Navigate from current screen to target screen.

    Strategies (in order):
    1. Press back up to MAX_BACK_ATTEMPTS times
    2. Relaunch app + replay path from start screen

    Returns the screen_id we ended up on, or None on failure.
    capture_fn: async () -> (screen_id, elements) - captures current screen state.
    """
    if current_screen_id == target_screen_id:
        return current_screen_id

    # Strategy 1: Try back button
    for attempt in range(MAX_BACK_ATTEMPTS):
        logger.info(f"Back attempt {attempt + 1}/{MAX_BACK_ATTEMPTS}")
        try:
            await controller.go_back()
        except Exception as e:
            logger.warning(f"go_back failed: {e}")
            break
        await asyncio.sleep(BACK_DELAY)

        new_screen_id, _ = await capture_fn()
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
                result = await replay_path(controller, path, capture_fn)
                if result == target_screen_id:
                    return result

    # Strategy 2: Relaunch app + replay
    logger.info("Back button failed, relaunching app")
    return await relaunch_and_navigate(
        controller, graph, target_screen_id, capture_fn
    )


async def relaunch_and_navigate(
    controller,
    graph: AppGraph,
    target_screen_id: str,
    capture_fn,
) -> str | None:
    """Kill app, relaunch, navigate to target via known path."""
    try:
        await controller.terminate_app(graph.app_bundle_id)
        await asyncio.sleep(1)
        await controller.launch_app(graph.app_bundle_id)
        await asyncio.sleep(LAUNCH_DELAY)
    except Exception as e:
        logger.error(f"Relaunch failed: {e}")
        return None

    start_screen_id, _ = await capture_fn()
    if start_screen_id == target_screen_id:
        return start_screen_id

    path = graph.find_path(start_screen_id, target_screen_id)
    if not path:
        logger.warning(
            f"No known path from start {start_screen_id[:8]} "
            f"to target {target_screen_id[:8]}"
        )
        return start_screen_id  # Return where we are, caller decides

    return await replay_path(controller, path, capture_fn)


async def replay_path(
    controller,
    path: list[GraphEdge],
    capture_fn,
) -> str | None:
    """Replay a sequence of actions to navigate along a known path."""
    for edge in path:
        success = await replay_action(controller, edge.action)
        if not success:
            logger.warning(f"Failed to replay action: {edge.action}")
            current_id, _ = await capture_fn()
            return current_id
        await asyncio.sleep(SETTLE_DELAY)

    current_id, _ = await capture_fn()
    return current_id


async def replay_action(controller, action: ActionDetail) -> bool:
    """Replay a single action."""
    try:
        match action.action_type:
            case "tap":
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
