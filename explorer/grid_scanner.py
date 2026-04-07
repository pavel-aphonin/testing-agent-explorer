"""
Grid-based accessibility scanner: replaces describe_all (which hangs on iOS 26)
with a grid of describe_point calls to discover all visible UI elements.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger("explorer.grid_scanner")

# Default grid spacing in logical points
DEFAULT_STEP_X = 60
DEFAULT_STEP_Y = 60

# Margin from edges (avoid status bar / home indicator)
MARGIN_TOP = 50
MARGIN_BOTTOM = 40
MARGIN_LEFT = 10
MARGIN_RIGHT = 10


def _element_key(el: dict) -> str:
    """Unique key for deduplication: type + label + frame."""
    frame = el.get("frame", {})
    return (
        f"{el.get('type', '')}|{el.get('AXLabel', '')}|"
        f"{int(frame.get('x', 0))}|{int(frame.get('y', 0))}|"
        f"{int(frame.get('width', 0))}|{int(frame.get('height', 0))}"
    )


def _normalize_element(raw: dict) -> dict:
    """Normalize a describe_point result to the standard element format."""
    return {
        "type": raw.get("type", ""),
        "label": raw.get("AXLabel", ""),
        "value": raw.get("AXValue", ""),
        "frame": raw.get("frame", {}),
        "enabled": raw.get("enabled", True),
        "visible": True,
    }


async def scan_screen_grid(
    idb_client,
    screen_width: int,
    screen_height: int,
    step_x: int = DEFAULT_STEP_X,
    step_y: int = DEFAULT_STEP_Y,
    timeout_per_point: float = 3.0,
) -> list[dict]:
    """
    Scan the screen with a grid of describe_point calls.
    Returns a deduplicated list of UI elements in standard format.

    Uses logical coordinates (points, not pixels).
    Screen dimensions should be in logical points.
    """
    # Build grid of points to probe
    points = []
    y = MARGIN_TOP
    while y < screen_height - MARGIN_BOTTOM:
        x = MARGIN_LEFT
        while x < screen_width - MARGIN_RIGHT:
            points.append((x, y))
            x += step_x
        y += step_y

    logger.debug(f"Scanning {len(points)} grid points ({step_x}x{step_y} step)")

    # Scan all points (sequentially to avoid overwhelming idb)
    seen_keys: set[str] = set()
    elements: list[dict] = []

    for x, y in points:
        try:
            raw = await asyncio.wait_for(
                idb_client.describe_point(x, y),
                timeout=timeout_per_point,
            )
            if raw and raw.get("type"):
                key = _element_key(raw)
                if key not in seen_keys:
                    seen_keys.add(key)
                    elements.append(_normalize_element(raw))
        except asyncio.TimeoutError:
            logger.debug(f"Timeout at ({x},{y}), skipping")
        except Exception as e:
            logger.debug(f"Error at ({x},{y}): {e}")

    logger.debug(f"Found {len(elements)} unique elements")
    return elements
