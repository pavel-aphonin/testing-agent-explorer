"""Web content the accessibility tree leaves out (PER-898).

What a WKWebView shows lives in a separate WebContent process, and
``axe describe-ui`` lists only the empty scroll view around it: Ionic
Conference gave 18 nodes for a screen with a button, a title and a
paragraph. A point query (``describe-ui --point``) does reach into the web
page, so the empty area is probed with a grid of them and what they find is
put back into the tree as children of that scroll view.

A point query takes about 0.75 s; sixteen in parallel cover a 402×874
screen (about 44 points) in 3–3.6 s. Points that fall inside an element
already found are skipped. The grid is coarse on purpose: a web page
answers a point query with the nearest element, not only one under it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

# A scroll view at least this share of the screen with nothing in it but
# its scroll bars is taken for a web page.
MIN_AREA_SHARE = 0.15
GRID_STEP_X = 100
GRID_STEP_Y = 80
GRID_MARGIN = 16
PARALLEL_PROBES = 16
# The status bar: probes there hit SpringBoard's clock, signal and battery
# (testbed run 22 tapped «01:54», «Cellular», «100% battery power»).
STATUS_BAR_PT = 54

# Point queries also hit the native frame around the page: its scroll bars
# and empty containers carry no action or text.
_SKIP_TYPES = frozenset({"Slider", "ScrollArea", "Group", "Application", "Window"})
_SCROLL_BAR_TYPES = frozenset({"Slider"})

Probe = Callable[[int, int], Awaitable["dict | None"]]

# A page read stays valid until the next input or this many seconds. The
# testbed's web screen was scanned 9 times in 42 steps (404 point queries):
# every capture, every settle re-check, every scroll lookup read it again.
# A read is reused only once two scans in a row agree: the first scan of a
# page still loading found 1 of its 2 elements (testbed run 27), and reusing
# it would make the next read «agree» with a half-loaded page.
CACHE_SECONDS = 10.0
_cache: dict[tuple, tuple[float, list[dict], bool]] = {}


def forget() -> None:
    """Something was tapped, typed or swiped: pages may have changed."""
    _cache.clear()


def _frame(node: dict) -> dict:
    return node.get("frame") or {}


def _has_content(node: dict) -> bool:
    """Anything labelled or a control below ``node``, scroll bars aside."""
    for child in node.get("children", []):
        if not isinstance(child, dict) or child.get("type") in _SCROLL_BAR_TYPES:
            continue
        if child.get("AXLabel") or child.get("AXValue") or child.get("type") not in ("Group", "ScrollArea", ""):
            return True
        if _has_content(child):
            return True
    return False


def hollow_areas(roots: list, width: float, height: float) -> list[dict]:
    """Scroll views covering a fair part of the screen with no content."""
    found: list[dict] = []

    def walk(node: dict) -> None:
        frame = _frame(node)
        area = max(0.0, frame.get("width", 0)) * max(0.0, frame.get("height", 0))
        if (
            node.get("type") == "ScrollArea"
            and area >= MIN_AREA_SHARE * width * height
            and not _has_content(node)
        ):
            found.append(node)
            return
        for child in node.get("children", []):
            if isinstance(child, dict):
                walk(child)

    for root in roots:
        # SpringBoard (home screen, system alerts) has a blank root label;
        # outside the app there is no page of the app's to read.
        if isinstance(root, dict) and (root.get("AXLabel") or "").strip():
            walk(root)
    return found


def grid(frame: dict, width: float, height: float) -> list[tuple[int, int]]:
    """Probe points inside ``frame`` clipped to the screen, row by row."""
    left = max(0.0, frame.get("x", 0)) + GRID_MARGIN
    top = max(STATUS_BAR_PT, frame.get("y", 0)) + GRID_MARGIN / 2
    right = min(width, frame.get("x", 0) + frame.get("width", 0)) - GRID_MARGIN
    bottom = min(height, frame.get("y", 0) + frame.get("height", 0)) - GRID_MARGIN / 2
    points = []
    y = top
    while y <= bottom:
        x = left
        while x <= right:
            points.append((int(x), int(y)))
            x += GRID_STEP_X
        y += GRID_STEP_Y
    return points


def _inside(point: tuple[int, int], frame: dict) -> bool:
    x, y = point
    return (
        frame.get("x", 0) <= x <= frame.get("x", 0) + frame.get("width", 0)
        and frame.get("y", 0) <= y <= frame.get("y", 0) + frame.get("height", 0)
    )


def _as_node(hit: dict) -> dict:
    """A probe result as a tree node. Web text often has its words in the
    value, not the label."""
    label = hit.get("AXLabel") or ""
    if not label and hit.get("type") == "StaticText":
        label = hit.get("AXValue") or ""
    return {
        "type": hit.get("type", ""),
        "AXLabel": label,
        "AXValue": hit.get("AXValue", ""),
        "AXUniqueId": hit.get("AXUniqueId") or "",
        "frame": _frame(hit),
        "enabled": hit.get("enabled", True),
        "children": [],
    }


async def scan(area: dict, probe: Probe, width: float, height: float) -> list[dict]:
    """Elements a grid of point queries finds inside ``area``."""
    found: list[dict] = []
    seen: set[tuple] = set()
    pending = grid(_frame(area), width, height)
    while pending:
        batch, pending = pending[:PARALLEL_PROBES], pending[PARALLEL_PROBES:]
        hits = await asyncio.gather(*(probe(x, y) for x, y in batch))
        for hit in hits:
            if not hit or hit.get("type") in _SKIP_TYPES:
                continue
            frame = _frame(hit)
            if frame.get("y", 0) + frame.get("height", 0) <= STATUS_BAR_PT:
                continue
            key = (hit.get("type"), hit.get("AXLabel"), *(round(frame.get(k, 0)) for k in ("x", "y", "width", "height")))
            if key in seen:
                continue
            seen.add(key)
            found.append(_as_node(hit))
        pending = [p for p in pending if not any(_inside(p, n["frame"]) for n in found)]
    found.sort(key=lambda n: (_frame(n).get("y", 0), _frame(n).get("x", 0)))
    return found


async def fill_hollow_areas(roots: list, probe: Probe, width: float, height: float) -> int:
    """Put what the probes find into each hollow scroll view; returns the
    number of elements added."""
    added = 0
    for area in hollow_areas(roots, width, height):
        key = tuple(round(_frame(area).get(k, 0)) for k in ("x", "y", "width", "height"))
        cached = _cache.get(key)
        fresh = cached is not None and time.monotonic() - cached[0] < CACHE_SECONDS
        if fresh and cached[2]:
            elements = cached[1]
        else:
            elements = await scan(area, probe, width, height)
            # An empty read may be a page still loading: read it again.
            if elements:
                confirmed = fresh and cached[1] == elements
                _cache[key] = (time.monotonic(), elements, confirmed)
        area.setdefault("children", []).extend(elements)
        added += len(elements)
    return added
