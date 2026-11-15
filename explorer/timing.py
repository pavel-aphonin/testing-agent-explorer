"""Where a run's time goes (PER-899).

aweassist run 9 took 17.4 s a step, and its log accounted for 40 of 102
minutes (tree reads, screenshots, settle waits). This records every
subprocess (AXe, simctl) by command and every ``asyncio.sleep`` by the
function that slept, so the rest has a name too.

Parallel point queries overlap, so their summed time can exceed the wall
clock; the report shows both.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_totals: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
_started = time.monotonic()
_real_sleep = asyncio.sleep
_installed = False


def record(key: str, seconds: float) -> None:
    entry = _totals[key]
    entry[0] += 1
    entry[1] += seconds


def record_caller(prefix: str, frame) -> None:
    """Count one call under «<prefix> <module>.<function>» of ``frame``."""
    module = frame.f_globals.get("__name__", "?").rsplit(".", 1)[-1]
    entry = _totals[f"{prefix} {module}.{frame.f_code.co_name}"]
    entry[0] += 1


def command_key(args: list[str]) -> str:
    """«axe describe-ui», «axe describe-ui --point», «simctl io», …"""
    name = Path(args[0]).name if args and args[0] else "?"
    if name == "xcrun" and len(args) > 2:
        return f"{args[1]} {args[2]}"
    sub = args[1] if len(args) > 1 else ""
    if "--point" in args:
        sub += " --point"
    return f"{name} {sub}".strip()


async def _timed_sleep(delay, result=None):
    frame = sys._getframe(1)
    module = frame.f_globals.get("__name__", "?").rsplit(".", 1)[-1]
    started = time.monotonic()
    try:
        return await _real_sleep(delay, result)
    finally:
        record(f"sleep {module}.{frame.f_code.co_name}", time.monotonic() - started)


def install() -> None:
    """Start counting sleeps; call once at the start of a run."""
    global _installed, _started
    _started = time.monotonic()
    _totals.clear()
    if not _installed:
        asyncio.sleep = _timed_sleep
        _installed = True


def uninstall() -> None:
    global _installed
    asyncio.sleep = _real_sleep
    _installed = False


def snapshot() -> dict:
    wall = time.monotonic() - _started
    rows = sorted(
        ({"what": k, "count": int(n), "seconds": round(s, 1)} for k, (n, s) in _totals.items()),
        key=lambda r: -r["seconds"],
    )
    return {"wall_seconds": round(wall, 1), "rows": rows}


def report() -> str:
    data = snapshot()
    wall = data["wall_seconds"] or 1
    lines = [f"--- Time ({data['wall_seconds'] / 60:.1f} min wall) ---"]
    for row in data["rows"][:25]:
        lines.append(
            f"{row['seconds'] / 60:6.1f} min {100 * row['seconds'] / wall:5.1f}%  "
            f"{row['count']:6d}×  {row['what']}"
        )
    return "\n".join(lines)


def save(path: str | Path) -> None:
    Path(path).write_text(json.dumps(snapshot(), ensure_ascii=False, indent=2))
