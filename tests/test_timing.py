"""Run timing (PER-899): every subprocess by command, every sleep by caller."""

from __future__ import annotations

import asyncio

from explorer import axe_client, timing


def test_command_keys():
    assert timing.command_key(["/opt/homebrew/bin/axe", "describe-ui", "--udid", "X"]) == "axe describe-ui"
    assert timing.command_key(["axe", "describe-ui", "--point", "1,2", "--udid", "X"]) == "axe describe-ui --point"
    assert timing.command_key(["xcrun", "simctl", "io", "X", "screenshot", "-"]) == "simctl io"


async def _napper():
    await asyncio.sleep(0)


async def test_sleeps_are_counted_by_caller():
    timing.install()
    try:
        await _napper()
    finally:
        timing.uninstall()

    rows = {r["what"]: r for r in timing.snapshot()["rows"]}
    assert rows["sleep test_timing._napper"]["count"] == 1
    assert asyncio.sleep is timing._real_sleep


async def test_subprocesses_are_counted(monkeypatch):
    timing.install()
    try:
        await axe_client._run(["/bin/echo", "hi"])
    finally:
        timing.uninstall()

    assert {r["what"] for r in timing.snapshot()["rows"]} == {"echo hi"}
    assert "echo hi" in timing.report()


def test_reads_are_counted_by_caller():
    import sys

    timing.install()
    try:
        def reader():
            timing.record_caller("read by", sys._getframe(0))
        reader()
    finally:
        timing.uninstall()
    rows = {r["what"]: r for r in timing.snapshot()["rows"]}
    assert rows["read by test_timing.reader"]["count"] == 1
