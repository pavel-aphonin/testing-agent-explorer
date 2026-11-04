"""Periodic dtuhidd restart against simulator degradation (testbed run 3)."""

from __future__ import annotations

import pytest

from explorer import axe_client


class _Proc:
    returncode = 0

    async def wait(self):
        return 0

    async def communicate(self):
        return b"", b""


@pytest.fixture
def spawned(monkeypatch):
    calls: list[tuple] = []

    async def fake_exec(*args, **kwargs):
        calls.append(args)
        return _Proc()

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(axe_client.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(axe_client.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(axe_client, "_hid_connections", {})
    monkeypatch.setattr(axe_client, "AXE", "/bin/axe")
    return calls


def _restarts(calls: list[tuple]) -> list[tuple]:
    return [c for c in calls if "kickstart" in c]


@pytest.mark.asyncio
async def test_dtuhidd_restarts_every_n_input_commands(spawned):
    n = axe_client._DTUHIDD_RESTART_EVERY
    for _ in range(2 * n):
        await axe_client._run(["/bin/axe", "tap", "--id", "x", "--udid", "SIM-1"])

    restarts = _restarts(spawned)
    assert len(restarts) == 2
    assert restarts[0][:4] == ("xcrun", "simctl", "spawn", "SIM-1")
    assert restarts[0][-1] == axe_client._DTUHIDD_LABEL


@pytest.mark.asyncio
async def test_describe_ui_does_not_count_toward_restart(spawned):
    for _ in range(2 * axe_client._DTUHIDD_RESTART_EVERY):
        await axe_client._run(["/bin/axe", "describe-ui", "--udid", "SIM-1"])

    assert _restarts(spawned) == []
