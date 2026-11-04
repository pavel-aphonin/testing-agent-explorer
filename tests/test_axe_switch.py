"""Verified switch taps (testbed run 7: a dropped touch left the toggle off)."""

from __future__ import annotations

import pytest

from explorer import axe_client
from explorer.axe_client import AXeExplorerClient


def _client(monkeypatch, values: list[str | None]):
    """Client whose switch reads ``values`` in turn; records axe calls."""
    calls: list[list[str]] = []

    async def fake_run(args, timeout=15):
        calls.append(args)
        return 0, "", ""

    async def no_sleep(_seconds):
        return None

    reads = iter(values)

    async def get_ui_elements(self):
        value = next(reads)
        if value is None:  # a permission prompt replaced the app
            return [{"type": "Application", "label": " ", "test_id": ""}]
        return [{"type": "CheckBox", "label": "Уведомления",
                 "test_id": "settings.notifications", "value": value}]

    monkeypatch.setattr(axe_client, "_run", fake_run)
    monkeypatch.setattr(axe_client.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(AXeExplorerClient, "get_ui_elements", get_ui_elements)
    client = AXeExplorerClient()
    client._udid = "SIM-1"
    return client, calls


@pytest.mark.asyncio
async def test_dropped_touch_is_retried_with_post_delay(monkeypatch):
    client, calls = _client(monkeypatch, ["0", "0", "1"])

    result = await client.tap_switch("settings.notifications")

    assert result.error is None
    assert len(calls) == 2
    assert "--post-delay" not in calls[0]
    assert calls[1][2:6] == ["--tap-style", "physical", "--post-delay", "0.5"]


@pytest.mark.asyncio
async def test_permission_prompt_counts_as_flipped(monkeypatch):
    client, calls = _client(monkeypatch, ["0", None])

    result = await client.tap_switch("settings.notifications")

    assert result.error is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_switch_that_never_flips_reports_error(monkeypatch):
    client, calls = _client(monkeypatch, ["0", "0", "0", "0"])

    result = await client.tap_switch("settings.notifications")

    assert result.error
    assert len(calls) == axe_client._SWITCH_ATTEMPTS
