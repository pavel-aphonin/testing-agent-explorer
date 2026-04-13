"""Test the worker's RealExecutor end-to-end against an injected fake controller.

This proves that the chain
    RealExecutor.run -> ExplorationEngine.run -> event_callback -> sink
works without spinning up a simulator. We pass `client_factory` to inject a
fake controller, capture every event the engine emits, and assert that the
worker's contract with the backend is preserved (each event is shaped the
way internal_runs.py expects).
"""

from __future__ import annotations

from typing import Any

import pytest

from explorer.worker import RealExecutor
from tests.test_engine import _make_three_screen_app


@pytest.mark.asyncio
async def test_real_executor_drives_engine_against_fake_controller(tmp_path):
    controller = _make_three_screen_app()

    # The factory has to ignore the udid kwarg that connect() will receive,
    # so we wrap connect on the fake.
    async def fake_connect(udid: str) -> None:
        return None

    async def fake_disconnect() -> None:
        return None

    controller.connect = fake_connect  # type: ignore[attr-defined]
    controller.disconnect = fake_disconnect  # type: ignore[attr-defined]

    executor = RealExecutor(
        output_root=tmp_path,
        client_factory=lambda: controller,
    )

    events: list[dict[str, Any]] = []

    async def sink(event: dict[str, Any]) -> None:
        events.append(event)

    await executor.run(
        config={
            "run_id": "11111111-1111-1111-1111-111111111111",
            "bundle_id": "test.app",
            "device_id": "BOOTED",
            "mode": "mc",
            "max_steps": 20,
        },
        event_sink=sink,
    )

    types = [e["type"] for e in events]
    assert "screen_discovered" in types
    assert "edge_discovered" in types
    assert "stats_update" in types

    # All screen events have the required field for the backend handler.
    for ev in events:
        if ev["type"] == "screen_discovered":
            assert ev.get("screen_id_hash"), "screen_id_hash required by /event endpoint"
        if ev["type"] == "edge_discovered":
            assert ev.get("source_screen_hash"), "source_screen_hash required"
            assert ev.get("target_screen_hash"), "target_screen_hash required"
            assert ev.get("action_type"), "action_type required"


@pytest.mark.asyncio
async def test_real_executor_falls_back_to_hybrid_for_unknown_mode(tmp_path):
    controller = _make_three_screen_app()

    async def fake_connect(udid: str) -> None:
        return None

    async def fake_disconnect() -> None:
        return None

    controller.connect = fake_connect  # type: ignore[attr-defined]
    controller.disconnect = fake_disconnect  # type: ignore[attr-defined]

    executor = RealExecutor(
        output_root=tmp_path,
        client_factory=lambda: controller,
    )

    # Should not raise even though "mystery-mode" isn't a real ExplorationMode.
    await executor.run(
        config={
            "run_id": "22222222-2222-2222-2222-222222222222",
            "bundle_id": "test.app",
            "mode": "mystery-mode",
            "max_steps": 5,
        },
        event_sink=lambda _ev: _noop(),
    )


async def _noop() -> None:
    return None
