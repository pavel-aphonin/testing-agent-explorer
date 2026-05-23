"""PER-162: unit tests for IOSSimulatorManager.clone_from.

The clone path is operator-critical (worker decides between create+
install and clone at run-claim time, based on whether the run row
has ``baseline_udid``). These tests don't hit real ``simctl`` —
they mock ``_exec`` and assert the command sequence so a typo or
arg-order regression surfaces in CI instead of on a live demo.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from explorer.simulator import IOSSimulatorManager


@pytest.mark.asyncio
async def test_clone_from_calls_simctl_clone_with_correct_args() -> None:
    """The clone command is `xcrun simctl clone <baseline> <new-name>`,
    new sim is named TA-<run_id_short>, returned UDID comes from the
    clone output."""
    manager = IOSSimulatorManager(
        device_identifier="ignored",
        runtime_identifier="ignored",
    )
    # _exec is the module-level helper that runs simctl commands;
    # patch it to return scripted stdout for both shutdown and clone.
    side_effects = ["", "ABCDEF12-3456-7890-1234-567890ABCDEF"]
    with patch(
        "explorer.simulator._exec",
        new=AsyncMock(side_effect=side_effects),
    ) as mock_exec:
        udid = await manager.clone_from(
            "BASELINE0-0000-0000-0000-000000000000",
            "abcdef1234567890",
        )
    assert udid == "ABCDEF12-3456-7890-1234-567890ABCDEF"
    assert manager.udid == udid
    # Two _exec calls: shutdown of source (safety), then clone.
    assert mock_exec.await_count == 2
    shutdown_args = mock_exec.await_args_list[0].args
    clone_args = mock_exec.await_args_list[1].args
    assert shutdown_args[:3] == ("xcrun", "simctl", "shutdown")
    assert shutdown_args[3] == "BASELINE0-0000-0000-0000-000000000000"
    assert clone_args[:3] == ("xcrun", "simctl", "clone")
    assert clone_args[3] == "BASELINE0-0000-0000-0000-000000000000"
    assert clone_args[4] == "TA-abcdef12"  # run_id[:8]


@pytest.mark.asyncio
async def test_clone_from_swallows_shutdown_error_on_already_stopped() -> None:
    """If the baseline is already shut down, ``simctl shutdown`` errors
    out (it's not idempotent in some Xcode releases). The clone path
    must swallow that and still proceed — otherwise an operator who
    forgot to leave the baseline running gets a confusing failure
    right when they're trying to use the feature."""
    manager = IOSSimulatorManager(
        device_identifier="ignored",
        runtime_identifier="ignored",
    )

    call_count = {"n": 0}

    async def fake_exec(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Shutdown raises (already stopped) — must be swallowed.
            raise RuntimeError("Unable to shutdown device in current state")
        return "NEWUDID00-1111-2222-3333-444444444444"

    with patch("explorer.simulator._exec", new=fake_exec):
        udid = await manager.clone_from(
            "BASELINE0-0000-0000-0000-000000000000",
            "abcdef1234567890",
        )
    assert udid == "NEWUDID00-1111-2222-3333-444444444444"
    assert call_count["n"] == 2  # both calls attempted


@pytest.mark.asyncio
async def test_clone_from_propagates_real_clone_failure() -> None:
    """The shutdown error is swallowed; a clone error is not.
    Operator needs to see "could not clone the baseline" loud and
    clear — silent failure would let the worker race ahead and crash
    on a missing UDID."""
    manager = IOSSimulatorManager(
        device_identifier="ignored",
        runtime_identifier="ignored",
    )

    async def fake_exec(*args, **kwargs):
        if args[2] == "shutdown":
            return ""  # shutdown ok
        raise RuntimeError("Source UDID not found")

    with patch("explorer.simulator._exec", new=fake_exec):
        with pytest.raises(RuntimeError, match="Source UDID not found"):
            await manager.clone_from(
                "BASELINE0-0000-0000-0000-000000000000",
                "abcdef1234567890",
            )
