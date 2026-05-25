"""PER-169: tests for EpisodicMemory wrapper.

These tests intentionally do NOT spin up real Graphiti / FalkorDB —
that path is covered by the manual end-to-end smoke. Here we only
guard the wrapper logic: sanitization, group-id namespacing, the
strong-ref bookkeeping for fire-and-forget tasks, and the graceful
degradation when graphiti-core isn't available.

The strong-ref test (T46_fire_and_forget_holds_ref) regresses the
Python 3.13+ asyncio gotcha that bit us in smoke #1: bare
``asyncio.create_task()`` only keeps a weak reference and the GC
reaped 4/4 add_episode tasks before they ran, leaving the FalkorDB
graph empty. Fix: keep them in a set with a discard-on-done callback.
"""
from __future__ import annotations

import asyncio
import gc
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from explorer.episodic_memory import EpisodicMemory, EpisodicMemoryConfig


def test_T44_sanitize_replaces_separators_with_underscores() -> None:
    """RediSearch rejects UUIDs with ``-`` because it treats the dash
    as a query operator (``Syntax error at offset N near run_xxx``).
    PER-169 retry #1: _sanitize maps every non-alnum/non-underscore
    char to ``_`` so the group_id stays RediSearch-clean."""
    s = EpisodicMemory._sanitize("d7b8000c-b2a9-4342-b1f6-4357db0a6fb8")
    assert s == "d7b8000c_b2a9_4342_b1f6_4357db0a6fb8"
    assert "-" not in s
    # Empty / fully-stripped input falls back to "x" so the namespace
    # never collapses to "" (which would mix runs together).
    assert EpisodicMemory._sanitize("") == "x"
    assert EpisodicMemory._sanitize("---") == "___"  # not empty


def test_T44b_group_id_with_and_without_goal() -> None:
    """The namespace must (a) include run+goal when goal is given,
    (b) fall back to run-only when goal is None — so episodes added
    without a node_id still land somewhere queryable."""
    em = EpisodicMemory()
    assert em._group_id("run-abc", "goal-1") == "run_run_abc__goal_goal_1"
    assert em._group_id("run-abc", None) == "run_run_abc"


def test_T45_recall_returns_empty_when_graphiti_unavailable() -> None:
    """If graphiti-core fails to import (or init fails) we must not
    raise — the agent should run unchanged, just without memory."""
    async def _run() -> list[str]:
        em = EpisodicMemory()
        # Latch the init-failed flag without touching the network.
        em._init_failed = True
        return await em.recall("r1", "g1", "any query")
    assert asyncio.run(_run()) == []


def test_T45b_summary_for_prompt_empty_when_no_facts() -> None:
    """``summary_for_prompt`` must return '' (not raise, not None)
    when recall comes back empty. scenario_runner uses falsiness of
    the return value to decide whether to prepend the «Память:»
    header to the prompt."""
    async def _run() -> str:
        em = EpisodicMemory()
        em._init_failed = True  # makes recall short-circuit to []
        return await em.summary_for_prompt("r1", "g1", "any query")
    assert asyncio.run(_run()) == ""


def test_T45c_summary_caps_at_budget() -> None:
    """``memory_block_max_chars`` is a hard cap — overflowing lines
    are dropped, not truncated mid-string. Guard against future
    «just slice it» refactors that would feed the LLM a half-line."""
    cfg = EpisodicMemoryConfig(memory_block_max_chars=80)
    em = EpisodicMemory(cfg)
    em._init_failed = True

    async def _patched_recall(*a: Any, **kw: Any) -> list[str]:
        return [
            "first fact about taps",  # ~22 chars + "- " + newline ≈ 25
            "second fact about taps",  # ~23 chars → another 26
            "third fact that pushes us over the 80-char budget",  # would exceed
        ]
    em.recall = _patched_recall  # type: ignore[method-assign]
    out = asyncio.run(em.summary_for_prompt("r", "g", "q"))
    assert "first fact" in out
    assert "second fact" in out
    assert "third fact" not in out  # dropped, not partial-included
    assert len(out) <= 80


def test_T46_fire_and_forget_holds_strong_ref() -> None:
    """Regression for PER-169 smoke #1: bare ``asyncio.create_task``
    is GC'd before running in Python 3.13+. Keep strong refs in
    ``_inflight_tasks`` and confirm the task survives a GC pass and
    completes."""
    completed = []

    async def _run() -> None:
        em = EpisodicMemory()
        # Don't actually call Graphiti — patch _ensure_init to return
        # a stub object whose add_episode just records the call.
        stub_g = MagicMock()
        async def _fake_add(**kw: Any) -> None:
            completed.append(kw["name"])
        stub_g.add_episode = AsyncMock(side_effect=_fake_add)
        # patch _ensure_init to return the stub immediately
        async def _ensure() -> Any:
            return stub_g
        em._ensure_init = _ensure  # type: ignore[method-assign]

        # Fire 3 episodes
        for i in range(3):
            await em.add_action_fire_and_forget(
                run_id="r1", goal_id="g1",
                episode_text=f"text {i}", episode_name=f"ep_{i}",
            )
        # All three should be tracked
        assert len(em._inflight_tasks) == 3

        # Force a GC pass — without strong refs the tasks would die here.
        gc.collect()

        # Wait for tasks to run
        await asyncio.sleep(0.1)
        # discard-on-done should empty the set
        assert len(em._inflight_tasks) == 0
        # And all three add_episode calls landed
        assert sorted(completed) == ["ep_0", "ep_1", "ep_2"]

    asyncio.run(_run())


def test_T47_recall_uses_scoped_driver_clone() -> None:
    """Regression for the FalkorDB per-group_id graph routing bug
    (PER-169 smoke #1): ``add_episode`` clones the driver to
    database=group_id, so writes land in that graph. ``recall`` must
    pass an explicit driver scoped the same way, else it reads
    default_db and returns 0 even when the data exists. Verify the
    driver passed to retrieve_episodes was cloned with the right
    database name."""
    captured: dict[str, Any] = {}

    async def _fake_retrieve(**kw: Any) -> list[Any]:
        # Capture the driver instance so the test can assert on it.
        captured["driver"] = kw.get("driver")
        captured["group_ids"] = kw.get("group_ids")
        ep = MagicMock()
        ep.content = "Step 0: tap on X — OK"
        ep.name = "step_0"
        return [ep]

    stub_g = MagicMock()
    stub_g.retrieve_episodes = AsyncMock(side_effect=_fake_retrieve)
    cloned_driver = MagicMock(name="cloned-driver")
    stub_g.driver = MagicMock()
    stub_g.driver.clone = MagicMock(return_value=cloned_driver)

    async def _run() -> list[str]:
        em = EpisodicMemory()
        async def _ensure() -> Any:
            return stub_g
        em._ensure_init = _ensure  # type: ignore[method-assign]
        return await em.recall("run-X", "goal-Y", "ignored")

    facts = asyncio.run(_run())
    assert facts == ["Step 0: tap on X — OK"]
    # driver was cloned with the sanitized group_id, not default_db
    expected_group = "run_run_X__goal_goal_Y"
    stub_g.driver.clone.assert_called_once_with(database=expected_group)
    assert captured["driver"] is cloned_driver
    assert captured["group_ids"] == [expected_group]


def test_T48_close_waits_for_inflight_tasks() -> None:
    """``close()`` must give pending add_episode tasks a chance to
    flush — otherwise a worker shutdown right after the last action
    would silently drop that episode."""
    flushed = []

    async def _run() -> None:
        em = EpisodicMemory()
        stub_g = MagicMock()
        async def _slow_add(**kw: Any) -> None:
            await asyncio.sleep(0.05)
            flushed.append(kw["name"])
        stub_g.add_episode = AsyncMock(side_effect=_slow_add)
        stub_g.close = AsyncMock()
        async def _ensure() -> Any:
            return stub_g
        em._ensure_init = _ensure  # type: ignore[method-assign]
        # Backdoor: set the cached graphiti so close() actually tries to flush
        em._graphiti = stub_g

        await em.add_action_fire_and_forget(
            run_id="r", goal_id="g",
            episode_text="t", episode_name="ep_late",
        )
        # close() must wait for ep_late to finish
        await em.close()
        assert flushed == ["ep_late"]

    asyncio.run(_run())
