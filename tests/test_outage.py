"""The environment going down is not a reason to write screens off.

aweassist run 8: at 01:53 the test backend answered «no healthy upstream».
The agent took it for a wrong PIN, could not get anywhere afterwards and
wrote off 25 screens (42 actions) in 23 minutes of relaunches.
"""

from __future__ import annotations

import pytest

from explorer import engine as engine_module
from explorer.engine import ENVIRONMENT_DOWN_REASON, MAX_PIN_FAILURES, OUTAGE_WAITS_S, ExplorationEngine
from explorer.models import WRITTEN_OFF_NOTE, ElementKind, ElementSnapshot, ScreenNode
from explorer.modes import ExplorationMode
from explorer.visualizer import generate_stats_report
from tests.test_engine import _app, _button, _PopupController

SCREENS = ["accounts", "payments", "profile", "cards", "chats"]


def _node(screen_id):
    actions = [
        ElementSnapshot(kind=ElementKind.BUTTON, element_type="Button", label=f"{screen_id} {c}",
                        frame={"x": 0, "y": 100 + 50 * n, "width": 402, "height": 40})
        for n, c in enumerate("абв")
    ]
    return ScreenNode(screen_id=screen_id, elements=actions, interactive_elements=actions)


@pytest.fixture
def waits(monkeypatch):
    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(engine_module.asyncio, "sleep", sleep)
    return slept


def _engine(tmp_path):
    engine = ExplorationEngine(controller=_PopupController(), app_bundle_id="com.aweassist.app",
                               output_dir=str(tmp_path), mode=ExplorationMode.MC)
    for screen_id in SCREENS:
        engine.graph.add_node(_node(screen_id))
        engine._update_archive_for(screen_id)
    return engine


async def _fail(engine, *screens):
    results = [await engine._navigation_failed(s, step=1) for s in screens]
    return results[-1]


async def test_screens_failing_twice_are_still_written_off(tmp_path, waits):
    engine = _engine(tmp_path)

    await _fail(engine, "accounts")
    engine._app_responds()
    await _fail(engine, "accounts")

    assert engine.graph.written_off_count == 3
    assert waits == []


async def test_failures_in_a_row_undo_the_write_offs_and_wait(tmp_path, waits):
    engine = _engine(tmp_path)

    assert await _fail(engine, "accounts", "accounts", "payments")

    assert engine.graph.written_off_count == 0
    assert {a.label for a in engine.graph.get_unexplored_actions("accounts")} == {"accounts а", "accounts б", "accounts в"}
    assert engine._nav_failures.get("accounts") is None
    assert waits == [OUTAGE_WAITS_S[0]]


async def test_no_write_offs_while_waiting(tmp_path, waits):
    engine = _engine(tmp_path)

    await _fail(engine, "accounts", "payments", "profile", "cards", "cards")

    assert engine.graph.written_off_count == 0
    assert waits == list(OUTAGE_WAITS_S[:3])


async def test_environment_that_stays_down_stops_the_run(tmp_path, waits):
    engine = _engine(tmp_path)

    keep_going = await _fail(engine, *(SCREENS[:2] + SCREENS * 2)[: 2 + len(OUTAGE_WAITS_S) + 1])

    assert keep_going is False
    assert engine.graph.stop_reason == ENVIRONMENT_DOWN_REASON
    assert f"Stopped early: {ENVIRONMENT_DOWN_REASON}" in generate_stats_report(engine.graph)
    assert engine.graph.written_off_count == 0


async def test_a_working_app_resets_the_count(tmp_path, waits):
    engine = _engine(tmp_path)

    await _fail(engine, "accounts", "payments")
    engine._app_responds()
    await _fail(engine, "profile", "cards")

    assert waits == []


async def test_no_waiting_once_the_pin_is_spent(tmp_path, waits):
    # Without the PIN nothing can be reached again; retrying it later would
    # risk locking the test account.
    engine = _engine(tmp_path)
    engine._pin_failures = MAX_PIN_FAILURES

    assert await _fail(engine, "accounts", "payments", "profile") is False
    assert waits == []
    assert engine.graph.stop_reason == ENVIRONMENT_DOWN_REASON


async def test_the_main_loop_stops_instead_of_writing_off(tmp_path, waits, monkeypatch):
    # Every trip fails, as after run 8's outage: the loop must stop with a
    # reason and leave the frontier intact.
    async def nowhere(*args, **kwargs):
        return None

    monkeypatch.setattr(engine_module, "go_back_to_screen", nowhere)
    engine = _engine(tmp_path)
    engine.controller.tree = _app("Деньги", _button("Числовой код", 230))
    engine.max_steps = 50

    graph = await engine.run()

    assert graph.stop_reason == ENVIRONMENT_DOWN_REASON
    assert graph.written_off_count == 0
    assert engine.step_idx < 50


async def test_landing_on_another_screen_resets_the_count(tmp_path, waits, monkeypatch):
    engine = _engine(tmp_path)
    await _fail(engine, "accounts", "payments")
    engine.controller.tree = _app("Деньги", _button("Перевод", 300))

    async def lands(element):
        engine.current_screen_id = "profile"

    monkeypatch.setattr(engine, "_explore_tap", lands)
    engine.max_steps = 1
    await engine.run()

    assert engine._failed_navs == 0


async def test_a_successful_trip_resets_the_count(tmp_path, waits, monkeypatch):
    engine = _engine(tmp_path)
    await _fail(engine, "accounts", "payments")
    engine.controller.tree = _app("Деньги")

    async def arrives(controller, graph, current, target, *args, **kwargs):
        return target

    monkeypatch.setattr(engine_module, "go_back_to_screen", arrives)
    engine.max_steps = 1
    await engine.run()

    assert engine._failed_navs == 0
