"""The environment going down is not a reason to write screens off.

aweassist run 8: at 01:53 the test backend answered «no healthy upstream».
The agent took it for a wrong PIN, could not get anywhere afterwards and
wrote off 25 screens (42 actions) in 23 minutes of relaunches.
"""

from __future__ import annotations

import pytest

from explorer import engine as engine_module
from explorer.engine import ENVIRONMENT_DOWN_REASON, MAX_PIN_FAILURES, OUTAGE_WAITS_S, ExplorationEngine
from explorer.models import WRITTEN_OFF_NOTE, ActionDetail, ActionType, ElementKind, ElementSnapshot, GraphEdge, ScreenNode
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


async def test_a_number_pad_under_a_text_field_is_not_a_pin_pad(tmp_path):
    # aweassist run 9, «Выберите контакт»: the number pad of the field «Имя
    # или телефон» got the PIN twice; with the PIN budget spent nothing
    # could be reached after the next relaunch.
    from tests.test_engine import _keypad_tree

    tree = _keypad_tree("Деньги")
    tree.append({"type": "TextField", "label": "", "value": "Имя или телефон",
                                "test_id": "", "enabled": True,
                                "frame": {"x": 20, "y": 148, "width": 362, "height": 46}, "children": []})
    engine = _engine(tmp_path)
    engine.controller.tree = tree

    node = await engine._capture_screen()

    assert node.screen_id not in engine._keypad_screens


async def test_a_trip_through_several_screens_is_not_an_outage(tmp_path, waits):
    # Wikipedia run 2: paths replayed, the app moved from screen to screen,
    # only the targets (a changing feed) were never found again.
    engine = _engine(tmp_path)
    for target in ["accounts", "payments", "profile", "cards"]:
        engine._screens_seen_in_trip = {"home", "map"}
        await engine._navigation_failed(target, step=1)

    assert waits == []
    assert engine.graph.stop_reason is None


async def test_pin_pads_do_not_count_as_movement(tmp_path, waits):
    engine = _engine(tmp_path)
    for target in ["accounts", "payments", "profile"]:
        engine._screens_seen_in_trip = {"home"}
        await engine._navigation_failed(target, step=1)

    assert waits == [OUTAGE_WAITS_S[0]]


async def test_each_trip_counts_its_own_screens(tmp_path, waits, monkeypatch):
    async def nowhere(*args, **kwargs):
        return None

    monkeypatch.setattr(engine_module, "go_back_to_screen", nowhere)
    engine = _engine(tmp_path)
    engine.controller.tree = _app("Деньги", _button("Числовой код", 230))
    engine.max_steps = 50
    engine._screens_seen_in_trip = {"left", "over"}  # from an earlier trip

    graph = await engine.run()

    assert graph.stop_reason == ENVIRONMENT_DOWN_REASON


async def test_a_pin_pad_passed_on_the_way_is_not_movement(tmp_path):
    from tests.test_engine import _keypad_engine, _keypad_tree, _KeypadController

    engine = _keypad_engine(tmp_path, _KeypadController(), test_data={})
    await engine._capture_screen_id()

    assert engine._screens_seen_in_trip == set()


async def test_no_path_from_a_live_start_screen_is_not_an_outage(tmp_path, waits):
    # Wikipedia run 7: after each relaunch the app showed its start screen,
    # the graph had no path from it; six trips in a row stopped the run.
    engine = _engine(tmp_path)
    engine.graph.add_node(_node("start"))
    for target in ["accounts", "payments", "profile", "cards", "chats", "accounts"]:
        assert await engine._navigation_failed(target, step=1, landed="start")

    assert waits == []
    assert engine.graph.stop_reason is None
    assert engine._failed_navs == 0
    assert engine.graph.written_off_count > 0


async def test_dead_end_start_that_is_a_pin_pad_is_an_outage(tmp_path, waits):
    engine = _engine(tmp_path)
    engine.graph.add_node(_node("start"))
    engine._keypad_screens.add("start")
    for target in ["accounts", "payments", "profile"]:
        await engine._navigation_failed(target, step=1, landed="start")
    assert waits == [OUTAGE_WAITS_S[0]]


async def test_trip_that_captured_nothing_is_still_an_outage(tmp_path, waits):
    engine = _engine(tmp_path)
    for target in ["accounts", "payments", "profile"]:
        await engine._navigation_failed(target, step=1, landed=None)
    assert waits == [OUTAGE_WAITS_S[0]]


async def test_failed_replay_on_a_live_screen_is_not_an_outage(tmp_path, waits):
    # Wikipedia run 9: the app showed 'Community' after every relaunch and a
    # path to the target existed, but a replayed step kept failing; six trips
    # in a row stopped the run as «Environment unavailable».
    engine = _engine(tmp_path)
    engine.graph.add_node(_node("start"))
    for target in ["accounts", "payments", "profile", "cards", "chats"]:
        # The graph knows a path, so this is no dead-end start: the replay
        # itself failed.
        engine._record_replay_edge(
            "start", ActionDetail(action_type=ActionType.TAP, target_label=target), target
        )
    for target in ["accounts", "payments", "profile", "cards", "chats", "accounts"]:
        assert await engine._navigation_failed(target, step=1, landed="start")

    assert waits == []
    assert engine.graph.stop_reason is None
    assert engine._failed_navs == 0


def _route(engine, source, target, via=None, label="go"):
    """Edges source -> target (directly or through ``via``); returns them."""
    hops = [source, via, target] if via else [source, target]
    edges = []
    for a, b in zip(hops, hops[1:]):
        edge = GraphEdge(source_screen_id=a, target_screen_id=b,
                         action=ActionDetail(action_type=ActionType.TAP, target_label=f"{label}-{a}-{b}"))
        engine.graph.add_edge(edge)
        edges.append(edge)
    return edges


async def test_failed_route_on_a_live_screen_tries_another_before_giving_up(tmp_path, waits, monkeypatch):
    engine = _engine(tmp_path)
    for sid in ("start", "mid"):
        engine.graph.add_node(_node(sid))
    (direct,) = _route(engine, "start", "accounts")
    alternative = _route(engine, "start", "accounts", via="mid")
    direct.stale = False
    engine.graph.failed_edge = direct
    replayed = []

    async def replay(controller, path, *args, **kwargs):
        replayed.append(path)
        return "accounts"

    monkeypatch.setattr(engine_module, "replay_path", replay)

    assert await engine._navigation_failed("accounts", step=1, landed="start")

    assert direct.stale
    assert replayed == [alternative]
    assert engine.current_screen_id == "accounts"
    assert engine._nav_failures.get("accounts") is None
    assert engine.graph.written_off_count == 0


async def test_alternative_routes_are_bounded_and_then_the_target_is_unreachable(tmp_path, waits, monkeypatch):
    engine = _engine(tmp_path)
    for sid in ("start", "m1", "m2", "m3"):
        engine.graph.add_node(_node(sid))
    (direct,) = _route(engine, "start", "accounts")
    for mid in ("m1", "m2", "m3"):
        _route(engine, "start", "accounts", via=mid)
    engine.graph.failed_edge = direct
    replayed = []

    async def replay(controller, path, *args, **kwargs):
        replayed.append(path)
        engine.graph.failed_edge = path[0]  # the first step of each fails
        return "start"

    monkeypatch.setattr(engine_module, "replay_path", replay)

    assert await engine._navigation_failed("accounts", step=1, landed="start")

    assert len(replayed) == engine_module.MAX_ALTERNATIVE_ROUTES == 2  # a third route exists, not tried
    assert engine._nav_failures["accounts"] == 1  # only now counted as a failed trip
    assert engine.graph.stop_reason is None
