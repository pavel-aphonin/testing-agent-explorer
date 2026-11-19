"""An edge that fails to replay twice is dead.

Wikipedia run 10: «Add some interests» left the For You feed once the agent
had picked interests. The edge stayed the only way to «Your interests», so
every trip there (and to the screens behind it) replayed it again: 8 failed
replays in 4 trips, each a relaunch plus a 3.5 s wait for the element. A dead
edge is used by no route, and a frontier only dead edges lead to is written
off without a trip.
"""

from __future__ import annotations

import pytest

from explorer.models import MAX_EDGE_FAILURES
from tests.test_engine import _app, _button, _popup_engine, _PopupController
from tests.test_reroute import _Screen, _edge, _graph, _replay


def _interests_graph():
    gone = _edge("feed", "interests", "Add some interests")
    behind = _edge("interests", "topic", "Art")
    return gone, behind, _graph(gone, behind)


async def test_two_failed_replays_kill_an_edge(monkeypatch):
    gone, _, graph = _interests_graph()
    screen = _Screen({}, start="feed")

    for _ in range(MAX_EDGE_FAILURES):
        assert not gone.dead
        await _replay(monkeypatch, graph, screen, [gone])
        screen.here = "feed"

    assert gone.dead
    assert graph.find_path("feed", "interests") is None
    assert graph.find_path("feed", "topic") is None


async def test_a_stale_edge_is_still_the_only_way_until_it_is_dead():
    gone, _, graph = _interests_graph()
    gone.replay_failed()

    assert not gone.dead
    assert graph.find_path("feed", "interests") == [gone]


async def test_an_edge_that_works_again_starts_counting_from_zero():
    gone, _, _ = _interests_graph()
    gone.replay_failed()
    gone.replay_worked()
    gone.replay_failed()

    assert not gone.dead and gone.stale


def test_reachability_without_and_with_dead_edges():
    gone, _, graph = _interests_graph()
    gone.failures = MAX_EDGE_FAILURES

    assert graph.reachable_from({"feed"}) == {"feed"}
    assert graph.reachable_from({"feed"}, include_dead=True) == {"feed", "interests", "topic"}


@pytest.mark.asyncio
async def test_the_engine_sees_screens_cut_off_by_dead_edges(tmp_path):
    engine = _popup_engine(tmp_path, _PopupController(tree=_app("Testbed", _button("Ok", 100))))
    gone, behind, graph = _interests_graph()
    other = _edge("settings", "elsewhere", "About")
    graph.add_edge(other)
    for sid in ("settings", "elsewhere"):
        graph.nodes.setdefault(sid, graph.nodes["feed"].model_copy(update={"screen_id": sid}))
    engine.graph = graph
    graph.launch_screens = ["feed"]
    engine.current_screen_id = "feed"

    assert not engine._cut_off_by_dead_edges("topic")  # nothing is dead yet
    gone.failures = MAX_EDGE_FAILURES
    assert engine._cut_off_by_dead_edges("interests")
    assert engine._cut_off_by_dead_edges("topic")
    # No route at all, dead or alive: left to the trip and its Back presses.
    assert not engine._cut_off_by_dead_edges("elsewhere")
