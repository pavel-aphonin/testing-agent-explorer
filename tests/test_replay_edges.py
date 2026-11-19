"""Navigation that lands somewhere new records how it got there.

Wikipedia runs 5 and 6: Back, relaunch and replay create nodes through the
capture with no incoming edge. find_path never reached them and their
actions were written off. Landing on a near-duplicate of the expected
target is not new (screen-family rule), and launch screens have no edge.
"""
import functools
import gzip
import json
from pathlib import Path

from explorer import navigator
from explorer.analyzer import is_variant
from explorer.models import ActionDetail, ActionType, AppGraph, GraphEdge, ScreenNode
from explorer.navigator import add_replay_edge, go_back_to_screen, relaunch_and_navigate, replay_path
from tests.test_variant_family import _screen

FIXTURES = Path(__file__).parent / "fixtures"


class _Result:
    error = None


class _Phone:
    """Taps move to whatever screen ``lands`` names; Back goes to ``back_to``."""

    def __init__(self, lands, here="A", back_to=None):
        self.lands, self.here, self.back_to = lands, here, back_to or {}

    async def scroll_into_view(self, element_type, label, test_id, frame):
        return frame

    async def tap_at(self, x, y):
        self.here = self.lands[self.here]
        return _Result()

    async def go_back(self):
        self.here = self.back_to.get(self.here, self.here)
        return True

    async def terminate_app(self, bundle_id):
        pass

    async def launch_app(self, bundle_id):
        self.here = "L"


def _tap(source, target, label="Go"):
    return GraphEdge(source_screen_id=source, target_screen_id=target, action=ActionDetail(
        action_type=ActionType.TAP, target_label=label, target_test_id=label,
        target_frame={"x": 0, "y": 100, "width": 100, "height": 40}))


def _graph(edges, *extra, nodes=None):
    graph = AppGraph(app_bundle_id="com.example")
    for sid in {s for e in edges for s in (e.source_screen_id, e.target_screen_id)} | set(extra):
        graph.add_node((nodes or {}).get(sid) or ScreenNode(screen_id=sid))
    for e in edges:
        graph.add_edge(e)
    return graph


def _run(monkeypatch, graph, phone, path):
    async def no_sleep(*_a, **_k):
        pass

    async def capture():
        node = (graph.nodes.get(phone.here)) or ScreenNode(screen_id=phone.here)
        graph.add_node(node)
        return phone.here, []

    monkeypatch.setattr("explorer.navigator.asyncio.sleep", no_sleep)
    record = functools.partial(add_replay_edge, graph)
    return capture, record


async def test_a_step_landing_elsewhere_gets_an_edge(monkeypatch):
    edge = _tap("A", "B")
    graph = _graph([edge], "B2")
    phone = _Phone({"A": "B2"})
    capture, record = _run(monkeypatch, graph, phone, [edge])

    assert await replay_path(phone, [edge], capture, graph=graph, on_transition=record) == "B2"

    assert [e.target_screen_id for e in graph.edges if e.notes == "replay"] == ["B2"]
    assert graph.find_path("A", "B2")
    # the next trip to B2 has a way there: not a dead end, not written off
    assert graph.find_path("A", "B2")[0].action.target_label == "Go"


async def test_without_the_callback_nothing_changes(monkeypatch):
    edge = _tap("A", "B")
    graph = _graph([edge], "B2")
    phone = _Phone({"A": "B2"})
    capture, _ = _run(monkeypatch, graph, phone, [edge])

    await replay_path(phone, [edge], capture, graph=graph)

    assert len(graph.edges) == 1
    assert not graph.find_path("A", "B2")


async def test_landing_on_the_target_adds_nothing(monkeypatch):
    edge = _tap("A", "B")
    graph = _graph([edge])
    phone = _Phone({"A": "B"})
    capture, record = _run(monkeypatch, graph, phone, [edge])

    await replay_path(phone, [edge], capture, graph=graph, on_transition=record)

    assert len(graph.edges) == 1


async def test_landing_on_a_variant_of_the_target_adds_no_edge(monkeypatch):
    b = _screen("B", ["Статья про кошек", "Новость дня", "Игра"])
    b2 = _screen("B2", ["Статья про лес", "Другая новость", "Фото дня", "Ещё"])
    assert is_variant(b, b2)
    edge = _tap("A", "B")
    graph = _graph([edge], "B2", nodes={"B": b, "B2": b2})
    phone = _Phone({"A": "B2"})
    capture, record = _run(monkeypatch, graph, phone, [edge])

    await replay_path(phone, [edge], capture, graph=graph, on_transition=record)

    assert len(graph.edges) == 1


async def test_a_second_replay_does_not_repeat_the_edge(monkeypatch):
    edge = _tap("A", "B")
    graph = _graph([edge], "B2")
    for _ in range(2):
        phone = _Phone({"A": "B2"})
        capture, record = _run(monkeypatch, graph, phone, [edge])
        await replay_path(phone, [edge], capture, graph=graph, on_transition=record)

    assert len(graph.edges) == 2


async def test_each_step_of_a_longer_path_is_checked(monkeypatch):
    ab, bc = _tap("A", "B", "one"), _tap("B", "C", "two")
    graph = _graph([ab, bc], "B2", "C2")
    phone = _Phone({"A": "B2", "B2": "C2"})
    capture, record = _run(monkeypatch, graph, phone, [ab, bc])

    await replay_path(phone, [ab, bc], capture, graph=graph, on_transition=record)

    # The first step landed off the path (B2, not B): the second is not tapped
    # blindly on it, the replay stops there.
    pairs = {(e.source_screen_id, e.target_screen_id) for e in graph.edges if e.notes == "replay"}
    assert pairs == {("A", "B2")}
    assert phone.here == "B2"


async def test_a_relaunch_landing_gets_no_edge(monkeypatch):
    graph = _graph([_tap("A", "B")], "L")
    phone = _Phone({}, here="B")
    capture, record = _run(monkeypatch, graph, phone, [])

    assert await relaunch_and_navigate(phone, graph, "B", capture, on_transition=record) == "L"

    assert len(graph.edges) == 1
    assert not any(e.target_screen_id == "L" for e in graph.edges)


async def test_back_onto_a_screen_gets_an_edge(monkeypatch):
    graph = _graph([_tap("A", "B")], "P", "Q")
    phone = _Phone({}, here="B", back_to={"B": "P"})
    capture, record = _run(monkeypatch, graph, phone, [])

    async def fast(*_a, **_k):
        return None

    monkeypatch.setattr(navigator, "_await_change", fast)
    await go_back_to_screen(phone, graph, "B", "P", capture, on_transition=record)

    back = [e for e in graph.edges if e.action.action_type == ActionType.BACK]
    assert [(e.source_screen_id, e.target_screen_id) for e in back] == [("B", "P")]


def test_edges_to_unknown_or_same_screens_are_not_made():
    graph = _graph([], "A", "B")
    act = ActionDetail(action_type=ActionType.TAP, target_label="x")
    assert not add_replay_edge(graph, "A", act, "A")
    assert not add_replay_edge(graph, "A", act, "ghost")
    assert not add_replay_edge(graph, "ghost", act, "B")
    assert graph.edges == []


def test_stored_run_edges_never_join_variants():
    raw = json.loads(gzip.open(FIXTURES / "variant_wikipedia_5.json.gz", "rt", encoding="utf-8").read())
    graph = AppGraph(app_bundle_id="org.wikimedia.wikipedia")
    for d in raw:
        node = ScreenNode(**d)
        if not any(is_variant(node, n) for n in graph.nodes.values()):
            graph.add_node(node)
    ids = list(graph.nodes)
    act = ActionDetail(action_type=ActionType.TAP, target_label="x")
    for source, landed in zip(ids, ids[1:]):
        add_replay_edge(graph, source, act, landed)
    # a chain of kept screens gives at most one edge per step; none of
    # them lies between near-duplicates
    assert len(graph.edges) <= len(ids) - 1
    assert not any(is_variant(graph.nodes[e.source_screen_id], graph.nodes[e.target_screen_id])
                   for e in graph.edges)


def test_the_engine_records_the_edge_and_the_screen_is_no_dead_end(tmp_path):
    from tests.test_variant_family import _engine

    engine = _engine(tmp_path)
    for sid in ("A", "B2"):
        engine.graph.add_node(ScreenNode(screen_id=sid))
    assert engine._is_dead_end_start("A", "B2")

    engine._record_replay_edge("A", ActionDetail(action_type=ActionType.TAP, target_label="Go"), "B2")

    assert not engine._is_dead_end_start("A", "B2")


class _CountingPhone(_Phone):
    taps = 0

    async def tap_at(self, x, y):
        type(self).taps += 1
        return await super().tap_at(x, y)


async def test_replay_does_not_tap_on_a_screen_other_than_the_step_source(monkeypatch):
    # A tap recorded for screen A must not hit screen X.
    edge = _tap("A", "B")
    graph = _graph([edge], "X")
    phone = _CountingPhone({"A": "B", "X": "B"}, here="X")
    _CountingPhone.taps = 0
    capture, record = _run(monkeypatch, graph, phone, [edge])

    landed = await replay_path(phone, [edge], capture, graph=graph, on_transition=record)

    assert landed == "X"
    assert _CountingPhone.taps == 0
    assert phone.here == "X"


async def test_replay_stops_midway_when_a_step_lands_off_the_path(monkeypatch):
    first, second = _tap("A", "B", "One"), _tap("B", "C", "Two")
    graph = _graph([first, second], "X")
    phone = _CountingPhone({"A": "X", "X": "X"})
    _CountingPhone.taps = 0
    capture, record = _run(monkeypatch, graph, phone, [first, second])

    landed = await replay_path(phone, [first, second], capture, graph=graph)

    assert landed == "X"
    assert _CountingPhone.taps == 1  # the second tap was not made
    assert graph.failed_edge is first


async def test_replay_accepts_a_variant_of_the_source(monkeypatch):
    a = _screen("A", ["Статья про кошек", "Новость дня", "Игра"])
    a2 = _screen("A2", ["Статья про лес", "Другая новость", "Фото дня", "Ещё"])
    assert is_variant(a, a2)
    edge = _tap("A", "B")
    graph = _graph([edge], "A2", nodes={"A": a, "A2": a2})
    phone = _Phone({"A2": "B"}, here="A2")
    capture, record = _run(monkeypatch, graph, phone, [edge])

    assert await replay_path(phone, [edge], capture, graph=graph, on_transition=record) == "B"


async def test_a_stale_edge_that_works_again_is_reset(monkeypatch):
    edge = _tap("A", "B")
    edge.stale = True
    graph = _graph([edge])
    phone = _Phone({"A": "B"})
    capture, record = _run(monkeypatch, graph, phone, [edge])

    await replay_path(phone, [edge], capture, graph=graph, on_transition=record)

    assert edge.stale is False


async def test_a_stale_twin_is_reset_when_the_transition_is_seen_again():
    edge = _tap("A", "B")
    edge.stale = True
    graph = _graph([edge])

    assert not add_replay_edge(graph, "A", edge.action, "B")

    assert edge.stale is False


async def test_a_failed_replay_after_back_goes_to_a_relaunch(monkeypatch):
    # Wikipedia run 10: Back, replay, the same broken step; Back again,
    # replay again — «Add some interests» failed three times in one step.
    class _Counting(_Phone):
        taps = 0
        launches = 0

        async def tap_at(self, x, y):
            self.taps += 1
            return await super().tap_at(x, y)

        async def launch_app(self, bundle_id):
            self.launches += 1
            await super().launch_app(bundle_id)

    graph = _graph([_tap("P", "T")], "C", "Q", "X")
    phone = _Counting({"P": "X", "Q": "X"}, here="C", back_to={"C": "P", "X": "P"})
    capture, _ = _run(monkeypatch, graph, phone, [])

    async def fast(*_a, **_k):
        return None

    monkeypatch.setattr(navigator, "_await_change", fast)
    monkeypatch.setattr(navigator, "_await_launch", fast)
    await go_back_to_screen(phone, graph, "C", "T", capture)

    assert phone.taps == 1
    assert phone.launches == 1
