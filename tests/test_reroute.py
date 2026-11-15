"""A path step that no longer replays is stale; the agent finds another way.

Wikipedia run (wave 1): «Map» and «Search» were written off because their
only recorded path went through «Enable location», a prompt answered once.
"""

from __future__ import annotations

from explorer.models import ActionDetail, ActionType, AppGraph, GraphEdge, ScreenNode
from explorer.navigator import replay_path


class _Result:
    def __init__(self, error=None):
        self.error = error


class _Screen:
    """Taps by label move between screens; missing labels cannot be found."""

    def __init__(self, links, start="home"):
        self.links, self.here, self.taps, self.lookups = links, start, [], 0

    async def scroll_into_view(self, element_type, label, test_id, frame):
        self.lookups += 1
        return frame if (self.here, label) in self.links else None

    async def tap_at(self, x, y):
        return _Result()

    async def tap_label(self, label):
        self.taps.append(label)
        self.here = self.links[(self.here, label)]


def _edge(source, target, label):
    return GraphEdge(source_screen_id=source, target_screen_id=target,
                     action=ActionDetail(action_type=ActionType.TAP, target_label=label, target_test_id=label,
                                         target_frame={"x": 0, "y": 100, "width": 100, "height": 40}))


def _graph(*edges):
    graph = AppGraph(app_bundle_id="org.wikimedia.wikipedia")
    for screen in {s for e in edges for s in (e.source_screen_id, e.target_screen_id)}:
        graph.add_node(ScreenNode(screen_id=screen))
    for e in edges:
        graph.add_edge(e)
    return graph


async def _replay(monkeypatch, graph, screen, path):
    async def no_sleep(*_):
        pass

    monkeypatch.setattr("explorer.navigator.asyncio.sleep", no_sleep)
    original = screen.scroll_into_view

    async def scroll(element_type, label, test_id, frame):
        found = await original(element_type, label, test_id, frame)
        if found is not None:
            await screen.tap_label(label)
        return found

    screen.scroll_into_view = scroll

    async def capture():
        return screen.here, []

    return await replay_path(screen, path, capture, graph=graph)


PROMPT = _edge("home", "map", "Enable location")
VIA_MENU = [_edge("home", "menu", "Menu"), _edge("menu", "map", "Map")]


async def test_a_gone_step_is_rerouted_around(monkeypatch):
    graph = _graph(PROMPT, *VIA_MENU)
    screen = _Screen({("home", "Menu"): "menu", ("menu", "Map"): "map"})

    reached = await _replay(monkeypatch, graph, screen, [PROMPT])

    assert reached == "map"
    assert PROMPT.stale


async def test_paths_avoid_stale_steps_while_others_exist():
    graph = _graph(PROMPT, *VIA_MENU)
    PROMPT.stale = True
    try:
        assert [e.action.target_label for e in graph.find_path("home", "map")] == ["Menu", "Map"]
    finally:
        PROMPT.stale = False


async def test_a_stale_step_is_still_used_when_it_is_the_only_way():
    prompt = _edge("home", "map", "Enable location")
    prompt.stale = True
    graph = _graph(prompt)
    assert graph.find_path("home", "map") == [prompt]


async def test_no_detour_means_stopping_where_we_are(monkeypatch):
    prompt = _edge("home", "map", "Enable location")
    graph = _graph(prompt)
    screen = _Screen({})

    assert await _replay(monkeypatch, graph, screen, [prompt]) == "home"
    assert prompt.stale
    assert screen.lookups == 1  # the same gone step is not retried as a «detour»
