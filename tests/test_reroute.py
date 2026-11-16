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
    from explorer.navigator import ELEMENT_TRIES

    # Waited for a while, then not retried again as a «detour».
    assert screen.lookups == ELEMENT_TRIES


class _Polled:
    """A screen read in order: ``screens[i]`` on the i-th read."""

    def __init__(self, screens):
        self.screens, self.reads = screens, 0

    async def get_ui_elements(self):
        screen = self.screens[min(self.reads, len(self.screens) - 1)]
        self.reads += 1
        return screen


def _tree(*labels):
    return [{"type": "Button", "label": l, "frame": {"x": 0, "y": 100 + 50 * i, "width": 100, "height": 40}}
            for i, l in enumerate(labels)]


async def _wait(monkeypatch, screens, max_s=3.5):
    from explorer import navigator
    from explorer.screen_id import native_fingerprint

    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(navigator.asyncio, "sleep", sleep)
    controller = _Polled(screens)
    await navigator._await_change(controller, native_fingerprint(_tree("Назад", "Карта")), max_s, 1.5)
    return controller.reads, slept


async def test_navigation_moves_on_once_the_new_screen_holds(monkeypatch):
    reads, _ = await _wait(monkeypatch, [_tree("Меню"), _tree("Меню")])
    assert reads == 2


async def test_navigation_waits_while_the_screen_still_changes(monkeypatch):
    reads, _ = await _wait(monkeypatch, [_tree("Меню"), _tree("Меню", "Поиск"), _tree("Меню", "Поиск")])
    assert reads == 3


async def test_navigation_gives_a_still_screen_the_full_window(monkeypatch):
    from explorer import navigator

    reads, _ = await _wait(monkeypatch, [_tree("Назад", "Карта")], max_s=1.5)
    assert reads == round(1.5 / navigator.CHANGE_POLL_S)


async def test_a_controller_that_cannot_read_gets_the_fixed_wait(monkeypatch):
    from explorer import navigator

    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(navigator.asyncio, "sleep", sleep)
    await navigator._await_change(object(), None, 3.5, 1.5)
    assert slept == [1.5]


async def test_replayed_steps_wait_for_the_screen_not_a_fixed_delay(monkeypatch):
    from explorer import navigator

    class _Readable(_Screen):
        async def get_ui_elements(self):
            return _tree(self.here)

    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    graph = _graph(*VIA_MENU)
    screen = _Readable({("home", "Menu"): "menu", ("menu", "Map"): "map"})
    original = screen.scroll_into_view

    async def scroll(element_type, label, test_id, frame):
        found = await original(element_type, label, test_id, frame)
        if found is not None:
            await screen.tap_label(label)
        return found

    screen.scroll_into_view = scroll
    monkeypatch.setattr(navigator.asyncio, "sleep", sleep)

    async def capture():
        return screen.here, []

    assert await replay_path(screen, list(VIA_MENU), capture, graph=graph) == "map"
    assert navigator.SETTLE_DELAY not in slept


class _Launching(_Screen):
    """Relaunch opens ``start`` (a new variant of the feed)."""

    def __init__(self, links, start):
        super().__init__(links, start="?")
        self.start, self.launches = start, 0

    async def terminate_app(self, bundle_id):
        pass

    async def launch_app(self, bundle_id):
        self.launches += 1
        self.here = self.start


async def _relaunch(monkeypatch, graph, screen, target):
    from explorer import navigator

    async def no_sleep(*_):
        pass

    monkeypatch.setattr(navigator.asyncio, "sleep", no_sleep)
    original = screen.scroll_into_view

    async def scroll(element_type, label, test_id, frame):
        found = await original(element_type, label, test_id, frame)
        if found is not None:
            await screen.tap_label(label)
        return found

    screen.scroll_into_view = scroll

    async def capture():
        return screen.here, []

    return await navigator.relaunch_and_navigate(screen, graph, target, capture)


async def test_a_new_launch_variant_uses_the_paths_of_an_earlier_one(monkeypatch):
    graph = _graph(_edge("feed-old", "map", "Map"))
    graph.launch_screens.append("feed-old")
    graph.add_node(ScreenNode(screen_id="feed-new"))
    screen = _Launching({("feed-new", "Map"): "map"}, start="feed-new")

    assert await _relaunch(monkeypatch, graph, screen, "map") == "map"
    assert graph.launch_screens == ["feed-old", "feed-new"]


async def test_without_earlier_launch_screens_there_is_no_path(monkeypatch):
    graph = _graph(_edge("feed-old", "map", "Map"))
    graph.add_node(ScreenNode(screen_id="feed-new"))
    screen = _Launching({("feed-new", "Map"): "map"}, start="feed-new")

    assert await _relaunch(monkeypatch, graph, screen, "map") == "feed-new"


async def test_a_failed_step_is_diagnosed(monkeypatch):
    seen = []

    async def diagnose(kind, edge):
        seen.append((kind, edge.action.target_label))

    from explorer import navigator

    async def no_sleep(*_):
        pass

    monkeypatch.setattr(navigator.asyncio, "sleep", no_sleep)
    prompt = _edge("home", "map", "Enable location")
    screen = _Screen({})

    async def capture():
        return screen.here, []

    await replay_path(screen, [prompt], capture, graph=_graph(prompt), diagnose=diagnose)
    assert seen == [("replay-failed", "Enable location")]


async def test_a_replayed_step_waits_for_its_element_to_appear(monkeypatch):
    from explorer import navigator

    async def no_sleep(*_):
        pass

    monkeypatch.setattr(navigator.asyncio, "sleep", no_sleep)
    looks = []

    async def scroll(element_type, label, test_id, frame):
        looks.append(label)
        return frame if len(looks) >= 3 else None  # the screen is still opening

    action = _edge("home", "map", "Map").action
    assert await navigator._await_element(scroll, action) == action.target_frame
    assert len(looks) == 3


async def test_idle_polls_slow_down(monkeypatch):
    from explorer import navigator

    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(navigator.asyncio, "sleep", sleep)
    from explorer.screen_id import native_fingerprint

    still = _tree("Назад", "Карта")
    await navigator._await_change(_Polled([still]), native_fingerprint(still), 3.5, 1.5)
    assert slept[0] == navigator.CHANGE_POLL_S
    assert slept[-1] > slept[0]
    assert len(slept) <= round(3.5 / navigator.CHANGE_POLL_S)


async def test_only_the_last_step_waits_for_the_screen_to_settle(monkeypatch):
    from explorer import navigator

    class _Readable(_Screen):
        async def get_ui_elements(self):
            return _tree(self.here)

    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(navigator.asyncio, "sleep", sleep)
    graph = _graph(*VIA_MENU)
    screen = _Readable({("home", "Menu"): "menu", ("menu", "Map"): "map"})
    original = screen.scroll_into_view

    async def scroll(element_type, label, test_id, frame):
        found = await original(element_type, label, test_id, frame)
        if found is not None:
            await screen.tap_label(label)
        return found

    screen.scroll_into_view = scroll

    async def capture():
        return screen.here, []

    await replay_path(screen, list(VIA_MENU), capture, graph=graph)
    assert slept.count(navigator.STEP_ANIMATION_S) == 1  # after «Menu», not after the last step
