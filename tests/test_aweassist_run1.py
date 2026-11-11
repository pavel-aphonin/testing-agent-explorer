"""Fixes from the first aweassist run (PER-892): tab bar, back button,
scrolling to off-screen elements, and the stuck rule.

aweassist-1: 62 screens, but 232 of 420 graph edges were write-offs, 49%
of actions lay below the fold, the bottom menu was invisible, and edge
swipes failed to go back 11 times.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from explorer import axe_client
from explorer.analyzer import analyze_screen
from explorer.axe_client import AXeExplorerClient
from explorer.engine import ExplorationEngine
from explorer.models import ElementKind
from explorer.modes import ExplorationMode
from explorer.navigator import go_back_to_screen
from tests.test_engine import _app, _button, _no_sleep, _PopupController, _Result


# ---------------------------------------------------------------- tab bar

def test_tab_bar_radio_buttons_are_actions():
    tabs = [{"type": "RadioButton", "label": name, "test_id": name, "enabled": True,
             "frame": {"x": 80 * i, "y": 800, "width": 80, "height": 50}}
            for i, name in enumerate(["Главный", "Платежи", "Выгода", "История", "Чаты"])]

    node = analyze_screen(tabs)

    assert [el.label for el in node.interactive_elements] == ["Главный", "Платежи", "Выгода", "История", "Чаты"]
    assert all(el.kind == ElementKind.BUTTON for el in node.interactive_elements)


# ------------------------------------------------------------ AXe client

def _client(monkeypatch, trees: list[list[dict]]):
    """AXe client reading ``trees`` in turn (the last repeats); records calls."""
    calls: list[list[str]] = []
    taps: list[tuple[int, int]] = []
    reads = iter(trees)
    last = {"tree": trees[0]}

    async def fake_run(args, timeout=15):
        calls.append(args)
        return 0, "", ""

    async def get_ui_elements(self):
        last["tree"] = next(reads, last["tree"])
        return last["tree"]

    async def tap_at(self, x, y):
        taps.append((x, y))
        return axe_client.TapResult()

    monkeypatch.setattr(axe_client, "_run", fake_run)
    monkeypatch.setattr(axe_client.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(AXeExplorerClient, "get_ui_elements", get_ui_elements)
    monkeypatch.setattr(AXeExplorerClient, "tap_at", tap_at)
    client = AXeExplorerClient()
    client._udid, client._width, client._height = "SIM-1", 402, 874
    return client, calls, taps


def _el(type_, label, y, test_id=None, x=0, width=402, height=60):
    return {"type": type_, "label": label, "test_id": test_id or "",
            "frame": {"x": x, "y": y, "width": width, "height": height}}


@pytest.mark.asyncio
async def test_go_back_taps_the_navigation_bar_back_button(monkeypatch):
    client, calls, taps = _client(monkeypatch, [[
        _el("Button", "Назад", 62, test_id="backButton", x=16, width=44, height=44),
        _el("Button", "Назад", 700),  # a «Назад» in content is not the bar's
    ]])

    await client.go_back()

    assert taps == [(38, 84)]
    assert not any("swipe" in c for c in calls)


@pytest.mark.asyncio
async def test_go_back_swipes_when_there_is_no_back_button(monkeypatch):
    client, calls, taps = _client(monkeypatch, [[_el("Button", "Станислав, профиль", 62)]])

    await client.go_back()

    assert taps == []
    assert any("swipe" in c for c in calls)


@pytest.mark.asyncio
async def test_scroll_into_view_scrolls_until_the_element_is_reachable(monkeypatch):
    target = lambda y: _el("Button", "Курсы валют и металлов", y)
    client, calls, _ = _client(monkeypatch, [
        [target(2821)], [target(2300)], [target(1700)], [target(1100)], [target(500)],
    ])

    frame = await client.scroll_into_view("Button", "Курсы валют и металлов",
                                          "Курсы валют и металлов", {"x": 0, "y": 2821, "width": 402, "height": 64})

    assert frame["y"] == 500
    assert sum("swipe" in c for c in calls) == 4


def _swipe_length(call: list[str]) -> float:
    return abs(float(call[call.index("--start-y") + 1]) - float(call[call.index("--end-y") + 1]))


@pytest.mark.asyncio
async def test_short_offset_gets_a_short_swipe(monkeypatch):
    # aweassist run 3: a fixed swipe of over half a screen overshot rows
    # just under the tab bar and bounced the list up and down.
    row = lambda y: _el("Button", "Обо мне", y, height=56)
    client, calls, _ = _client(monkeypatch, [[row(790)], [row(560)]])

    frame = await client.scroll_into_view("Button", "Обо мне", "Обо мне",
                                          {"x": 0, "y": 790, "width": 402, "height": 56})

    assert frame["y"] == 560
    (swipe,) = [c for c in calls if "swipe" in c]
    assert _swipe_length(swipe) < 400


@pytest.mark.asyncio
async def test_end_of_content_taps_visible_element_in_place(monkeypatch):
    # The list cannot scroll further: the row stays under the tab bar but
    # is on screen, so it is tapped where it is after one swipe — not after
    # eight swipes that move nothing and then a skip.
    row = lambda: _el("Button", "Адрес проживания", 790, height=56)
    client, calls, _ = _client(monkeypatch, [[row()], [row()]])

    frame = await client.scroll_into_view("Button", "Адрес проживания", "Адрес проживания",
                                          {"x": 0, "y": 790, "width": 402, "height": 56})

    assert frame["y"] == 790
    assert sum("swipe" in c for c in calls) == 1


@pytest.mark.asyncio
async def test_end_of_content_gives_up_on_off_screen_element(monkeypatch):
    stuck = lambda: _el("Button", "Скрытый пункт", 1500)
    client, calls, _ = _client(monkeypatch, [[stuck()], [stuck()]])

    frame = await client.scroll_into_view("Button", "Скрытый пункт", "Скрытый пункт",
                                          {"x": 0, "y": 1500, "width": 402, "height": 60})

    assert frame is None
    assert sum("swipe" in c for c in calls) == 2  # one more, for a dropped swipe


@pytest.mark.asyncio
async def test_scroll_into_view_keeps_the_ordinal_among_identical_labels(monkeypatch):
    # Two cards both labelled «Кэшбэк 10%»; we want the second one.
    cards = lambda dy: [_el("GenericElement", "Кэшбэк 10%", 400 + dy),
                        _el("GenericElement", "Кэшбэк 10%", 1400 + dy)]
    client, _, _ = _client(monkeypatch, [cards(0), cards(-900)])

    frame = await client.scroll_into_view("GenericElement", "Кэшбэк 10%", "Кэшбэк 10%",
                                          {"x": 0, "y": 1400, "width": 402, "height": 60})

    assert frame["y"] == 500


def test_bar_items_need_no_scroll_but_content_under_the_tab_bar_does(monkeypatch):
    # Testbed run 11: margins treated as off screen skipped every «Назад»,
    # «Отмена» and «Сохранить» in the navigation bar.
    client, _, _ = _client(monkeypatch, [[]])

    back = {"x": 16, "y": 62, "width": 44, "height": 44}
    tab = {"x": 80, "y": 800, "width": 80, "height": 50}
    card_under_tab_bar = {"x": 0, "y": 780, "width": 402, "height": 60}
    below_fold = {"x": 0, "y": 2821, "width": 402, "height": 64}

    assert not client.needs_scroll("Button", back)
    assert not client.needs_scroll("RadioButton", tab)
    assert client.needs_scroll("Button", card_under_tab_bar)
    assert client.needs_scroll("Button", below_fold)


@pytest.mark.asyncio
async def test_scroll_into_view_returns_navigation_bar_items_in_place(monkeypatch):
    client, calls, _ = _client(monkeypatch, [[_el("Button", "Отмена", 82, x=16, width=80, height=44)]])

    frame = await client.scroll_into_view("Button", "Отмена", "Отмена",
                                          {"x": 16, "y": 82, "width": 80, "height": 44})

    assert frame["y"] == 82
    assert calls == []


# -------------------------------------------------------------- navigator

@dataclass
class _NoBackController:
    backs: int = 0
    launches: int = 0

    async def go_back(self):
        self.backs += 1
        return True

    async def terminate_app(self, bundle_id):
        pass

    async def launch_app(self, bundle_id):
        self.launches += 1


@pytest.mark.asyncio
async def test_back_that_does_not_move_gives_up_after_two_tries(monkeypatch):
    monkeypatch.setattr("explorer.navigator.asyncio.sleep", _no_sleep)
    from explorer.models import AppGraph

    graph = AppGraph(app_bundle_id="com.aweassist.app")
    controller = _NoBackController()

    async def capture():
        return ("accounts", [])

    await go_back_to_screen(controller, graph, "accounts", "home", capture)

    assert controller.backs == 2       # not 5
    assert controller.launches == 1    # straight to relaunch


# ----------------------------------------------------------------- engine

@dataclass
class _LongScreenController(_PopupController):
    scrolls: list = field(default_factory=list)

    async def scroll_into_view(self, element_type, label, test_id, frame):
        self.scrolls.append(label)
        return {"x": 0, "y": 500, "width": 402, "height": 60}

    def needs_scroll(self, element_type, frame):
        return frame["y"] >= 760


def _engine(tmp_path, controller):
    engine = ExplorationEngine(
        controller=controller, app_bundle_id="com.aweassist.app",
        output_dir=str(tmp_path), mode=ExplorationMode.MC,
    )
    engine._app_label = "Деньги"
    return engine


@pytest.mark.asyncio
async def test_off_screen_element_is_scrolled_into_view_before_tapping(tmp_path, monkeypatch):
    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    controller = _LongScreenController(tree=_app("Деньги", _button("Поиск", 300), _button("Настройки", 2900)))
    engine = _engine(tmp_path, controller)
    node = await engine._capture_screen()
    engine.graph.add_node(node)
    engine.current_screen_id = node.screen_id
    settings = [el for el in node.interactive_elements if el.label == "Настройки"][0]

    await engine._explore_tap(settings)

    assert controller.scrolls == ["Настройки"]
    assert controller.taps == [530]  # the scrolled frame, not y=2920
    assert node.screen_id in engine._scrolled_screens


@dataclass
class _LateTransitionController(_PopupController):
    """The tapped screen changes only on the 2nd read after the tap."""

    reads_after_tap: int = -1
    target: list = field(default_factory=list)

    async def get_ui_elements(self):
        if self.reads_after_tap >= 0:
            self.reads_after_tap += 1
            if self.reads_after_tap >= 2:
                self.tree = self.target
        return self.tree

    async def tap_at(self, x, y):
        self.taps.append(y)
        self.reads_after_tap = 0
        return _Result()


@pytest.mark.asyncio
async def test_late_transition_is_not_recorded_as_self_loop(tmp_path, monkeypatch):
    # Testbed run 12: a back tap landed after the capture and was recorded
    # as «same screen», so profile → home was never found.
    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    controller = _LateTransitionController(
        tree=_app("Деньги", _button("Назад", 200)),
        target=_app("Деньги", _button("Профиль", 300), _button("Счета", 400)),
    )
    engine = _engine(tmp_path, controller)
    node = await engine._capture_screen()
    engine.graph.add_node(node)
    engine.current_screen_id = node.screen_id

    await engine._explore_tap(node.interactive_elements[0])

    (edge,) = engine.graph.edges
    assert edge.target_screen_id != node.screen_id
    assert {el.label for el in engine.graph.nodes[edge.target_screen_id].interactive_elements} == {"Профиль", "Счета"}


@pytest.mark.asyncio
async def test_stuck_screen_is_left_not_written_off(tmp_path, monkeypatch):
    # Every tap self-loops, like information tiles. The rest of the screen
    # must stay unexplored for later, not be marked explored.
    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    monkeypatch.setattr("explorer.engine.SETTLE_DELAY", 0)
    buttons = [_button(f"Плитка {i}", 100 + 40 * i) for i in range(15)]
    controller = _PopupController(tree=_app("Деньги", *buttons))
    engine = ExplorationEngine(
        controller=controller, app_bundle_id="com.aweassist.app",
        output_dir=str(tmp_path), mode=ExplorationMode.MC, max_steps=4,
    )

    await engine.run()

    written_off = [e for e in engine.graph.edges if (e.notes or "").startswith("Action failed")]
    assert written_off == []
    assert len(engine.graph.get_unexplored_actions(engine.current_screen_id)) == 15 - len(controller.taps)


@pytest.mark.asyncio
async def test_watchdog_counts_explored_actions_as_progress(tmp_path, monkeypatch):
    # 40 tiles that all self-loop: no new screens for 40 steps, but every
    # step explores something new — the dead-end watchdog (25) must not fire.
    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    monkeypatch.setattr("explorer.engine.SETTLE_DELAY", 0)
    buttons = [_button(f"Плитка {i}", 100 + 20 * i) for i in range(40)]
    controller = _PopupController(tree=_app("Деньги", *buttons))
    engine = ExplorationEngine(
        controller=controller, app_bundle_id="com.aweassist.app",
        output_dir=str(tmp_path), mode=ExplorationMode.MC, max_steps=100,
    )

    await engine.run()

    assert len(set(controller.taps)) == 40
