"""Fixes from the fifth aweassist run (PER-892).

aweassist-5 spent 93 of 400 steps on «Счета»: an expanded panel that locks
scrolling while the page it covers stays in the tree. Each of those actions
was tried one by one, and the panel's real «Настройки» was missed: on screen
it is a text, and the button of that name sits 4000 points below.
"""

from __future__ import annotations

import pytest

from explorer.engine import ExplorationEngine
from explorer.models import ElementKind, ElementSnapshot, ScreenNode
from explorer.modes import ExplorationMode
from explorer.visualizer import generate_stats_report
from tests.test_aweassist_run1 import _client, _el
from tests.test_engine import _PopupController


def _frame(y):
    return {"x": 0, "y": y, "width": 402, "height": 60}


async def test_two_swipes_without_movement_mean_the_end_of_content(monkeypatch):
    stuck = _el("Button", "Маркет", 1500)
    client, calls, _ = _client(monkeypatch, [[stuck]])

    frame = await client.scroll_into_view("Button", "Маркет", "Маркет", _frame(1500))

    assert frame is None
    assert client.scroll_stuck
    assert sum("swipe" in c for c in calls) == 2


async def test_a_dropped_swipe_is_not_the_end_of_content(monkeypatch):
    # AXe drops input: the first swipe moved nothing, the second did.
    row = lambda y: _el("Button", "Маркет", y)
    client, _, _ = _client(monkeypatch, [[row(1500)], [row(1500)], [row(400)]])

    frame = await client.scroll_into_view("Button", "Маркет", "Маркет", _frame(1500))

    assert frame["y"] == 400
    assert not client.scroll_stuck


async def test_stuck_going_up_says_nothing_about_what_is_below(monkeypatch):
    above = _el("Button", "Шапка", -500)
    client, _, _ = _client(monkeypatch, [[above]])

    assert await client.scroll_into_view("Button", "Шапка", "Шапка", _frame(-500)) is None
    assert not client.scroll_stuck


async def test_carousel_item_does_not_count_as_end_of_content(monkeypatch):
    client, _, _ = _client(monkeypatch, [[]])
    client.scroll_stuck = True

    frame = {"x": 540, "y": 300, "width": 128, "height": 44}
    assert await client.scroll_into_view("GenericElement", "Ж/д", "Ж/д", frame) is None
    assert not client.scroll_stuck


async def test_visible_text_stands_in_for_an_unreachable_button(monkeypatch):
    client, _, _ = _client(monkeypatch, [[
        _el("StaticText", "Настройки", 416, height=20),
        _el("Button", "Настройки", 4535),
    ]])

    frame = await client.scroll_into_view("Button", "Настройки", "Настройки", _frame(4535))

    assert frame["y"] == 416


async def test_covered_text_does_not_stand_in(monkeypatch):
    client, _, _ = _client(monkeypatch, [[
        {**_el("StaticText", "Настройки", 416, height=20), "obscured": True},
        _el("Button", "Настройки", 4535),
    ]])

    assert await client.scroll_into_view("Button", "Настройки", "Настройки", _frame(4535)) is None


class _StuckController(_PopupController):
    """A panel that does not scroll: everything off screen is out of reach."""

    def __init__(self, stuck=True, found=None):
        super().__init__()
        self.scroll_stuck = False
        self._stuck, self._found = stuck, found
        self.tapped = []

    def is_on_screen(self, frame):
        return frame["y"] < 874

    def needs_scroll(self, element_type, frame):
        return not self.is_on_screen(frame)

    async def scroll_into_view(self, element_type, label, test_id, frame):
        self.scroll_stuck = self._stuck
        return self._found


def _action(label, y, type_="Button"):
    return ElementSnapshot(kind=ElementKind.BUTTON, element_type=type_, label=label, test_id=label, frame=_frame(y))


def _text(label, y):
    return ElementSnapshot(kind=ElementKind.LABEL, element_type="StaticText", label=label, frame=_frame(y))


def _setup(tmp_path, controller, actions, texts=()):
    engine = ExplorationEngine(controller=controller, app_bundle_id="com.aweassist.app",
                               output_dir=str(tmp_path), mode=ExplorationMode.MC)
    node = ScreenNode(screen_id="accounts", elements=[*actions, *texts], interactive_elements=actions)
    engine.graph.add_node(node)
    engine.current_screen_id = "accounts"
    return engine


def _left(engine):
    return {el.label for el in engine.graph.get_unexplored_actions("accounts")}


ACTIONS = [
    _action("Назад", 715), _action("Главный", 792, "RadioButton"), _action("Мои платежи", 966),
    _action("Накопления", 1215), _action("Настройки", 4535),
]


async def test_one_failed_attempt_settles_everything_below(tmp_path):
    engine = _setup(tmp_path, _StuckController(), ACTIONS)

    await engine._explore_tap(ACTIONS[2])

    assert _left(engine) == {"Назад", "Главный"}
    assert engine.graph.unreachable_count == 2  # the tried one is an ordinary failure
    assert "Out of scroll reach: 2" in generate_stats_report(engine.graph)


async def test_action_with_a_visible_text_is_kept(tmp_path):
    engine = _setup(tmp_path, _StuckController(), ACTIONS, texts=[_text("Настройки", 416)])

    await engine._explore_tap(ACTIONS[2])

    assert _left(engine) == {"Назад", "Главный", "Настройки"}


async def test_tab_bar_below_the_failed_element_is_kept(tmp_path):
    tab = _action("Чаты", 1000, "RadioButton")
    engine = _setup(tmp_path, _StuckController(), [ACTIONS[2], tab])

    await engine._explore_tap(ACTIONS[2])

    assert _left(engine) == {"Чаты"}


async def test_ordinary_failure_writes_off_only_itself(tmp_path):
    engine = _setup(tmp_path, _StuckController(stuck=False), ACTIONS)

    await engine._explore_tap(ACTIONS[2])

    assert _left(engine) == {"Назад", "Главный", "Накопления", "Настройки"}
    assert engine.graph.unreachable_count == 0
