"""An element that cannot be brought on screen stops cycling.

Wikipedia run 10: a Map's markers sat outside the scrollable frame. Each one
was chosen again and again, failed to scroll into view, and was recorded as
another self-loop, 56 «Could not bring … on screen» lines in one run. After
MAX_UNREACHABLE_ATTEMPTS failures on the same screen it is written off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from explorer import engine as engine_module
from explorer.models import UNREACHABLE_NOTE
from tests.test_engine import _app, _button, _popup_engine, _PopupController

# Matches explorer.engine.MAX_UNREACHABLE_ATTEMPTS; written out so the
# assertions read as the rule.
ATTEMPTS = 2
MARKER_FRAME = {"x": 100, "y": 300, "width": 100, "height": 40}


@dataclass
class _ScrollController(_PopupController):
    """Scroll results are taken from ``answers`` in order; once they run out
    the element is reachable at MARKER_FRAME."""

    answers: list = field(default_factory=list)
    scrolls: int = 0

    def needs_scroll(self, element_type, frame) -> bool:
        return True

    async def scroll_into_view(self, element_type, label, test_id, frame):
        self.scrolls += 1
        if self.answers:
            return self.answers.pop(0)
        return dict(MARKER_FRAME)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def sleep(seconds):
        return None

    monkeypatch.setattr(engine_module.asyncio, "sleep", sleep)


async def _map_engine(tmp_path, controller):
    engine = _popup_engine(tmp_path, controller)
    node = await engine._capture_screen()
    engine.graph.add_node(node)
    engine.current_screen_id = node.screen_id
    return engine, node


@pytest.mark.asyncio
async def test_unreachable_element_is_written_off_after_the_attempts(tmp_path):
    controller = _ScrollController(answers=[None] * 10, tree=_app("Testbed", _button("Marker", 900)))
    engine, node = await _map_engine(tmp_path, controller)
    marker = node.interactive_elements[0]

    for _ in range(3):
        await engine._explore_tap(marker)

    # Scrolled for twice, never a third time, never tapped.
    assert controller.scrolls == ATTEMPTS
    assert controller.taps == []
    # Written off: nothing left to explore on the screen, and it is marked
    # as out of reach like the other scrolling write-offs.
    assert engine.graph.get_unexplored_actions(node.screen_id) == []
    assert any(e.notes == UNREACHABLE_NOTE for e in engine.graph.edges)


@pytest.mark.asyncio
async def test_reachable_element_is_unaffected(tmp_path):
    controller = _ScrollController(tree=_app("Testbed", _button("Marker", 900)))
    engine, node = await _map_engine(tmp_path, controller)

    await engine._explore_tap(node.interactive_elements[0])

    assert controller.scrolls == 1
    assert controller.taps == [MARKER_FRAME["y"] + MARKER_FRAME["height"] // 2]
    assert not any(e.notes == UNREACHABLE_NOTE for e in engine.graph.edges)


@pytest.mark.asyncio
async def test_element_reached_on_the_second_try_is_not_written_off(tmp_path):
    controller = _ScrollController(answers=[None], tree=_app("Testbed", _button("Marker", 900)))
    engine, node = await _map_engine(tmp_path, controller)
    marker = node.interactive_elements[0]

    await engine._explore_tap(marker)  # fails once
    await engine._explore_tap(marker)  # reached now

    assert controller.scrolls == 2
    assert controller.taps == [MARKER_FRAME["y"] + MARKER_FRAME["height"] // 2]
    assert not any(e.notes == UNREACHABLE_NOTE for e in engine.graph.edges)
