"""Testbed input traps (runs traps-1 and traps-2).

«Лента» / «Избранное»: two tab roots with nothing but text of their own had
one fingerprint, so switching tabs was a self-loop and «Избранное» was never
recorded. «Поиск»: «Найти» is disabled until a query is typed; the screen id
ignores enabled state, the stored node kept its old actions, and «Найти» was
never tapped.
"""

from __future__ import annotations

import pytest

from explorer.screen_id import compute_screen_id
from tests.test_engine import _app, _button, _popup_engine, _PopupController


def _tab(label: str, selected: bool, x: int) -> dict:
    return {"type": "RadioButton", "label": label, "value": "1" if selected else "0",
            "identifier": "", "enabled": True,
            "frame": {"x": x, "y": 795, "width": 97, "height": 54}, "children": []}


def _heading(label: str) -> dict:
    return {"type": "Heading", "label": label, "value": None, "identifier": "",
            "enabled": True, "frame": {"x": 160, "y": 78, "width": 80, "height": 26},
            "children": []}


def test_tab_roots_differ_by_the_selected_tab():
    feed = [_heading("Лента"), _tab("Лента", True, 108), _tab("Избранное", False, 197)]
    saved = [_heading("Избранное"), _tab("Лента", False, 108), _tab("Избранное", True, 197)]

    assert compute_screen_id(feed) != compute_screen_id(saved)


def test_one_tab_root_keeps_its_id():
    a = [_heading("Лента"), _tab("Лента", True, 108), _tab("Избранное", False, 197)]
    b = [_heading("Лента (обновлено)"), _tab("Лента", True, 108), _tab("Избранное", False, 197)]

    assert compute_screen_id(a) == compute_screen_id(b)


def _search(enabled: bool) -> list[dict]:
    submit = _button("Найти", 300)
    submit["enabled"] = enabled
    return _app("Testbed", _button("Назад", 80), submit)


@pytest.mark.asyncio
async def test_a_button_that_became_enabled_becomes_an_action(tmp_path):
    controller = _PopupController(tree=_search(enabled=False))
    engine = _popup_engine(tmp_path, controller)
    first = await engine._capture_screen()
    engine.graph.add_node(first)
    assert "Найти" not in {el.label for el in first.interactive_elements}

    controller.tree = _search(enabled=True)
    second = await engine._capture_screen()

    assert second.screen_id == first.screen_id
    stored = engine.graph.nodes[first.screen_id]
    assert "Найти" in {el.label for el in stored.interactive_elements}
    assert "Найти" in {el.label for el in engine.graph.get_unexplored_actions(first.screen_id)}


@pytest.mark.asyncio
async def test_new_content_under_the_same_id_is_not_added(tmp_path):
    # Only buttons the stored node lists as disabled are added: rows that a
    # feed shows under the same id are content.
    controller = _PopupController(tree=_search(enabled=False))
    engine = _popup_engine(tmp_path, controller)
    first = await engine._capture_screen()
    engine.graph.add_node(first)
    before = len(first.interactive_elements)

    row = {"type": "Cell", "label": "Статья дня", "value": None, "identifier": "",
           "enabled": True, "frame": {"x": 0, "y": 500, "width": 400, "height": 44},
           "children": []}
    controller.tree = [*_search(enabled=False), row]
    second = await engine._capture_screen()

    if second.screen_id == first.screen_id:
        assert len(engine.graph.nodes[first.screen_id].interactive_elements) == before


@pytest.mark.asyncio
async def test_a_rejected_pin_on_launch_is_the_stop_reason(tmp_path, monkeypatch):
    # aweassist run 10: the PIN was «not accepted» twice on the launch
    # screen; two screens were found and the report said «Stopped early: no».
    from dataclasses import dataclass

    from explorer.engine import PIN_REJECTED_REASON
    from tests.test_engine import _KeypadController, _keypad_engine, _no_sleep

    @dataclass
    class _Locked(_KeypadController):
        async def terminate_app(self, bundle_id):
            pass

        async def launch_app(self, bundle_id):
            pass

        async def go_back(self):
            return True

    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    monkeypatch.setattr("explorer.navigator.asyncio.sleep", _no_sleep)
    engine = _keypad_engine(tmp_path, _Locked(), test_data={"pin": "9999"})
    engine.max_steps = 20

    graph = await engine.run()

    assert graph.stop_reason == PIN_REJECTED_REASON
