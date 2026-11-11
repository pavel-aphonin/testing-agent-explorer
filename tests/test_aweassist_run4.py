"""Fixes from the fourth aweassist run (PER-892).

aweassist-4 reported 118 screens and 698 transitions. About a third of the
screens were variants: the main screen with one card more or less, the
profile with its collapsing title, one sheet above different tabs. Each
variant was explored from scratch («Поиск» tapped 12 times) and three of
them were written off, which added 308 transitions nobody ever tapped.
"""

from __future__ import annotations

import json

from explorer import axe_client
from explorer.analyzer import analyze_screen, is_variant
from explorer.axe_client import AXeExplorerClient
from explorer.engine import ExplorationEngine
from explorer.models import ActionDetail, ActionType, AppGraph, ElementKind, ElementSnapshot, GraphEdge, ScreenNode
from explorer.modes import ExplorationMode
from explorer.screen_id import compute_screen_id
from explorer.visualizer import generate_stats_report
from tests.test_engine import _app, _button, _PopupController

# Letters, not numbers: numbers in labels are normalized away.
CARDS = [f"Карта {letter}" for letter in "абвгдежзиклмнопрстуф"]
MAIN = ["Поиск", "Мои платежи", "Накопления", "Кэшбэк", "Вклады", "Кредиты", "Инвестиции", "Страхование"]


def _el(label, type_="Button", y=100):
    return ElementSnapshot(kind=ElementKind.BUTTON, element_type=type_, label=label, test_id=label,
                           frame={"x": 0, "y": y, "width": 402, "height": 40})


def _screen(screen_id, labels, heading=None, tabs=()):
    actions = [_el(label, y=100 + 40 * i) for i, label in enumerate(labels)]
    actions += [_el(tab, "RadioButton", 800) for tab in tabs]
    title = [_el(heading, "Heading", 60)] if heading else []
    return ScreenNode(screen_id=screen_id, elements=title + actions, interactive_elements=actions)


def test_one_card_more_is_a_variant():
    assert is_variant(_screen("a", MAIN + CARDS[:8]), _screen("b", MAIN + CARDS[:7]))


def test_expanded_panel_with_a_back_button_is_not_a_variant():
    # «Все мои продукты» on the main screen: the panel covers the content,
    # which stays in the tree, so 97 of 101 actions match. Only «Назад» and
    # the panel's own items tell the two apart.
    assert not is_variant(_screen("main", CARDS), _screen("accounts", CARDS + ["Назад", "Общий баланс"]))


def test_different_titles_are_different_screens():
    labels = MAIN

    assert not is_variant(_screen("a", labels, heading="Выбор карты"), _screen("b", labels, heading="Обзор активов"))


def test_title_that_comes_and_goes_is_a_variant():
    # The profile's large title leaves the tree once the header collapses.
    assert is_variant(_screen("a", MAIN), _screen("b", MAIN, heading="Станислав"))


def test_same_tab_bar_alone_is_no_evidence():
    tabs = ["Главный", "Платежи", "Выгода", "История", "Чаты"]

    assert not is_variant(_screen("a", ["Закрыть"], tabs=tabs), _screen("b", ["Закрыть"], tabs=tabs))


def _obscured(label, y=100):
    return {"type": "Button", "label": label, "frame": {"x": 0, "y": y, "width": 402, "height": 40}, "obscured": True}


def _top(type_, label, y=500):
    return {"type": type_, "label": label, "frame": {"x": 0, "y": y, "width": 402, "height": 40}}


def test_a_sheet_is_the_same_screen_above_any_background():
    sheet = [_top("Button", "Понятно"), _top("StaticText", "Голосовой ввод недоступен")]

    assert compute_screen_id([_obscured("Кэшбэк"), *sheet]) == compute_screen_id([_obscured("Фильтры"), *sheet])


def test_sheets_with_the_same_buttons_differ_by_their_text():
    delete = [_obscured("Выйти"), _top("Button", "Отмена"), _top("StaticText", "Удалить аккаунт?")]
    logout = [_obscured("Выйти"), _top("Button", "Отмена"), _top("StaticText", "Выйти из профиля?")]

    assert compute_screen_id(delete) != compute_screen_id(logout)


def test_text_still_does_not_change_an_ordinary_screen():
    assert compute_screen_id([_top("Button", "Поиск"), _top("StaticText", "Ошибка")]) == \
        compute_screen_id([_top("Button", "Поиск"), _top("StaticText", "Загружено")])


def test_analyzer_marks_a_screen_under_a_sheet_as_modal():
    assert analyze_screen([_obscured("Кэшбэк"), _top("Button", "Понятно")]).is_modal
    assert not analyze_screen([_top("Button", "Поиск")]).is_modal


def _engine(tmp_path, tree) -> ExplorationEngine:
    return ExplorationEngine(
        controller=_PopupController(tree=tree), app_bundle_id="com.aweassist.app",
        output_dir=str(tmp_path), mode=ExplorationMode.MC,
    )


async def _land(engine):
    node = await engine._capture_screen()
    is_new = engine.graph.add_node(node)
    await engine._on_screen_landed(node, is_new=is_new, step=0)
    return node


async def test_a_variant_lands_on_the_known_screen_and_keeps_its_new_actions(tmp_path):
    buttons = [_button(label, 100 + 40 * i) for i, label in enumerate(MAIN)]
    cards = [_button(label, 500 + 10 * i) for i, label in enumerate(CARDS[:10])]
    engine = _engine(tmp_path, _app("Деньги", *buttons, *cards))
    first = await _land(engine)

    engine.controller.tree = _app("Деньги", *buttons, *cards, _button("Новая карта", 700))
    again = await _land(engine)

    assert again.screen_id == first.screen_id
    assert len(engine.graph.nodes) == 1
    assert "Новая карта" in {el.label for el in engine.graph.nodes[first.screen_id].interactive_elements}
    engine.current_screen_id = again.screen_id
    tried = {el.label for el in engine.graph.get_unexplored_actions(again.screen_id)} - {"Новая карта"}
    for el in engine.graph.nodes[again.screen_id].interactive_elements:
        if el.label in tried:
            engine._record_self_loop(el, ActionType.TAP)
    only_left = engine._select_action(again.screen_id, engine.graph.get_unexplored_actions(again.screen_id))
    assert engine._action_id_to_element[only_left].label == "Новая карта"


async def test_a_variant_switches_taps_to_live_lookup(tmp_path):
    # Its frames belong to the first capture: a banner shifts everything.
    buttons = [_button(label, 100 + 40 * i) for i, label in enumerate(CARDS[:10])]
    engine = _engine(tmp_path, _app("Деньги", *buttons))
    first = await _land(engine)

    engine.controller.tree = _app("Деньги", *buttons, _button("Баннер", 60))
    await _land(engine)

    assert first.screen_id in engine._scrolled_screens


def _edge(source, target, label="tap"):
    return GraphEdge(source_screen_id=source, target_screen_id=target,
                     action=ActionDetail(action_type=ActionType.TAP, target_label=label, target_test_id=label))


def _graph_with_shared_sheet() -> AppGraph:
    graph = AppGraph(app_bundle_id="app")
    graph.add_node(ScreenNode(screen_id="benefits"))
    graph.add_node(ScreenNode(screen_id="history"))
    graph.add_node(ScreenNode(screen_id="sheet", is_modal=True))
    graph.add_node(ScreenNode(screen_id="inside"))
    for source, target in [("benefits", "sheet"), ("sheet", "benefits"), ("history", "sheet"),
                           ("sheet", "history"), ("sheet", "inside")]:
        graph.add_edge(_edge(source, target))
    return graph


def test_closing_a_shared_sheet_is_not_a_route_between_its_backgrounds():
    assert _graph_with_shared_sheet().find_path("benefits", "history") is None


def test_a_screen_behind_a_sheet_stays_reachable():
    path = _graph_with_shared_sheet().find_path("benefits", "inside")

    assert [e.target_screen_id for e in path] == ["sheet", "inside"]


def test_a_sheet_can_still_be_left_when_the_agent_is_on_it():
    assert _graph_with_shared_sheet().find_path("sheet", "history") is not None


async def test_write_offs_are_not_counted_as_transitions(tmp_path):
    engine = _engine(tmp_path, _app("Деньги", _button("Поиск", 100), _button("Кэшбэк", 200)))
    node = await _land(engine)
    engine.graph.add_edge(_edge(node.screen_id, node.screen_id, "Поиск"))

    engine._mark_screen_fully_explored(node.screen_id)

    report = generate_stats_report(engine.graph)
    assert "Transitions recorded: 1" in report
    assert "Written off untried: 1" in report


def _ax(type_, label=None, y=100, children=(), enabled=True):
    return {"type": type_, "AXLabel": label, "enabled": enabled, "children": list(children),
            "frame": {"x": 0, "y": y, "width": 402, "height": 60}}


def _row(title, y):
    return _ax("Group", y=y, children=[_ax("StaticText", title, y), _ax("Button", "chevron", y, enabled=False)])


async def test_rows_are_found_when_the_table_comes_without_its_scroll_area(monkeypatch):
    rows = [_row("Текущий счёт", 180), _row("Накопительный", 250)]
    tree = [_ax("Application", "Testbed", children=[
        _ax("Group", children=[_ax("Button", "Testbed", 60), _ax("Group", children=rows)]),
    ])]

    async def fake_run(args, timeout=15):
        return 0, json.dumps(tree), ""

    monkeypatch.setattr(axe_client, "_run", fake_run)
    client = AXeExplorerClient()
    client._width, client._height = 402, 874

    elements = await client.get_ui_elements()

    assert {el["label"] for el in elements if el["type"] == "Cell"} == {"Текущий счёт", "Накопительный"}
