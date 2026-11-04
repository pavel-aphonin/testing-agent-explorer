"""Fixes from the second aweassist run (PER-892).

aweassist-2: «Счета» got a new fingerprint on every visit (scroll-bar label
«1 page» / «2 pages», a reload button while an error banner was up), so
navigation never «reached» it and 116 of its 117 actions were written off;
the archive kept sending the agent there anyway; bottom-sheet items were
treated as hidden under a tab bar that a modal covered.
"""

from __future__ import annotations

import json

import pytest

from explorer import axe_client
from explorer.axe_client import AXeExplorerClient
from explorer.engine import ExplorationEngine
from explorer.models import ElementKind, ElementSnapshot, ScreenNode
from explorer.modes import ExplorationMode
from explorer.screen_id import compute_screen_id
from tests.test_engine import _PopupController


def _el(type_, label, y=100):
    return {"type": type_, "label": label, "frame": {"x": 0, "y": y, "width": 402, "height": 40}}


ACCOUNTS = [
    _el("Button", "Общий баланс"),
    _el("GenericElement", "Общий баланс, 787 ₽", 150),
    _el("Button", "Текущий счёт. 787 ₽", 250),
    _el("GenericElement", "Ставка 14,51 %", 300),
    _el("Slider", "Vertical scroll bar, 1 page", 116),
]


def test_scroll_bar_length_does_not_change_the_screen():
    longer = ACCOUNTS[:-1] + [_el("Slider", "Vertical scroll bar, 2 pages", 116)]

    assert compute_screen_id(ACCOUNTS) == compute_screen_id(longer)


def test_numbers_in_labels_do_not_change_the_screen():
    refreshed = [
        _el("Button", "Общий баланс"),
        _el("GenericElement", "Общий баланс, 1 250,50 ₽", 150),
        _el("Button", "Текущий счёт. 1 250,50 ₽", 250),
        _el("GenericElement", "Ставка 15,02 %", 300),
        _el("Slider", "Vertical scroll bar, 1 page", 116),
    ]

    assert compute_screen_id(ACCOUNTS) == compute_screen_id(refreshed)


def test_a_different_screen_still_gets_a_different_id():
    other = ACCOUNTS + [_el("Button", "Открыть вклад", 400)]

    assert compute_screen_id(ACCOUNTS) != compute_screen_id(other)


def _node(screen_id: str, labels: list[str]) -> ScreenNode:
    return ScreenNode(screen_id=screen_id, interactive_elements=[
        ElementSnapshot(kind=ElementKind.BUTTON, element_type="Button", label=label,
                        frame={"x": 0, "y": 100 + 40 * i, "width": 402, "height": 40})
        for i, label in enumerate(labels)
    ])


def _engine(tmp_path) -> ExplorationEngine:
    return ExplorationEngine(
        controller=_PopupController(), app_bundle_id="com.aweassist.app",
        output_dir=str(tmp_path), mode=ExplorationMode.MC,
    )


ACCOUNT_ACTIONS = [
    "Общий баланс", "Текущий счёт", "Накопительный", "Валютный", "Кредитка",
    "Все мои продукты", "Поиск", "Подарок за друга", "Настройки", "Новый продукт",
]


def test_variant_with_a_reload_button_is_the_same_screen(tmp_path):
    engine = _engine(tmp_path)
    base = ACCOUNT_ACTIONS
    engine.graph.add_node(_node("accounts", base))
    engine.graph.add_node(_node("accounts-error", base + ["glyph repeat m"]))
    engine.graph.add_node(_node("profile", ["Назад", "Фото профиля", "Обо мне"]))

    assert engine._same_screen("accounts-error", "accounts")
    assert not engine._same_screen("profile", "accounts")


def test_written_off_screen_stops_being_a_frontier(tmp_path):
    engine = _engine(tmp_path)
    engine.graph.add_node(_node("accounts", ACCOUNT_ACTIONS))
    engine._update_archive_for("accounts")
    assert engine.archive.best_frontier().state_id == "accounts"

    engine._note_unreachable("accounts", step=1)
    engine._note_unreachable("accounts", step=2)

    assert engine.archive.best_frontier() is None


def _client(tab_bar_visible: bool) -> AXeExplorerClient:
    client = AXeExplorerClient()
    client._width, client._height = 402, 874
    client._tab_bar_visible = tab_bar_visible
    return client


def test_bottom_items_need_scrolling_only_when_a_tab_bar_is_visible():
    take_photo = {"x": 16, "y": 770, "width": 370, "height": 56}  # bottom sheet item

    assert _client(tab_bar_visible=True).needs_scroll("Button", take_photo)
    assert not _client(tab_bar_visible=False).needs_scroll("Button", take_photo)


@pytest.mark.parametrize("label", [
    "Выбрать из галереи", "Сделать фото", "Камера", "Поделиться", "Выбрать файл",
    "Choose from Library", "Take Photo", "Camera", "Share",
])
def test_system_ui_triggers_are_recognised(label):
    from explorer.engine import opens_system_ui

    assert opens_system_ui(label)


@pytest.mark.parametrize("label", [
    "Установить фото профиля", "Фото профиля", "Счета", "Shared accounts", "Открыть настройки iOS", None,
])
def test_ordinary_actions_are_not_system_ui(label):
    from explorer.engine import opens_system_ui

    assert not opens_system_ui(label)


@pytest.mark.parametrize("guard", [True, False])
async def test_system_ui_guard_drops_picker_actions(tmp_path, guard):
    # aweassist run 3: «Выбрать из галереи» opened the Photos picker, which
    # is not in the app's AX tree; the agent then tapped blindly inside it.
    from tests.test_engine import _app, _button

    controller = _PopupController(tree=_app(
        "Деньги", _button("Выбрать из галереи", 790), _button("Сделать фото", 730), _button("Закрыть", 60),
    ))
    engine = ExplorationEngine(
        controller=controller, app_bundle_id="com.aweassist.app",
        output_dir=str(tmp_path), mode=ExplorationMode.MC, system_ui_guard=guard,
    )

    node = await engine._capture_screen()

    labels = {el.label for el in node.interactive_elements}
    if guard:
        assert labels == {"Закрыть"}
    else:
        assert labels == {"Выбрать из галереи", "Сделать фото", "Закрыть"}


def _tab(label: str, x: int) -> dict:
    return {"type": "RadioButton", "label": label, "value": None, "test_id": label,
            "enabled": True, "frame": {"x": x, "y": 800, "width": 80, "height": 50}, "children": []}


async def test_tab_bar_items_are_tried_before_screen_content(tmp_path):
    # aweassist run 2: tabs came last («Чаты» at step 166) because ties are
    # broken in tree order and the tab bar is last in the tree.
    from tests.test_engine import _app, _button

    cards = [_button(f"Карточка {name}", 100 + 40 * i) for i, name in enumerate("абвгдежзик")]
    tabs = [_tab(name, 80 * i) for i, name in enumerate(["Главный", "Платежи", "Выгода", "История", "Чаты"])]
    engine = _engine(tmp_path)
    engine.controller.tree = _app("Деньги", *cards, *tabs)
    node = await engine._capture_screen()
    engine.graph.add_node(node)
    await engine._on_screen_landed(node, is_new=True, step=0)

    first = engine._select_action(node.screen_id, node.interactive_elements)

    assert engine._action_id_to_element[first].element_type == "RadioButton"


async def test_visited_tab_drops_below_content_on_every_screen(tmp_path, monkeypatch):
    # aweassist run 3: with the boost kept, every screen inside a section
    # re-tried all five tabs first (steps 63–69 were nothing but tab hops).
    from tests.test_engine import _app, _button, _no_sleep

    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    tabs = [_tab(name, 80 * i) for i, name in enumerate(["Главный", "Платежи"])]
    engine = _engine(tmp_path)

    async def land(*buttons):
        engine.controller.tree = _app("Деньги", *buttons, *tabs)
        node = await engine._capture_screen()
        engine.graph.add_node(node)
        await engine._on_screen_landed(node, is_new=True, step=0)
        return node

    def first_label(node, skip=()):
        pool = [el for el in node.interactive_elements if el.label not in skip]
        return engine._action_id_to_element[engine._select_action(node.screen_id, pool)].label

    main = await land(_button("Карточка", 100))
    engine.current_screen_id = main.screen_id
    payments_tab = [el for el in main.interactive_elements if el.label == "Платежи"][0]
    await engine._explore_tap(payments_tab)

    # Already-registered screen: «Платежи» now ranks below the card.
    assert first_label(main, skip={"Главный"}) == "Карточка"
    # A screen registered afterwards: the unvisited tab still comes first,
    # the visited one comes after ordinary content.
    other = await land(_button("Перевод", 100))
    assert first_label(other) == "Главный"
    assert first_label(other, skip={"Главный"}) == "Перевод"


async def test_screen_without_tab_bar_keeps_uniform_priors(tmp_path):
    from explorer.engine import structural_priors

    assert structural_priors(_node("profile", ["Назад", "Фото профиля", "Обо мне"])) is None


def _tree(*windows):
    return json.dumps([{"type": "Application", "AXLabel": "Деньги", "children": list(windows)}])


def _window(*children):
    return {"type": "Group", "children": list(children)}


def _ax(type_, label, y):
    return {"type": type_, "AXLabel": label, "AXUniqueId": label,
            "frame": {"x": 0, "y": y, "width": 80, "height": 50}, "children": []}


@pytest.mark.parametrize("covered", [False, True], ids=["tab-bar-visible", "covered-by-sheet"])
async def test_tab_bar_visibility_is_read_from_the_tree(monkeypatch, covered):
    screen = _window(_ax("RadioButton", "Главный", 800), _ax("Button", "Профиль", 300))
    windows = [_window(screen, _window(_ax("Button", "Сделать фото", 770)))] if covered else [_window(screen)]

    async def fake_run(args, timeout=15):
        return 0, _tree(*windows), ""

    monkeypatch.setattr(axe_client, "_run", fake_run)
    client = _client(tab_bar_visible=covered)  # the opposite: the read must set it

    await client.get_ui_elements()

    assert client._tab_bar_visible is (not covered)
