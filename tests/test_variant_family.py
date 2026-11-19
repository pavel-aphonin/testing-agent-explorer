"""Live screens (feeds, maps, search) are one screen however their content
changes; screens that really differ stay apart.

Wikipedia run 5 had «Community» as 21 nodes among 130, and 22 of the 25
screens written off as unreachable had a reachable same-name sibling.
"""
import gzip
import json
from pathlib import Path

from explorer.analyzer import is_same_family, is_variant
from explorer.engine import ExplorationEngine
from explorer.models import (
    ActionDetail, ActionType, ElementKind, ElementSnapshot, GraphEdge, ScreenNode,
)
from explorer.modes import ExplorationMode
from tests.test_engine import _PopupController

FIXTURES = Path(__file__).parent / "fixtures"
TABS = ["Лента", "Карта", "Поиск", "Сохранённое"]


def _el(label, type_="Button", y=100, value=None):
    return ElementSnapshot(
        kind=ElementKind.BUTTON, element_type=type_, label=label, test_id=label, value=value,
        frame={"x": 0, "y": y, "width": 402, "height": 40},
    )


def _screen(screen_id, cards, name="Лента", selected="Лента", tabs=TABS, back=False, modal=False,
            field=False):
    actions = [_el(c, y=100 + 40 * i) for i, c in enumerate(cards)]
    actions += [_el(t, "RadioButton", 800, "1" if t == selected else None) for t in tabs]
    if back:
        actions.append(_el("Back"))
    if field:
        actions.append(ElementSnapshot(kind=ElementKind.TEXT_FIELD, element_type="TextField",
                                       label="Запрос", frame={"x": 0, "y": 60, "width": 300, "height": 40}))
    return ScreenNode(screen_id=screen_id, name=name, elements=list(actions),
                      interactive_elements=actions, is_modal=modal, app_label="Приложение")


def test_feed_with_other_cards_is_the_same_screen():
    a = _screen("a", ["Статья про кошек", "Новость дня", "Игра"])
    b = _screen("b", ["Статья про лес", "Другая новость", "Фото дня", "Ещё"])

    assert is_same_family(a, b)
    assert is_variant(a, b)


def test_another_selected_tab_is_another_screen():
    a = _screen("a", ["x1", "x2"], name="Ещё", selected="Лента")
    b = _screen("b", ["x1", "x2"], name="Ещё", selected="Карта")

    assert not is_variant(a, b)


def test_sheet_over_a_feed_is_not_the_feed():
    assert not is_variant(_screen("a", ["карточка"]), _screen("b", ["карточка"], modal=True))


def test_pushed_page_is_not_the_root_with_the_same_name():
    assert not is_variant(_screen("a", ["карточка"]), _screen("b", ["карточка"], back=True))


def test_search_field_tells_screens_apart():
    assert not is_variant(_screen("a", ["карточка"]), _screen("b", ["карточка"], field=True))


def test_same_name_without_a_tab_bar_needs_the_actions_to_agree():
    # Testbed: «Веб» twice with two different buttons each.
    a = _screen("a", ["Назад", "Открыть"], tabs=[])
    b = _screen("b", ["Закрыть", "Обновить"], tabs=[])

    assert not is_variant(a, b)


def test_same_name_sheets_with_half_the_actions_are_one_screen():
    topics = [f"Тема {c}" for c in "абвгдежзик"]
    a = _screen("a", topics[:8], name="Интересы", tabs=[], modal=True)
    b = _screen("b", topics[3:], name="Интересы", tabs=[], modal=True)

    assert is_variant(a, b)


def test_name_alone_is_no_evidence():
    a = _screen("a", ["x"], tabs=[], name="Лента")
    b = _screen("b", ["y"], tabs=[], name="Лента")
    a.app_label = b.app_label = "Лента"

    assert not is_variant(a, b)


def _engine(tmp_path) -> ExplorationEngine:
    return ExplorationEngine(
        controller=_PopupController(), app_bundle_id="com.example.app",
        output_dir=str(tmp_path), mode=ExplorationMode.MC,
    )


def test_engine_merges_a_feed_into_the_known_node(tmp_path):
    engine = _engine(tmp_path)
    known = _screen("a", ["Статья про кошек", "Игра"])
    engine.graph.add_node(known)

    merged = engine._merge_into_known_variant(_screen("b", ["Статья про лес", "Фото дня"]))

    assert merged.screen_id == "a"
    assert "Фото дня" in {el.label for el in known.interactive_elements}


def _tap(source, target, label):
    return GraphEdge(source_screen_id=source, target_screen_id=target,
                     action=ActionDetail(action_type=ActionType.TAP, target_label=label))


def test_unreachable_screen_hands_its_actions_to_a_reachable_sibling(tmp_path):
    engine = _engine(tmp_path)
    home = _screen("home", ["Домой"], name="Домой", tabs=[], selected="")
    feed_ok = _screen("feed-ok", ["Статья про кошек", "Игра"])
    feed_lost = _screen("feed-lost", ["Статья про лес", "Фото дня"])
    for n in (home, feed_ok, feed_lost):
        engine.graph.add_node(n)
    engine.graph.launch_screens.append("home")
    engine.graph.add_edge(_tap("home", "feed-ok", "Лента"))

    engine._note_unreachable("feed-lost", step=1)
    engine._note_unreachable("feed-lost", step=2)

    labels = {el.label for el in feed_ok.interactive_elements}
    assert {"Статья про лес", "Фото дня"} <= labels
    assert "Фото дня" in {el.label for el in engine.graph.get_unexplored_actions("feed-ok")}
    assert not engine.graph.get_unexplored_actions("feed-lost")


def test_no_handover_to_a_sibling_nobody_can_reach(tmp_path):
    engine = _engine(tmp_path)
    feed_island = _screen("island", ["Статья про кошек"])
    feed_lost = _screen("feed-lost", ["Статья про лес", "Фото дня"])
    engine.graph.add_node(feed_island)
    engine.graph.add_node(feed_lost)

    engine._note_unreachable("feed-lost", step=1)
    engine._note_unreachable("feed-lost", step=2)

    assert "Фото дня" not in {el.label for el in feed_island.interactive_elements}


# --- stored runs -----------------------------------------------------------

def _replay(tmp_path, fixture):
    """Feed the nodes of a stored run through the engine's matching in
    capture order; returns (nodes in the run, nodes kept, merged pairs)."""
    raw = json.loads(gzip.open(FIXTURES / fixture, "rt", encoding="utf-8").read())
    engine = _engine(tmp_path)
    merged = []
    for d in raw:
        node = ScreenNode(**d)
        kept = engine._merge_into_known_variant(node)
        if kept is not node:
            merged.append((node.screen_id, kept.screen_id))
        engine.graph.add_node(kept)
    return len(raw), len(engine.graph.nodes), merged


def test_wikipedia_run_5_screens_collapse(tmp_path):
    total, kept, _ = _replay(tmp_path, "variant_wikipedia_5.json.gz")

    assert total == 130
    assert kept <= 90  # 130 before: not one of them matched


def test_testbed_screens_are_never_merged(tmp_path):
    total, kept, merged = _replay(tmp_path, "variant_testbed_34.json.gz")

    assert total == 15
    assert merged == [] and kept == 15


def test_a_small_sheet_with_the_same_actions_is_one_screen():
    # Wikipedia run 11: «Activity», 4 actions, two ids; a replay refused to
    # tap on the second one 7 times.
    actions = ["Customize", "Done", "Edit", "Learn more"]
    a = _screen("a", actions, name="Activity", tabs=[], modal=True)
    b = _screen("b", actions, name="Activity", tabs=[], modal=True)

    assert is_variant(a, b)


def test_one_or_two_equal_actions_are_not_enough():
    a = _screen("a", ["Close"], name="Activity", tabs=[], modal=True)
    b = _screen("b", ["Close"], name="Activity", tabs=[], modal=True)

    assert not is_variant(a, b)
