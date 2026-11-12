"""«--forbid ~text»: on a live service (Wikipedia) write actions come under
many labels, and none of them may ever be tapped."""

from __future__ import annotations

from explorer.engine import ExplorationEngine
from explorer.modes import ExplorationMode
from tests.test_engine import _app, _button, _PopupController


async def _actions(tmp_path, forbidden, *labels):
    controller = _PopupController(tree=_app("Wikipedia", *[_button(l, 100 + 50 * i) for i, l in enumerate(labels)]))
    engine = ExplorationEngine(controller=controller, app_bundle_id="org.wikimedia.wikipedia",
                               output_dir=str(tmp_path), mode=ExplorationMode.MC, forbidden=forbidden)
    node = await engine._capture_screen()
    return [el.label for el in node.interactive_elements]


async def test_part_of_a_label_forbids_it_in_any_case(tmp_path):
    assert await _actions(tmp_path, ["~edit", "~Publish"],
                          "Edit section: History", "EDIT", "Publish changes", "Search") == ["Search"]


async def test_exact_entries_still_match_whole_labels_only(tmp_path):
    assert await _actions(tmp_path, ["Выход"], "Выход", "Выходные") == ["Выходные"]


async def test_part_matches_the_accessibility_id_too(tmp_path):
    from tests.test_engine import _app as app
    controller = _PopupController(tree=app("Wikipedia", {**_button("Pencil", 100), "test_id": "article_edit_button"}))
    engine = ExplorationEngine(controller=controller, app_bundle_id="org.wikimedia.wikipedia",
                               output_dir=str(tmp_path), mode=ExplorationMode.MC, forbidden=["~edit"])

    node = await engine._capture_screen()

    assert node.interactive_elements == []


async def test_a_bare_tilde_forbids_nothing(tmp_path):
    assert await _actions(tmp_path, ["~"], "Search") == ["Search"]
