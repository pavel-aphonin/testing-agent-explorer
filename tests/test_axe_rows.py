"""List rows and element values in iOS 27 AX trees (testbed run 2).

The fixture is a real ``axe describe-ui`` dump of the testbed «Счета»
screen: a UITableView whose cells AXe reports as plain ``Group`` children
of a ``ScrollArea``, each with two StaticTexts and a «chevron» button.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from explorer.analyzer import analyze_screen
from explorer.axe_client import AXeExplorerClient
from explorer.screen_id import compute_screen_id

FIXTURE = Path(__file__).parent / "fixtures" / "ios27_accounts_uikit_table.json"


def _flatten(raw: list) -> list[dict]:
    elements: list[dict] = []
    AXeExplorerClient()._flatten(raw, elements)
    return elements


def _walk(node: dict):
    yield node
    for child in node.get("children", []):
        yield from _walk(child)


def test_table_rows_are_tappable_cells():
    node = analyze_screen(_flatten(json.loads(FIXTURE.read_text())))

    cells = {
        el.test_id: el.label
        for el in node.interactive_elements
        if el.element_type == "Cell"
    }
    assert cells == {
        "accounts.row.0": "Текущий счёт",
        "accounts.row.1": "Накопительный",
        "accounts.row.2": "Валютный",
    }


def test_row_disclosure_chevron_is_not_an_action():
    node = analyze_screen(_flatten(json.loads(FIXTURE.read_text())))

    assert all(el.label != "chevron" for el in node.interactive_elements)


def test_row_text_does_not_change_screen_id():
    raw = json.loads(FIXTURE.read_text())
    changed = copy.deepcopy(raw)
    for n in _walk(changed[0]):
        if n.get("type") == "StaticText" and n.get("AXLabel"):
            n["AXLabel"] = n["AXLabel"] + " (обновлено)"

    assert compute_screen_id(_flatten(raw)) == compute_screen_id(_flatten(changed))


def test_rows_with_their_own_controls_stay_as_controls():
    # A NavigationLink / toggle row: the control inside is the action.
    raw = [{
        "type": "ScrollArea", "children": [{
            "type": "Group", "AXUniqueId": "row", "children": [
                {"type": "StaticText", "AXLabel": "Уведомления", "children": []},
                {"type": "Switch", "AXLabel": "Уведомления", "children": []},
            ],
        }],
    }]

    types = [el["type"] for el in _flatten(raw)]
    assert "Cell" not in types
    assert "Switch" in types


def test_non_string_ax_value_becomes_string():
    # iOS Settings → Fonts slider reported AXValue 0.5555…, which crashed
    # ElementSnapshot validation in testbed run 2.
    raw = [{"type": "Slider", "AXLabel": "Размер", "AXValue": 0.5555555820465088, "children": []}]

    (element,) = _flatten(raw)
    assert element["value"] == "0.5555555820465088"
    analyze_screen([element])  # must not raise
