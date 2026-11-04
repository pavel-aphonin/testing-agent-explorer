"""Modals and switches in iOS 27 AX trees (testbed run 4).

Fixtures are real ``axe describe-ui`` dumps from the testbed:
- the «Удалить аккаунт?» alert over Settings (alert = ``Sheet`` element);
- the profile-edit SwiftUI sheet over Profile.
In both, the modal is a later child container of the app window and the
screen underneath is still listed in the tree.
"""

from __future__ import annotations

import json
from pathlib import Path

from explorer.analyzer import analyze_screen
from explorer.axe_client import AXeExplorerClient
from explorer.models import ElementKind

FIXTURES = Path(__file__).parent / "fixtures"


def _screen(name: str):
    elements: list[dict] = []
    AXeExplorerClient()._flatten(json.loads((FIXTURES / name).read_text()), elements)
    return analyze_screen(elements)


def test_alert_buttons_are_the_only_actions():
    node = _screen("ios27_settings_delete_alert.json")

    assert {el.label for el in node.interactive_elements} == {"Отмена", "Удалить"}


def test_sheet_controls_are_the_only_actions():
    node = _screen("ios27_profile_edit_sheet.json")

    assert {el.test_id for el in node.interactive_elements} == {
        "profile_edit.cancel", "profile_edit.save",
        "profile_edit.name", "profile_edit.email",
    }


def test_screen_under_modal_is_still_part_of_the_screen():
    # The fingerprint and the evaluator's markers still see what's below.
    node = _screen("ios27_profile_edit_sheet.json")

    assert any(el.test_id == "screen.profile" for el in node.elements)


def test_screen_without_modal_keeps_all_actions():
    node = _screen("ios27_accounts_uikit_table.json")

    assert "BackButton" in {el.test_id for el in node.interactive_elements}


def test_checkbox_toggle_is_a_switch_action():
    # Settings under the alert: the toggle is obscured there, so build the
    # same screen without the alert window's modal container.
    raw = json.loads((FIXTURES / "ios27_settings_delete_alert.json").read_text())
    window = raw[0]["children"][0]
    window["children"] = window["children"][:1]
    elements: list[dict] = []
    AXeExplorerClient()._flatten(raw, elements)

    node = analyze_screen(elements)

    toggle = [el for el in node.interactive_elements if el.test_id == "settings.notifications"]
    assert toggle and toggle[0].kind == ElementKind.SWITCH
    # The unlabeled CheckBox nested inside is the same control.
    assert [el.element_type for el in node.interactive_elements].count("CheckBox") == 1
