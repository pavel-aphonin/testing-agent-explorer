"""Keyboard handling for iOS 27 AX trees (testbed baseline-1 findings).

The fixture is a real ``axe describe-ui`` dump of the testbed login
screen with the system keyboard up (iOS 27.2, AXe 1.8.0). iOS 27 has no
``Keyboard`` element type: the keyboard is plain Group/Button nodes in
separate top-level windows of the Application.
"""

from __future__ import annotations

import json
from pathlib import Path

from explorer.analyzer import analyze_screen
from explorer.axe_client import AXeExplorerClient
from explorer.screen_id import compute_screen_id

FIXTURE = Path(__file__).parent / "fixtures" / "ios27_login_with_keyboard.json"


def _flatten(raw: list) -> list[dict]:
    elements: list[dict] = []
    AXeExplorerClient()._flatten(raw, elements)
    return elements


def _without_keyboard_windows(raw: list) -> list:
    """The same tree with the keyboard windows removed (keyboard down)."""
    app = dict(raw[0])
    app["children"] = [
        child for child in app["children"]
        if "SystemInputAssistantView" not in json.dumps(child)
        and "inputView" not in json.dumps(child)
    ]
    return [app]


def test_keyboard_keys_are_not_interactive_elements():
    node = analyze_screen(_flatten(json.loads(FIXTURE.read_text())))

    test_ids = {el.test_id for el in node.interactive_elements}
    assert test_ids == {"login.email", "login.password", "login.submit"}


def test_keyboard_does_not_change_screen_id():
    raw = json.loads(FIXTURE.read_text())

    with_keyboard = compute_screen_id(_flatten(raw))
    without_keyboard = compute_screen_id(_flatten(_without_keyboard_windows(raw)))

    assert with_keyboard == without_keyboard


def test_keyboard_elements_are_tagged_for_wait_for_keyboard():
    # wait_for_keyboard() looks for «Keyboard» in the element type.
    elements = _flatten(json.loads(FIXTURE.read_text()))

    keys = [el for el in elements if "Keyboard" in el["type"]]
    assert any(el["label"] == "q" for el in keys)
    assert not any((el["test_id"] or "").startswith("login.") for el in keys)
