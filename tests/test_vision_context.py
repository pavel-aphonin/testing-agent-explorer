"""PER-175 Phase F: tests for the Holo2 VLM Context Identifier parser.

The parser is the pure, critical half — it turns Holo2's JSON reply into the
AffordanceMap the Platform Adapter + keypad gate consume. These lock the
canvas-PIN case (the whole point of Holo2): a keypad reply must yield
has_keypad=True with digit keys, so should_fire_keypad_macro fires where
SigLIP+OmniParser couldn't (smoke dddd4444).
"""

from __future__ import annotations

import json

from explorer.affordances import AffordanceKind
from explorer.agents.vision_context import parse_vlm_affordances
from explorer.platform_adapter import should_fire_keypad_macro


def _pin_reply() -> str:
    return json.dumps({
        "screen_type": "pin_entry",
        "confidence": 0.94,
        "elements": (
            [{"kind": "keypad_key", "label": str(d), "value": str(d),
              "bbox": [30 * (d % 3), 100 + 40 * (d // 3), 30 * (d % 3) + 28, 138 + 40 * (d // 3)]}
             for d in range(10)]
            + [{"kind": "submit", "label": "Вперёд", "value": None, "bbox": [100, 700, 300, 750]}]
        ),
    })


def test_parses_pin_keypad_reply() -> None:
    amap = parse_vlm_affordances(_pin_reply())
    assert amap is not None
    assert amap.screen_type == "pin_entry"
    assert amap.screen_confidence == 0.94
    assert amap.has_keypad is True
    assert len(amap.keypad_keys) == 10
    assert [a.value for a in amap.digit_keys_in_order()] == [str(d) for d in range(10)]
    assert amap.submit_buttons[0].label == "Вперёд"


def test_pin_reply_triggers_keypad_gate() -> None:
    """End-to-end of the fix: a Holo2 PIN reply makes the safety gate fire,
    where the SigLIP path (uniform 0.09, no boxes) never did."""
    amap = parse_vlm_affordances(_pin_reply())
    assert should_fire_keypad_macro(amap) is True


def test_parses_login_reply_with_editable_field() -> None:
    reply = json.dumps({
        "screen_type": "login",
        "confidence": 0.88,
        "elements": [
            {"kind": "text_field", "label": "Телефон", "value": None, "bbox": [20, 200, 400, 250]},
            {"kind": "submit", "label": "Войти", "value": None, "bbox": [100, 600, 300, 650]},
        ],
    })
    amap = parse_vlm_affordances(reply)
    assert amap.has_keypad is False
    assert len(amap.editable_fields) == 1
    assert amap.editable_fields[0].kind is AffordanceKind.TEXT_FIELD
    assert amap.editable_fields[0].editable is True
    assert should_fire_keypad_macro(amap) is False


def test_tolerates_think_preamble_and_fences() -> None:
    reply = (
        "<think>This looks like a PIN pad with ten keys.</think>\n"
        "```json\n" + _pin_reply() + "\n```"
    )
    amap = parse_vlm_affordances(reply)
    assert amap is not None
    assert amap.has_keypad is True


def test_tolerates_trailing_chatter() -> None:
    reply = _pin_reply() + "\n\nHope this helps!"
    amap = parse_vlm_affordances(reply)
    assert amap is not None
    assert amap.screen_type == "pin_entry"


def test_unknown_kind_degrades_to_other() -> None:
    reply = json.dumps({
        "screen_type": "home", "confidence": 0.7,
        "elements": [{"kind": "wibble", "label": "x", "bbox": [0, 0, 10, 10]}],
    })
    amap = parse_vlm_affordances(reply)
    assert amap.affordances[0].kind is AffordanceKind.OTHER


def test_garbage_returns_none() -> None:
    assert parse_vlm_affordances("") is None
    assert parse_vlm_affordances("not json at all") is None
    assert parse_vlm_affordances("<think>only reasoning, no json</think>") is None


def test_missing_bbox_is_tolerated() -> None:
    reply = json.dumps({
        "screen_type": "pin_entry", "confidence": 0.9,
        "elements": [{"kind": "keypad_key", "label": "8", "value": "8"}],  # no bbox
    })
    amap = parse_vlm_affordances(reply)
    assert amap is not None
    assert amap.keypad_keys[0].value == "8"
    assert amap.keypad_keys[0].bbox is None
