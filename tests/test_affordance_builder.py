"""PER-175 Phase B: tests for the vision→AffordanceMap builder.

These lock the perception classification rules that the canvas-PIN bug
turned out to hinge on:

  * a box containing a single digit on a keypad grid is a tappable
    KEYPAD_KEY (never a text field) — and the screen reads has_keypad;
  * a handful of stray numbers is NOT a keypad (no false "enter a code");
  * a box overlapping an AX text-field rect is editable (hybrid);
  * submit/back vocabulary is typed correctly so the adapter can find the
    confirm button.
"""

from __future__ import annotations

from explorer.affordance_builder import build_affordance_map
from explorer.affordances import AffordanceKind


def _digit_boxes(n: int = 10) -> list[dict]:
    # a 0-9 grid, each box 40px wide
    return [
        {"bbox": [50 * (d % 5), 100 + 50 * (d // 5), 50 * (d % 5) + 40, 140 + 50 * (d // 5)],
         "text": str(d), "confidence": 0.9}
        for d in range(n)
    ]


def test_canvas_keypad_detected_with_submit() -> None:
    boxes = _digit_boxes(10) + [
        {"bbox": [200, 800, 320, 850], "text": "Вперёд", "confidence": 0.8}
    ]
    m = build_affordance_map(screen_type="pin_entry", screen_confidence=0.8, boxes=boxes)
    assert m.has_keypad is True
    assert len(m.keypad_keys) == 10
    assert [a.value for a in m.digit_keys_in_order()] == [str(d) for d in range(10)]
    assert len(m.submit_buttons) == 1
    assert m.submit_buttons[0].label == "Вперёд"
    # A keypad screen has NO editable field → adapter routes to taps.
    assert m.editable_fields == []


def test_stray_digits_are_not_a_keypad() -> None:
    """Two numeric labels (e.g. a balance, a count) must NOT read as a
    keypad — otherwise the adapter would try to 'enter a code' on a home
    screen."""
    boxes = [
        {"bbox": [10, 10, 50, 30], "text": "5", "confidence": 0.9},
        {"bbox": [10, 40, 50, 60], "text": "8", "confidence": 0.9},
        {"bbox": [100, 10, 300, 40], "text": "Баланс", "confidence": 0.9},
    ]
    m = build_affordance_map(screen_type="home", boxes=boxes)
    assert m.has_keypad is False
    assert m.keypad_keys == []


def test_editable_field_from_ax_region() -> None:
    """A vision box overlapping an AX text-field rect is marked editable
    even though the box text is empty (hybrid perception)."""
    boxes = [{"bbox": [20, 200, 400, 250], "text": "", "confidence": 0.7}]
    regions = [(0, 190, 440, 260)]  # AX field rect overlapping the box
    m = build_affordance_map(screen_type="login", boxes=boxes, editable_regions=regions)
    assert len(m.editable_fields) == 1
    assert m.editable_fields[0].editable is True
    assert m.editable_fields[0].source == "hybrid"


def test_editable_field_from_kind_hint() -> None:
    boxes = [{"bbox": [20, 200, 400, 250], "text": "Телефон",
              "kind_hint": "TextField", "confidence": 0.7}]
    m = build_affordance_map(screen_type="login", boxes=boxes)
    fields = m.editable_fields
    assert len(fields) == 1
    assert fields[0].kind is AffordanceKind.TEXT_FIELD


def test_secure_field_detected() -> None:
    boxes = [{"bbox": [20, 200, 400, 250], "text": "Пароль",
              "kind_hint": "SecureTextField", "confidence": 0.7}]
    m = build_affordance_map(boxes=boxes)
    assert len(m.editable_fields) == 1
    assert m.editable_fields[0].kind is AffordanceKind.SECURE_FIELD


def test_submit_and_back_vocab() -> None:
    boxes = [
        {"bbox": [0, 0, 50, 40], "text": "Назад", "confidence": 0.9},
        {"bbox": [100, 800, 300, 850], "text": "Войти", "confidence": 0.9},
        {"bbox": [0, 400, 200, 440], "text": "Настройки", "confidence": 0.9},
    ]
    m = build_affordance_map(boxes=boxes)
    kinds = {a.label: a.kind for a in m.affordances}
    assert kinds["Назад"] is AffordanceKind.BACK
    assert kinds["Войти"] is AffordanceKind.SUBMIT
    assert kinds["Настройки"] is AffordanceKind.BUTTON  # generic labelled control


def test_empty_label_box_is_other() -> None:
    boxes = [{"bbox": [0, 0, 40, 40], "text": "", "confidence": 0.5}]
    m = build_affordance_map(boxes=boxes)
    assert m.affordances[0].kind is AffordanceKind.OTHER


def test_empty_screen() -> None:
    m = build_affordance_map(screen_type="unknown", boxes=[])
    assert m.affordances == []
    assert m.has_keypad is False
    assert m.is_pin_entry is False


def test_screen_type_drives_is_pin_entry() -> None:
    m = build_affordance_map(screen_type="pin_entry", boxes=_digit_boxes(10))
    assert m.is_pin_entry is True


def test_map_roundtrips_after_build() -> None:
    from explorer.affordances import AffordanceMap
    m = build_affordance_map(
        screen_type="pin_entry", screen_confidence=0.7,
        boxes=_digit_boxes(10) + [{"bbox": [200, 800, 320, 850], "text": "Войти"}],
    )
    again = AffordanceMap.from_dict(m.to_dict())
    assert again.has_keypad is True
    assert len(again.keypad_keys) == 10
    assert again.submit_buttons[0].label == "Войти"
