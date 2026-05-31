"""PER-175 full integration: tests for the shared affordance vocabulary.

Guards the perception→Platform-Adapter contract: a canvas keypad must be
recognised as tappable keys (NOT an editable field), a real text field as
editable, and the map must round-trip through the bus JSON form intact.
"""

from __future__ import annotations

from explorer.affordances import (
    Affordance,
    AffordanceKind,
    AffordanceMap,
    EDITABLE_KINDS,
)


def _pin_keypad_map() -> AffordanceMap:
    keys = [
        Affordance(
            kind=AffordanceKind.KEYPAD_KEY,
            label=str(d),
            value=str(d),
            bbox=[10 * d, 100, 10 * d + 40, 140],
            confidence=0.9,
        )
        for d in range(10)
    ]
    submit = Affordance(
        kind=AffordanceKind.SUBMIT, label="Вперёд", bbox=[200, 800, 320, 850]
    )
    return AffordanceMap(
        screen_type="pin_entry",
        screen_confidence=0.81,
        affordances=keys + [submit],
    )


def test_keypad_screen_has_keypad_and_no_editable_fields() -> None:
    """The canvas PIN bug: keys are tappable, NOT typed into. The map must
    report a keypad and zero editable fields so the Platform Adapter routes
    to tap_at, never enter_text."""
    m = _pin_keypad_map()
    assert m.has_keypad is True
    assert m.editable_fields == []
    assert len(m.keypad_keys) == 10
    assert len(m.submit_buttons) == 1


def test_digit_keys_in_order() -> None:
    """Resolver taps digits by value order, regardless of detection order."""
    m = _pin_keypad_map()
    ordered = m.digit_keys_in_order()
    assert [a.value for a in ordered] == [str(d) for d in range(10)]


def test_text_field_is_editable() -> None:
    """A real text field must come back as editable (enter_text allowed)."""
    m = AffordanceMap(
        screen_type="login",
        affordances=[
            Affordance(kind=AffordanceKind.TEXT_FIELD, label="Телефон", editable=True),
            Affordance(kind=AffordanceKind.BUTTON, label="Войти"),
        ],
    )
    assert m.has_keypad is False
    assert len(m.editable_fields) == 1
    assert m.editable_fields[0].label == "Телефон"


def test_secure_field_counts_as_editable() -> None:
    assert AffordanceKind.SECURE_FIELD in EDITABLE_KINDS
    m = AffordanceMap(
        affordances=[Affordance(kind=AffordanceKind.SECURE_FIELD, editable=True)]
    )
    assert len(m.editable_fields) == 1


def test_roundtrip_through_bus_json_form() -> None:
    """The map crosses the Redis Streams bus as a dict — it must survive
    to_dict → from_dict byte-for-byte in content."""
    m = _pin_keypad_map()
    again = AffordanceMap.from_dict(m.to_dict())
    assert again.screen_type == "pin_entry"
    assert again.has_keypad is True
    assert len(again.keypad_keys) == 10
    assert [a.value for a in again.digit_keys_in_order()] == [
        str(d) for d in range(10)
    ]
    assert again.submit_buttons[0].label == "Вперёд"


def test_affordance_center() -> None:
    a = Affordance(kind=AffordanceKind.SUBMIT, bbox=[100, 200, 300, 240])
    assert a.center() == (200, 220)
    assert Affordance(kind=AffordanceKind.OTHER).center() is None


def test_is_pin_entry_bridge() -> None:
    """Back-compat bridge for the retired boolean."""
    assert AffordanceMap(screen_type="pin_entry").is_pin_entry is True
    assert AffordanceMap(screen_type="PIN code entry screen").is_pin_entry is True
    assert AffordanceMap(screen_type="login").is_pin_entry is False


def test_unknown_kind_degrades_to_other() -> None:
    a = Affordance.from_dict({"kind": "nonsense_kind", "label": "x"})
    assert a.kind is AffordanceKind.OTHER
