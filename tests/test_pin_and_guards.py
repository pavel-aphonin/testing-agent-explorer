"""PER-204 / PER-205 unit tests — PIN-submit macro + editable guard.

These exercise the pure, transport-agnostic helpers that both the sync
(_goal_decide) and bus (planner-runner) paths share, plus the goal_schema
editability predicate. No device, no LLM, no bus — just the logic that
the run bbbb2222 post-mortem traced the login failure to:

  * PER-204: the planner emits the 4 digit taps but DROPS the final
    «Вперёд» submit, so the PIN screen never advances. ``append_pin_submit``
    adds it deterministically.
  * PER-205: the planner aims ``enter_text`` at a heading (not a field);
    typing there is a silent no-op. ``_is_editable_kind`` + the
    elements-block marker steer it to a real field, and the dispatch
    guard skips a mistargeted entry with feedback.
"""

from __future__ import annotations

from explorer.goal_schema import _is_editable_kind, build_elements_block
from explorer.planning.hints import append_pin_submit


# ----------------------------------------------------------------- PER-204


def _digit(d: str) -> dict:
    """A keypad digit tap as the planner emits it (tap_at by description)."""
    return {
        "action": "tap_at",
        "action_args": {"target_description": f"цифра {d} на PIN-клавиатуре"},
        "element_id": None,
        "value_source": "test_data.pin_code",
        "reasoning": f"tap {d}",
    }


def test_append_pin_submit_adds_submit_when_missing() -> None:
    """4 digit taps, no submit, PIN screen → a trailing submit tap_at is
    appended (the run bbbb2222 fix)."""
    batch = [_digit("8"), _digit("5"), _digit("2"), _digit("0")]
    out = append_pin_submit(batch, context_is_pin=True)
    assert len(out) == 5
    submit = out[-1]
    assert submit["action"] == "tap_at"
    # The submit must carry a target_description so the grounder can
    # localise the «Вперёд» button.
    assert "target_description" in submit["action_args"]
    assert submit["action_args"]["target_description"]
    # Input is never mutated in place.
    assert len(batch) == 4


def test_append_pin_submit_noop_when_submit_already_present() -> None:
    """If the model already included a «Войти» tap, don't double it."""
    batch = [
        _digit("8"), _digit("5"), _digit("2"), _digit("0"),
        {
            "action": "tap_at",
            "action_args": {"target_description": "кнопка Войти"},
            "reasoning": "submit",
        },
    ]
    out = append_pin_submit(batch, context_is_pin=True)
    assert len(out) == 5  # unchanged
    assert out is batch  # no-op returns the same list


def test_append_pin_submit_noop_when_not_pin_screen() -> None:
    """Off a PIN screen the macro never fires, even with 4 digit taps."""
    batch = [_digit("8"), _digit("5"), _digit("2"), _digit("0")]
    out = append_pin_submit(batch, context_is_pin=False)
    assert out == batch


def test_append_pin_submit_noop_below_threshold() -> None:
    """Fewer than 4 digits → the code isn't complete yet, no submit."""
    batch = [_digit("8"), _digit("5")]
    out = append_pin_submit(batch, context_is_pin=True)
    assert len(out) == 2


def test_append_pin_submit_handles_empty_and_none() -> None:
    """Defensive: empty / None batches return a list, never raise."""
    assert append_pin_submit([], context_is_pin=True) == []
    assert append_pin_submit(None, context_is_pin=True) == []


def test_append_pin_submit_recognizes_top_level_target_description() -> None:
    """Digit taps whose description lives at the top level (not in
    action_args) must still count — the bus grounder mirrors it there."""
    batch = [
        {"action": "tap_at", "target_description": f"цифра {d}"}
        for d in "8520"
    ]
    out = append_pin_submit(batch, context_is_pin=True)
    assert len(out) == 5
    assert out[-1]["action"] == "tap_at"


# ----------------------------------------------------------------- PER-205


def test_is_editable_kind_true_for_text_inputs() -> None:
    """iOS text-entry kinds (any spacing / casing) are editable."""
    for kind in (
        "TextField", "textfield", "Text Field", "text_field",
        "SecureTextField", "Secure Text Field", "TextView", "SearchField",
    ):
        assert _is_editable_kind(kind) is True, kind


def test_is_editable_kind_false_for_non_inputs() -> None:
    """Headings / labels / buttons / images / containers are NOT editable
    — the bug was the planner typing into Heading_vhod_68."""
    for kind in (
        "Heading", "StaticText", "Button", "Image", "Other",
        "AXHeading", "label", "", None,
    ):
        assert _is_editable_kind(kind) is False, kind


def test_elements_block_marks_editable_fields() -> None:
    """``build_elements_block`` flags a text field so the model knows
    where input/enter_text is allowed, and does NOT flag a heading."""
    elements = [
        {"id": "phone", "label": "Телефон", "kind": "TextField"},
        {"id": "Heading_vhod_68", "label": "Вход", "kind": "Heading"},
    ]
    block, ids = build_elements_block(elements)
    assert ids == ["phone", "Heading_vhod_68"]
    lines = block.splitlines()
    phone_line = next(l for l in lines if "id=phone" in l)
    heading_line = next(l for l in lines if "id=Heading_vhod_68" in l)
    assert "поле ввода" in phone_line
    assert "поле ввода" not in heading_line
