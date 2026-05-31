"""PER-175 Phase C: tests for the deterministic keypad macro.

This is the run-cccc3333 fix: on a vision-classified PIN screen, drive the
canvas keypad deterministically (tap the secret's digits in order, then
submit) rather than letting the planner pick enter_text on a canvas.
"""

from __future__ import annotations

import re

from explorer.platform_adapter import keypad_macro


def _digit(desc: str) -> str:
    m = re.search(r"цифра (\d)", desc or "")
    return m.group(1) if m else "?"


def test_macro_fires_on_pin_screen_with_code() -> None:
    batch = keypad_macro("pin_entry", {"pin_code": "8520"})
    assert batch is not None
    # 4 digit taps (8,5,2,0) in secret order + submit
    assert len(batch) == 5
    assert all(a["action"] == "tap_at" for a in batch)
    assert [_digit(a["target_description"]) for a in batch[:-1]] == ["8", "5", "2", "0"]
    # submit is last, no coords (grounded by description)
    assert "x" not in batch[-1]["action_args"]
    assert any(w in batch[-1]["target_description"] for w in ("Вперёд", "Продолжить", "Войти"))


def test_macro_taps_by_description_only() -> None:
    """Digit taps carry a description and NO coords — the proven Grounder
    path (cccc3333: UI-TARS >0.88) localises them."""
    batch = keypad_macro("PIN code entry screen", {"pin_code": "1234"})
    assert batch is not None
    for a in batch[:-1]:
        assert a["target_description"].startswith("цифра ")
        assert "x" not in a["action_args"] and "y" not in a["action_args"]


def test_macro_handles_russian_kod_label() -> None:
    batch = keypad_macro("экран ввода кода", {"pin_code": "0000"})
    assert batch is not None
    assert len(batch) == 5  # 4 zeros + submit
    assert [_digit(a["target_description"]) for a in batch[:-1]] == ["0", "0", "0", "0"]


def test_macro_silent_on_non_pin_screen() -> None:
    assert keypad_macro("login", {"pin_code": "8520"}) is None
    assert keypad_macro("home", {"pin_code": "8520"}) is None
    assert keypad_macro("payment", {"pin_code": "8520"}) is None


def test_macro_silent_without_code() -> None:
    assert keypad_macro("pin_entry", {}) is None
    assert keypad_macro("pin_entry", None) is None
    assert keypad_macro("pin_entry", {"phone": "9051234567"}) is None


def test_macro_ignores_non_digit_code() -> None:
    assert keypad_macro("pin_entry", {"pin_code": "----"}) is None


def test_macro_custom_credential_key() -> None:
    batch = keypad_macro("pin_entry", {"secret": "4321"}, credential="secret")
    assert batch is not None
    assert [_digit(a["target_description"]) for a in batch[:-1]] == ["4", "3", "2", "1"]
