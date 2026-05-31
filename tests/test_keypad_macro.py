"""PER-175 Phase C: tests for the deterministic keypad macro + its gate.

Two halves, both hardened after live smoke dddd4444 (where SigLIP's weak
screen-type label fired the macro on the LOGIN screen):

  * should_fire_keypad_macro — the SAFETY gate. Fires only on corroborated
    evidence: a real keypad grid (OmniParser >=6 digit keys) OR a
    high-confidence PIN screen_type. A weak label alone must NOT fire.
  * keypad_macro — builds the batch (secret's digits in order + submit),
    by description so the proven Grounder path places coordinates.
"""

from __future__ import annotations

import re

from explorer.affordances import Affordance, AffordanceKind, AffordanceMap
from explorer.platform_adapter import keypad_macro, should_fire_keypad_macro


def _digit(desc: str) -> str:
    m = re.search(r"цифра (\d)", desc or "")
    return m.group(1) if m else "?"


def _keypad_map(screen_type: str = "pin_entry", conf: float = 0.1) -> AffordanceMap:
    keys = [
        Affordance(kind=AffordanceKind.KEYPAD_KEY, value=str(d), label=str(d))
        for d in range(10)
    ]
    return AffordanceMap(screen_type=screen_type, screen_confidence=conf,
                         affordances=keys)


# ── the SAFETY gate ───────────────────────────────────────────────────


def test_gate_fires_on_real_keypad_grid_regardless_of_label() -> None:
    """has_keypad (>=6 digit keys) is unambiguous structural evidence — it
    fires even if the (weak) screen-type label disagrees."""
    m = _keypad_map(screen_type="login", conf=0.09)  # label says login!
    assert m.has_keypad is True
    assert should_fire_keypad_macro(m) is True


def test_gate_does_not_fire_on_weak_label_without_keypad() -> None:
    """The dddd4444 bug: SigLIP said pin_entry @0.09 on the LOGIN screen,
    no keypad detected → must NOT fire."""
    m = AffordanceMap(screen_type="pin_entry", screen_confidence=0.09,
                      affordances=[])
    assert m.has_keypad is False
    assert should_fire_keypad_macro(m) is False


def test_gate_fires_on_high_confidence_pin_label() -> None:
    """Secondary path: once a real VLM (Holo2) gives a confident PIN label
    (>=0.6), fire even before OmniParser corroboration."""
    m = AffordanceMap(screen_type="pin_entry", screen_confidence=0.82,
                      affordances=[])
    assert should_fire_keypad_macro(m) is True


def test_gate_does_not_fire_on_non_pin_high_confidence() -> None:
    m = AffordanceMap(screen_type="login", screen_confidence=0.95, affordances=[])
    assert should_fire_keypad_macro(m) is False


# ── the batch builder ─────────────────────────────────────────────────


def test_macro_taps_secret_digits_in_order_then_submit() -> None:
    batch = keypad_macro({"pin_code": "8520"})
    assert batch is not None
    assert len(batch) == 5  # 4 digits + submit
    assert all(a["action"] == "tap_at" for a in batch)
    assert [_digit(a["target_description"]) for a in batch[:-1]] == ["8", "5", "2", "0"]
    assert any(w in batch[-1]["target_description"] for w in ("Вперёд", "Продолжить", "Войти"))


def test_macro_taps_by_description_only_no_coords() -> None:
    batch = keypad_macro({"pin_code": "1234"})
    assert batch is not None
    for a in batch[:-1]:
        assert a["target_description"].startswith("цифра ")
        assert "x" not in a["action_args"] and "y" not in a["action_args"]


def test_macro_silent_without_code() -> None:
    assert keypad_macro({}) is None
    assert keypad_macro(None) is None
    assert keypad_macro({"phone": "9051234567"}) is None


def test_macro_ignores_non_digit_code() -> None:
    assert keypad_macro({"pin_code": "----"}) is None


def test_macro_custom_credential_key() -> None:
    batch = keypad_macro({"secret": "4321"}, credential="secret")
    assert batch is not None
    assert [_digit(a["target_description"]) for a in batch[:-1]] == ["4", "3", "2", "1"]
