"""PER-175: tests for the Platform Adapter (12th module).

The keystone of the WHAT→HOW split. These lock in the two behaviours that
the run cccc3333 post-mortem turned into deterministic rules:

  * PER-204 — a credential on a canvas keypad resolves to digit taps + a
    trailing submit (the submit can no longer be "forgotten" by the model).
  * PER-205 — a credential on an editable field resolves to enter_text,
    and is NEVER typed into a non-editable target.
"""

from __future__ import annotations

from explorer.affordances import Affordance, AffordanceKind, AffordanceMap
from explorer.platform_adapter import resolve_intent, resolve_plan


def _keypad_map(with_submit: bool = True) -> AffordanceMap:
    keys = [
        Affordance(kind=AffordanceKind.KEYPAD_KEY, value=str(d), label=str(d),
                   bbox=[30 * d, 100, 30 * d + 28, 140])
        for d in range(10)
    ]
    affs = list(keys)
    if with_submit:
        affs.append(Affordance(kind=AffordanceKind.SUBMIT, label="Вперёд",
                               bbox=[200, 800, 320, 850]))
    return AffordanceMap(screen_type="pin_entry", affordances=affs)


def _field_map() -> AffordanceMap:
    return AffordanceMap(
        screen_type="login",
        affordances=[
            Affordance(kind=AffordanceKind.TEXT_FIELD, label="Телефон",
                       editable=True, meta={"element_id": "phoneField"}),
            Affordance(kind=AffordanceKind.SUBMIT, label="Войти"),
        ],
    )


# ── PER-204: keypad credential → the SECRET's digits in order + submit ─


def test_credential_on_keypad_taps_secret_digits_in_order_then_submit() -> None:
    """The keypad path taps the digits of the ACTUAL secret (8-5-2-0), in
    secret order — not the ten keys 0-9. The value comes from the worker's
    resolve_value callable, never from a prompt."""
    amap = _keypad_map(with_submit=True)
    batch = resolve_intent(
        {"intent": "provide_credential", "credential": "pin_code"},
        amap, ["pin_code"],
        resolve_value=lambda k: "8520" if k == "pin_code" else None,
    )
    actions = [a["action"] for a in batch]
    assert all(a == "tap_at" for a in actions)   # canvas → taps, never enter_text
    # 4 digit taps (8,5,2,0) + submit
    assert len(batch) == 5
    digit_taps = batch[:-1]
    assert [_digit_of(a["target_description"]) for a in digit_taps] == ["8", "5", "2", "0"]
    # The submit is last (PER-204 rule).
    assert batch[-1]["target_description"] == "Вперёд"
    # Each digit tap carries the detected key's coords (for ScreenSeekeR).
    assert all("x" in a["action_args"] for a in digit_taps)


def test_keypad_submit_appended_even_without_detected_button() -> None:
    """No SUBMIT affordance detected → still append a submit by description
    (PER-204 must hold regardless of detection)."""
    amap = _keypad_map(with_submit=False)
    batch = resolve_intent(
        {"intent": "provide_credential", "credential": "pin_code"},
        amap, ["pin_code"],
        resolve_value=lambda k: "8520",
    )
    assert batch[-1]["action"] == "tap_at"
    desc = batch[-1]["target_description"]
    assert any(w in desc for w in ("Вперёд", "Продолжить", "Войти"))
    assert len(batch) == 5  # 4 digits + submit


def test_keypad_without_resolved_value_falls_back_not_guesses() -> None:
    """No resolve_value → must NOT guess 0-9; falls back to a keypad
    enter_text (value via dispatcher) + submit. Never taps random digits."""
    amap = _keypad_map(with_submit=True)
    batch = resolve_intent(
        {"intent": "provide_credential", "credential": "pin_code"},
        amap, ["pin_code"],  # no resolve_value
    )
    # No digit taps were guessed.
    digit_taps = [a for a in batch if "цифра" in (a.get("target_description") or "")]
    assert digit_taps == []
    # Ends with submit.
    assert batch[-1]["target_description"] == "Вперёд"


def _digit_of(desc: str) -> str:
    """Extract the digit char from 'цифра 8 на экранной клавиатуре'."""
    import re
    m = re.search(r"цифра (\d)", desc or "")
    return m.group(1) if m else "?"


# ── PER-205: field credential → enter_text, never into non-editable ───


def test_credential_on_editable_field_uses_enter_text() -> None:
    amap = _field_map()
    batch = resolve_intent(
        {"intent": "provide_credential", "credential": "phone"},
        amap, ["phone"],
    )
    assert len(batch) == 1
    assert batch[0]["action"] == "enter_text"
    assert batch[0]["element_label"] == "Телефон"
    assert batch[0]["value_source"] == "test_data.phone"


def test_credential_with_no_affordance_falls_back_to_grounded_tap() -> None:
    """Neither keypad nor field detected → a tap_at by description so the
    grounder can still localise. Never a blind enter_text."""
    amap = AffordanceMap(screen_type="unknown", affordances=[])
    batch = resolve_intent(
        {"intent": "provide_credential", "credential": "pin_code"},
        amap, ["pin_code"],
    )
    assert len(batch) == 1
    assert batch[0]["action"] == "tap_at"
    assert batch[0]["target_description"]


# ── intent vocabulary + passthrough ──────────────────────────────────


def test_submit_intent_resolves_to_detected_button() -> None:
    amap = _field_map()
    batch = resolve_intent({"intent": "submit"}, amap, [])
    assert batch[0]["action"] == "tap_at"
    assert batch[0]["target_description"] == "Войти"


def test_navigate_back_intent() -> None:
    batch = resolve_intent({"intent": "navigate_back"}, AffordanceMap(), [])
    assert batch == [b for b in batch if b["action"] == "back"]
    assert batch[0]["action"] == "back"


def test_concrete_action_passes_through_unchanged() -> None:
    """A planner that already emitted a concrete action (legacy path) is
    returned as-is — the adapter only adds mechanism to abstract intents."""
    concrete = {"action": "wait", "action_args": {"ms": 500}, "reasoning": "x"}
    out = resolve_intent(concrete, AffordanceMap(), [])
    assert out == [concrete]


def test_resolve_plan_flattens_mixed_intents() -> None:
    amap = _keypad_map()
    plan = [
        {"intent": "provide_credential", "credential": "pin_code"},
        {"action": "wait", "action_args": {"ms": 300}},
    ]
    out = resolve_plan(plan, amap, ["pin_code"],
                       resolve_value=lambda k: "8520")
    # 4 secret digits + submit + the passthrough wait = 6
    assert len(out) == 6
    assert out[-1]["action"] == "wait"


def test_resolve_plan_handles_empty_and_none() -> None:
    assert resolve_plan(None, AffordanceMap(), []) == []
    assert resolve_plan([], AffordanceMap(), []) == []
