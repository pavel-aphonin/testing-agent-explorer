"""PER-203 Phase 3: pure planner-prompt hints (shared sync + bus).

These were inline in ``scenario_runner``; extracted verbatim (same
wording, same thresholds) so behaviour is identical and the bus
planner-runner can import the exact same intelligence. All pure — no
``self``, no I/O — except they take already-fetched inputs (e.g. the
digit-tap count, the context classification is done by the caller).
"""

from __future__ import annotations

import re


def loop_breaker_hint(history: list[str] | None) -> str | None:
    """PER-200 loop-breaker: 3+ near-identical recent actions → «you're
    stuck» rule. Digits/whitespace stripped so re-tapping different
    keypad digits counts as one repeated kind. None if <3 or they differ.
    """
    if not history or len(history) < 3:
        return None

    def _norm(s: str) -> str:
        return re.sub(r"\d+|\s+", "", s.lower())[:60]

    last3 = [_norm(h) for h in history[-3:]]
    if last3[0] and last3[0] == last3[1] == last3[2]:
        return (
            "⚠️ ЦИКЛ: последние 3 действия практически одинаковы и НЕ "
            "продвигают к цели. ОБЯЗАТЕЛЬНО выбери ДРУГОЕ действие — "
            "другой элемент, другой тип действия (например tap_at по "
            "видимой кнопке подтверждения внизу экрана), либо проверь, "
            "не нужно ли подтвердить уже введённые данные."
        )
    return None


def credential_routing_hint(test_data_keys: set[str]) -> str | None:
    """PER-200 credential→screen routing. Tells the Planner which
    test_data key maps to which code-entry screen so it stops grabbing
    the wrong credential (the sms_code-on-PIN-screen bug). Lists only
    keys that exist; values stay masked. None if <2 keys to disambiguate.
    """
    rules: list[str] = []
    if "pin_code" in test_data_keys:
        rules.append("экран ПИН-кода (4 цифры, заголовок про код/PIN) → используй test_data.pin_code")
    if "sms_code" in test_data_keys:
        rules.append("экран кода из СМС (упоминание SMS/смс/одноразовый код) → используй test_data.sms_code")
    if "password" in test_data_keys:
        rules.append("экран временного пароля / пароля → используй test_data.password")
    if "phone" in test_data_keys:
        rules.append("экран номера телефона → используй test_data.phone")
    if len(rules) < 2:
        return None
    return (
        "📋 СООТВЕТСТВИЕ ДАННЫХ И ЭКРАНОВ (выбирай value_source строго "
        "по типу текущего экрана, НЕ путай коды между собой):\n  - "
        + "\n  - ".join(rules)
    )


def count_digit_taps(history: list[str] | None) -> int:
    """How many digit/keypad/code entries are already in history.

    Loose by design: over-counting only fires the submit hint a step
    early (harmless); under-counting is the failure we're fixing.
    """
    if not history:
        return 0
    n = 0
    for h in history:
        hl = h.lower()
        if ("tap" in hl or "input" in hl or "ввод" in hl) and re.search(r"\b\d\b|цифр|pin|код", hl):
            n += 1
    return n


def pin_keypad_hint() -> str:
    """PER-203 Phase 4 fix: on a PIN/code screen, force the digit-by-digit
    keypad strategy as a single batch.

    The bus smoke showed 8B-Think choosing ``enter_text "8520"`` on a
    canvas keypad — it knew the value but the screen has NO text field,
    so enter_text silently no-ops and the model loops. This hint tells
    it explicitly: tap the digit buttons one by one (a full batch), then
    the submit button — never enter_text/input. Fires whenever the
    Context Identifier says PIN, BEFORE the >=4-digit submit gate.
    """
    return (
        "⚠️ Это экран ввода кода через ЭКРАННУЮ цифровую клавиатуру БЕЗ "
        "текстового поля. НЕ используй enter_text или input — печатать "
        "некуда, действие ничего не сделает и экран не изменится. Чтобы "
        "ввести код, верни МАССИВ actions: по одному действию tap_at на "
        "каждую цифру кода ПО ПОРЯДКУ (нажми кнопку-цифру за кнопкой-"
        "цифрой), и ПОСЛЕДНИМ действием — tap_at по кнопке подтверждения "
        "(«Вперёд»/«Продолжить»/«Войти»). Все действия одним батчем."
    )


def pin_submit_hint(digit_taps: int) -> str | None:
    """PER-198 PIN-submit rule. Returns the hard «code is full, press
    submit now» rule once >=4 digit taps happened. Caller gates on the
    screen actually being a PIN screen (Context Identifier). None below
    threshold.
    """
    if digit_taps < 4:
        return None
    return (
        "⚠️ КОНТЕКСТ ЭКРАНА: это экран ввода PIN/секретного кода, и "
        f"ты уже ввёл {digit_taps} цифр (достаточно для полного кода). "
        "НЕ нажимай больше цифры и НЕ нажимай «Назад». СЛЕДУЮЩЕЕ "
        "действие ОБЯЗАНО быть подтверждением: action=tap_at с "
        "target_description=\"кнопка Вперёд/Продолжить/Войти внизу "
        "экрана\". Это единственный способ продвинуться дальше."
    )
