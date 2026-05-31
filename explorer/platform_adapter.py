"""PER-175 full integration: the Platform Adapter — the 12th module.

This is the **HOW** half of the decoupled Planner→resolver→Grounder
architecture the 2025-26 research (CODA, SeeAct-V, Agent S2) converged on.

The Planner now emits *intents* — WHAT to achieve, referencing a visible
affordance — e.g. ``provide_credential(pin_code)`` or ``tap(submit)``. The
Platform Adapter maps each intent + the vision-derived ``AffordanceMap``
to a **concrete atomic action batch** the executor can dispatch — choosing
the mechanism from what the screen actually affords:

  * credential + on-screen (canvas) keypad → tap each digit key in order,
    then the submit button. **This is PER-204 as a deterministic rule**,
    not a prompt hint the model may ignore.
  * credential + editable text field → ``enter_text`` into that field.
    **PER-205 guard**: only when the target is genuinely editable; a
    non-editable target is never typed into (the canvas no-op bug).
  * credential + neither detected → emit a ``tap_at`` with a
    target_description so the Grounder can still localise by vision.

Pure code, no model, no I/O — fully unit-testable. It is the keystone
that makes the canvas-PIN screen work regardless of which planner/grounder
models are plugged in.
"""

from __future__ import annotations

import logging
from typing import Any

from explorer.affordances import AffordanceKind, AffordanceMap

logger = logging.getLogger("explorer.platform_adapter")


# ── intent vocabulary ────────────────────────────────────────────────
# The planner emits these (WHAT). Kept deliberately small; each maps to a
# resolution rule below. ``provide_credential`` is the one that needed the
# mechanism split (field vs keypad) — the rest are near-passthrough.
INTENT_PROVIDE_CREDENTIAL = "provide_credential"
INTENT_TAP = "tap"
INTENT_SUBMIT = "submit"
INTENT_NAVIGATE_BACK = "navigate_back"
INTENT_WAIT = "wait"
INTENT_SCROLL = "scroll"
INTENT_DONE = "done"


_SUBMIT_DESC = "кнопка подтверждения внизу экрана (Вперёд / Продолжить / Войти)"


def _action(
    action: str,
    *,
    target_description: str | None = None,
    x: int | None = None,
    y: int | None = None,
    element_id: str | None = None,
    element_label: str | None = None,
    value_source: str = "none",
    value_literal: str | None = None,
    reasoning: str = "",
    **extra_args: Any,
) -> dict[str, Any]:
    """Build one concrete action dict in the canonical executor shape
    (same keys the goal-node dispatch + grounder already consume)."""
    args: dict[str, Any] = dict(extra_args)
    if target_description is not None:
        args["target_description"] = target_description
    if x is not None:
        args["x"] = x
    if y is not None:
        args["y"] = y
    return {
        "action": action,
        "action_args": args,
        "target_description": target_description,
        "element_id": element_id,
        "element_label": element_label,
        "value_source": value_source,
        "value_literal": value_literal,
        "reasoning": reasoning,
    }


def _submit_action(amap: AffordanceMap) -> dict[str, Any]:
    """PER-204 as a rule: the submit press that ends a keypad entry.
    Prefer a detected submit affordance; else ground by description."""
    subs = amap.submit_buttons
    if subs:
        c = subs[0].center()
        return _action(
            "tap_at",
            target_description=subs[0].label or _SUBMIT_DESC,
            x=c[0] if c else None,
            y=c[1] if c else None,
            reasoning="PER-204: submit after the code is entered.",
        )
    return _action(
        "tap_at", target_description=_SUBMIT_DESC,
        reasoning="PER-204: submit after the code (grounded by description).",
    )


def _resolve_credential(
    intent: dict[str, Any],
    amap: AffordanceMap,
    test_data_keys: list[str],
    resolved_value: str | None = None,
) -> list[dict[str, Any]]:
    """Map a ``provide_credential`` intent to a concrete batch using the
    screen's affordances. This is the canvas-PIN fix, generalised.

    ``resolved_value`` is the actual secret the worker substituted from
    test_data (e.g. "8520"). It's needed ONLY for the keypad path —
    tapping a canvas keypad is inherently value-dependent (you tap 8-5-2-0,
    not the ten keys). It is pure code in the worker; the value never
    enters an LLM prompt. For the text-field path the value stays a
    value_source the dispatcher substitutes at type time.
    """
    cred = (
        intent.get("credential")
        or intent.get("value_source", "").replace("test_data.", "")
        or "pin_code"
    )
    value_source = f"test_data.{cred}"
    reason = f"PER-175 resolver: provide {cred} via the screen's input mechanism."

    # 1) On-screen keypad (canvas, no text field) → tap the secret's digits
    #    IN SECRET ORDER, then submit.
    if amap.has_keypad:
        batch: list[dict[str, Any]] = []
        # Map detected keys by their character so we can attach a bbox
        # (lets ScreenSeekeR pre-narrow); description drives the grounder.
        key_by_char = {
            (k.value or ""): k for k in amap.keypad_keys if k.value
        }
        digits = [ch for ch in (resolved_value or "") if ch.isdigit()]
        if digits:
            for ch in digits:
                k = key_by_char.get(ch)
                c = k.center() if k else None
                batch.append(_action(
                    "tap_at",
                    target_description=f"цифра {ch} на экранной клавиатуре",
                    x=c[0] if c else None,
                    y=c[1] if c else None,
                    reasoning=f"tap keypad digit {ch}",
                ))
        else:
            # No resolved value (shouldn't happen — worker has test_data).
            # Don't guess digits; hand the keypad to the grounder by
            # description so the step is at least attempted, then submit.
            logger.info(
                "[adapter] keypad credential with no resolved value — "
                "description fallback"
            )
            batch.append(_action(
                "enter_text",
                target_description="экранная клавиатура ввода кода",
                value_source=value_source,
                reasoning=f"{reason} (keypad, value via dispatcher).",
            ))
        batch.append(_submit_action(amap))
        return batch

    # 2) Editable text field present → type into it (PER-205: only editable).
    fields = amap.editable_fields
    if fields:
        f = fields[0]
        return [_action(
            "enter_text",
            target_description=f.label or "поле ввода",
            element_id=(f.meta or {}).get("element_id"),
            element_label=f.label or None,
            value_source=value_source,
            reasoning=f"{reason} (editable field).",
        )]

    # 3) Nothing detected → hand the grounder a description to localise.
    logger.info("[adapter] no keypad/field detected for credential — grounder fallback")
    return [_action(
        "tap_at",
        target_description=f"поле или клавиатура для ввода {cred}",
        value_source=value_source,
        reasoning=f"{reason} (no affordance detected; grounder will localise).",
    )]


def resolve_intent(
    intent: dict[str, Any],
    amap: AffordanceMap,
    test_data_keys: list[str] | None = None,
    resolve_value: "Any" = None,
) -> list[dict[str, Any]]:
    """Resolve ONE planner intent into a concrete action batch.

    Unknown / already-concrete intents pass through unchanged (wrapped in a
    list) so the adapter is safe to run over any planner output — it only
    *adds* mechanism where an abstract intent needs it.

    ``resolve_value`` (optional) is a callable ``credential_key -> value``
    the worker provides so the keypad path can expand a secret into its
    digit taps. Pure code; the value never enters a prompt. When omitted,
    keypad credential entry falls back to a description + value_source the
    dispatcher substitutes.
    """
    test_data_keys = test_data_keys or []
    name = (intent.get("intent") or intent.get("action") or "").strip().lower()

    if name == INTENT_PROVIDE_CREDENTIAL:
        cred = (
            intent.get("credential")
            or intent.get("value_source", "").replace("test_data.", "")
            or "pin_code"
        )
        resolved = None
        if callable(resolve_value):
            try:
                resolved = resolve_value(cred)
            except Exception:  # never let value lookup break resolution
                logger.exception("[adapter] resolve_value(%s) failed", cred)
        return _resolve_credential(intent, amap, test_data_keys, resolved)

    if name in (INTENT_SUBMIT,):
        subs = amap.submit_buttons
        if subs:
            c = subs[0].center()
            return [_action(
                "tap_at", target_description=subs[0].label or _SUBMIT_DESC,
                x=c[0] if c else None, y=c[1] if c else None,
                reasoning="resolve submit intent to the detected submit button.",
            )]
        return [_action("tap_at", target_description=_SUBMIT_DESC,
                        reasoning="submit intent (grounded by description).")]

    if name == INTENT_NAVIGATE_BACK:
        return [_action("back", reasoning="navigate back intent.")]

    # Passthrough: the planner already emitted a concrete action
    # (tap/tap_at/enter_text/wait/scroll/…). Return it as-is.
    return [intent]


def keypad_macro(
    screen_type: str,
    test_data: dict[str, str] | None,
    *,
    credential: str | None = None,
) -> list[dict[str, Any]] | None:
    """PER-175 Phase C: deterministic PIN-keypad driver.

    The blindness-fix payoff. When VISION classifies the screen as a
    PIN/secret-code entry (``screen_type``) and we hold the code in
    ``test_data``, we DON'T ask the planner how to type it — on a canvas
    keypad the answer is always "tap the digit buttons in order, then
    submit". This sidesteps the run-cccc3333 failure where the planner
    chose ``enter_text`` on a canvas (a no-op) and looped.

    Returns the concrete batch (digit taps by description + submit) for the
    worker to dispatch — each ``tap_at`` carries only a ``target_description``
    so the proven Grounder-by-description path (UI-TARS localised these at
    >0.88 confidence in cccc3333) places the actual coordinate. The secret
    is read here in pure worker-side code and never leaves the process.

    ``None`` when the screen isn't a PIN entry or we have no code — the
    normal planner flow then handles the screen.
    """
    st = (screen_type or "").lower()
    is_pin = "pin" in st or "secret" in st or ("код" in st and "qr" not in st)
    if not is_pin:
        return None
    cred = credential or "pin_code"
    value = (test_data or {}).get(cred)
    digits = [c for c in str(value or "") if c.isdigit()]
    if not digits:
        return None
    batch = [
        _action(
            "tap_at",
            target_description=f"цифра {d} на экранной клавиатуре",
            reasoning=f"PER-175 keypad macro: tap digit {d} (grounded by description)",
        )
        for d in digits
    ]
    batch.append(_action(
        "tap_at", target_description=_SUBMIT_DESC,
        reasoning="PER-204: submit after the code is entered.",
    ))
    return batch


def resolve_plan(
    intents: list[dict[str, Any]] | None,
    amap: AffordanceMap,
    test_data_keys: list[str] | None = None,
    resolve_value: "Any" = None,
) -> list[dict[str, Any]]:
    """Resolve a whole plan (list of intents) into a flat concrete batch.

    This is what the ``actions.resolved`` bus stage publishes and the
    synchronous path calls before grounding/dispatch. ``resolve_value`` is
    the worker's ``credential_key -> value`` callable (see resolve_intent).
    """
    out: list[dict[str, Any]] = []
    for intent in intents or []:
        if isinstance(intent, dict):
            out.extend(resolve_intent(intent, amap, test_data_keys, resolve_value))
    return out
