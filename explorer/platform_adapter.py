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


def _resolve_credential(
    intent: dict[str, Any],
    amap: AffordanceMap,
    test_data_keys: list[str],
) -> list[dict[str, Any]]:
    """Map a ``provide_credential`` intent to a concrete batch using the
    screen's affordances. This is the canvas-PIN fix, generalised."""
    cred = (
        intent.get("credential")
        or intent.get("value_source", "").replace("test_data.", "")
        or "pin_code"
    )
    value_source = f"test_data.{cred}" if cred in test_data_keys else "test_data." + cred
    reason = f"PER-175 resolver: provide {cred} via the screen's input mechanism."

    # 1) On-screen keypad (canvas, no text field) → tap digits in order + submit.
    if amap.has_keypad:
        digits = amap.digit_keys_in_order()
        batch: list[dict[str, Any]] = []
        # We tap by digit description so the grounder localises each key;
        # bbox (when present) lets ScreenSeekeR pre-narrow the region.
        for a in digits:
            c = a.center()
            batch.append(_action(
                "tap_at",
                target_description=f"цифра {a.value} на экранной клавиатуре",
                x=c[0] if c else None,
                y=c[1] if c else None,
                value_source=value_source,
                reasoning=f"tap keypad digit {a.value}",
            ))
        # PER-204 as a RULE: a credential entry on a keypad always ends
        # with the submit press. Prefer a detected submit affordance.
        subs = amap.submit_buttons
        if subs:
            c = subs[0].center()
            batch.append(_action(
                "tap_at",
                target_description=subs[0].label or _SUBMIT_DESC,
                x=c[0] if c else None,
                y=c[1] if c else None,
                reasoning="PER-204: submit after the code is entered.",
            ))
        else:
            batch.append(_action(
                "tap_at", target_description=_SUBMIT_DESC,
                reasoning="PER-204: submit after the code (grounded by description).",
            ))
        if not digits:
            # Keypad detected but digits not individually resolved → let the
            # grounder type via the keypad by description, still + submit.
            logger.info("[adapter] keypad with no resolved digit keys — description fallback")
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
) -> list[dict[str, Any]]:
    """Resolve ONE planner intent into a concrete action batch.

    Unknown / already-concrete intents pass through unchanged (wrapped in a
    list) so the adapter is safe to run over any planner output — it only
    *adds* mechanism where an abstract intent needs it.
    """
    test_data_keys = test_data_keys or []
    name = (intent.get("intent") or intent.get("action") or "").strip().lower()

    if name == INTENT_PROVIDE_CREDENTIAL:
        return _resolve_credential(intent, amap, test_data_keys)

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


def resolve_plan(
    intents: list[dict[str, Any]] | None,
    amap: AffordanceMap,
    test_data_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve a whole plan (list of intents) into a flat concrete batch.

    This is what the ``actions.resolved`` bus stage publishes and the
    synchronous path calls before grounding/dispatch.
    """
    out: list[dict[str, Any]] = []
    for intent in intents or []:
        if isinstance(intent, dict):
            out.extend(resolve_intent(intent, amap, test_data_keys))
    return out
