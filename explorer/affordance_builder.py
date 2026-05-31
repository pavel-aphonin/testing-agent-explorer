"""PER-175 Phase B: build an AffordanceMap from vision detection output.

This is the deterministic bridge between the perception models and the
Platform Adapter. The perception layer (Screen Parser / OmniParser for
boxes, OCR for box text, SigLIP2 or Holo2 for screen type) produces raw
detections; this module classifies each detection into a typed
``Affordance`` and assembles the ``AffordanceMap`` the rest of the
pipeline routes on.

Pure code — no model, no I/O — so the classification logic is unit-tested
independently of whichever vision backend produced the boxes. That
matters because the canvas-PIN failure (run cccc3333) was a *perception*
gap, and the rule "a box containing a single digit on a keypad grid is a
tappable KEYPAD_KEY, not a text field" must be correct and stable
regardless of model.
"""

from __future__ import annotations

import re
from typing import Any

from explorer.affordances import Affordance, AffordanceKind, AffordanceMap

# Submit / confirm vocabulary (RU + EN). Reused conceptually from the
# planner hints; kept local so this module has no import cycle.
_SUBMIT_WORDS: tuple[str, ...] = (
    "вперёд", "вперед", "продолжить", "продолжай", "войти", "вход",
    "готово", "подтвердить", "подтверждаю", "далее", "оплатить",
    "submit", "continue", "next", "done", "enter", "forward", "login",
    "log in", "sign in", "confirm", "pay", "ok", "ок",
)
_BACK_WORDS: tuple[str, ...] = (
    "назад", "вернуться", "отмена", "отменить", "закрыть",
    "back", "cancel", "close", "dismiss",
)

# A keypad is declared once at least this many single-digit keys are seen.
# A 0-9 pad has 10; a 4-key partial still reads as a pad. 6 avoids
# false-positives from a couple of stray numeric labels (e.g. a balance).
_KEYPAD_MIN_DIGITS = 6

_DIGIT_RE = re.compile(r"^\s*([0-9])\s*$")


def _norm(text: str | None) -> str:
    return (text or "").strip().lower()


def _is_digit(text: str | None) -> str | None:
    """Return the single digit a box's text represents, else None."""
    if not text:
        return None
    m = _DIGIT_RE.match(text)
    return m.group(1) if m else None


def _matches(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def _classify_box(
    box: dict[str, Any],
    *,
    editable_regions: list[tuple[int, int, int, int]],
) -> Affordance:
    """Turn one detection box into a typed Affordance.

    ``box`` keys: ``bbox`` ([x1,y1,x2,y2] pixels), ``text`` (OCR/caption,
    optional), ``kind_hint`` (optional backend hint e.g. "textfield"),
    ``confidence``. ``editable_regions`` are AX-derived rects known to be
    text inputs — a box overlapping one is treated as editable even if the
    vision text is empty.
    """
    bbox = box.get("bbox")
    text_raw = box.get("text") or box.get("label") or ""
    text = _norm(text_raw)
    hint = _norm(box.get("kind_hint") or box.get("kind"))
    conf = float(box.get("confidence") or 0.0)

    digit = _is_digit(text_raw)
    if digit is not None:
        return Affordance(
            kind=AffordanceKind.KEYPAD_KEY, label=digit, value=digit,
            bbox=bbox, confidence=conf, source="vision",
        )

    # Editable: an explicit hint, or geometric overlap with an AX field.
    is_editable = (
        "securefield" in hint or "securetext" in hint
        or "textfield" in hint or "textview" in hint
        or "searchfield" in hint or hint == "field"
        or _overlaps_any(bbox, editable_regions)
    )
    if is_editable:
        secure = "secure" in hint or "password" in text or "пароль" in text
        return Affordance(
            kind=AffordanceKind.SECURE_FIELD if secure else AffordanceKind.TEXT_FIELD,
            label=text_raw.strip(), bbox=bbox, editable=True,
            confidence=conf, source="hybrid" if editable_regions else "vision",
        )

    if text and _matches(text, _SUBMIT_WORDS):
        return Affordance(kind=AffordanceKind.SUBMIT, label=text_raw.strip(),
                          bbox=bbox, confidence=conf, source="vision")
    if text and _matches(text, _BACK_WORDS):
        return Affordance(kind=AffordanceKind.BACK, label=text_raw.strip(),
                          bbox=bbox, confidence=conf, source="vision")

    # A detected box with a label but no special role → a generic button;
    # an empty-label box is still actionable (icon button) → OTHER.
    kind = AffordanceKind.BUTTON if text else AffordanceKind.OTHER
    return Affordance(kind=kind, label=text_raw.strip(), bbox=bbox,
                      confidence=conf, source="vision")


def _overlaps_any(
    bbox: list[int] | None, regions: list[tuple[int, int, int, int]]
) -> bool:
    if not bbox or len(bbox) != 4 or not regions:
        return False
    ax1, ay1, ax2, ay2 = bbox
    for bx1, by1, bx2, by2 in regions:
        # any intersection
        if ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2:
            return True
    return False


def build_affordance_map(
    *,
    screen_type: str = "unknown",
    screen_confidence: float = 0.0,
    boxes: list[dict[str, Any]] | None = None,
    editable_regions: list[tuple[int, int, int, int]] | None = None,
    source: str = "vision",
    meta: dict[str, Any] | None = None,
) -> AffordanceMap:
    """Assemble an AffordanceMap from detection boxes + a screen type.

    ``editable_regions`` (optional) are AX-derived text-field rects in the
    same pixel space as the boxes — they let the builder mark a field
    editable even when vision alone can't tell (hybrid perception). When
    omitted the map is pure-vision, which is the canvas case.

    Demotes a lone-digit false positive: single-digit boxes are only
    promoted to a *keypad* (``has_keypad``) when there are
    ``_KEYPAD_MIN_DIGITS`` of them; below that they're still KEYPAD_KEY
    affordances but the screen isn't treated as a keypad screen — the
    Platform Adapter checks ``has_keypad`` before choosing the tap path.
    """
    boxes = boxes or []
    editable_regions = editable_regions or []
    affs = [_classify_box(b, editable_regions=editable_regions) for b in boxes]

    # Keypad sanity: if fewer than the threshold of digit keys, the digits
    # are probably incidental numbers, not a pad — re-type them as OTHER so
    # has_keypad stays False and the adapter doesn't try to "enter a code".
    digit_keys = [a for a in affs if a.kind is AffordanceKind.KEYPAD_KEY]
    if 0 < len(digit_keys) < _KEYPAD_MIN_DIGITS:
        for a in digit_keys:
            a.kind = AffordanceKind.OTHER

    return AffordanceMap(
        screen_type=screen_type or "unknown",
        screen_confidence=screen_confidence,
        affordances=affs,
        source=source,
        meta=meta or {},
    )
