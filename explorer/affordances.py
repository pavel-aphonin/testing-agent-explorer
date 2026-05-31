"""PER-175 full integration: the shared affordance vocabulary.

The **affordance map** is the perception layer's structured answer to
"what can be done on this screen" — derived from VISION (OmniParser
boxes + a vision screen-type classification) rather than the
accessibility tree. This is the fix for canvas/Flutter screens (e.g. a
PIN keypad painted on a canvas) where the AX-tree is empty and the old
text-only Context Identifier went blind (run cccc3333).

It is the contract between the perception modules (Screen Parser,
Dynamic Perceiver, Context Identifier) and the **Platform Adapter** —
the 12th module, pure code — which turns a planner *intent* ("provide
pin_code here") into a concrete atomic action batch ("tap digits 8,5,2,0
then the submit button"). Splitting WHAT (intent) from HOW (mechanism)
is the decoupled Planner→resolver→Grounder pattern the 2025-26 research
(CODA, SeeAct-V, Agent S2) converged on.

Pure data + plain types so it serialises onto the Redis Streams bus
without dragging perception deps into every runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AffordanceKind(str, Enum):
    """What a single on-screen element lets the agent do.

    The distinction that matters for the canvas-keypad bug:
    ``TEXT_FIELD``/``SECURE_FIELD`` accept typed text (``enter_text`` works),
    while ``KEYPAD_KEY`` is a painted button you must *tap* — typing into
    the screen does nothing. The Platform Adapter routes on exactly this.
    """

    TEXT_FIELD = "text_field"      # editable, accepts typed characters
    SECURE_FIELD = "secure_field"  # password/secure editable field
    KEYPAD_KEY = "keypad_key"      # one key of an on-screen (canvas) keypad
    SUBMIT = "submit"             # confirm / continue / login / forward
    BUTTON = "button"
    TAB = "tab"
    LINK = "link"
    TOGGLE = "toggle"
    BACK = "back"
    OTHER = "other"


# Kinds that physically accept typed text. enter_text / input on anything
# else is a silent device no-op (the PER-205 failure mode).
EDITABLE_KINDS: frozenset[AffordanceKind] = frozenset(
    {AffordanceKind.TEXT_FIELD, AffordanceKind.SECURE_FIELD}
)


@dataclass
class Affordance:
    """One actionable element on the screen.

    ``bbox`` is ``[x1, y1, x2, y2]`` in the pixel space of the screenshot
    that produced it (the Grounder/dispatcher scales to device points).
    ``value`` carries the key's character for ``KEYPAD_KEY`` (e.g. "8").
    """

    kind: AffordanceKind
    label: str = ""
    bbox: list[int] | None = None
    value: str | None = None            # e.g. the digit for a keypad key
    editable: bool = False
    source: str = "vision"             # vision | ax | hybrid
    confidence: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def center(self) -> tuple[int, int] | None:
        if not self.bbox or len(self.bbox) != 4:
            return None
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "label": self.label,
            "bbox": list(self.bbox) if self.bbox else None,
            "value": self.value,
            "editable": self.editable,
            "source": self.source,
            "confidence": self.confidence,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Affordance":
        kind_raw = d.get("kind") or AffordanceKind.OTHER.value
        try:
            kind = AffordanceKind(kind_raw)
        except ValueError:
            kind = AffordanceKind.OTHER
        return cls(
            kind=kind,
            label=d.get("label") or "",
            bbox=list(d["bbox"]) if d.get("bbox") else None,
            value=d.get("value"),
            editable=bool(d.get("editable", kind in EDITABLE_KINDS)),
            source=d.get("source") or "vision",
            confidence=float(d.get("confidence") or 0.0),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class AffordanceMap:
    """The full vision-derived description of the current screen.

    ``screen_type`` is the Context Identifier's verdict (``pin_entry`` /
    ``sms_entry`` / ``login`` / ``home`` / …) — replacing the boolean
    ``context_is_pin`` that conflated screen type with input mechanism.
    """

    screen_type: str = "unknown"
    screen_confidence: float = 0.0
    affordances: list[Affordance] = field(default_factory=list)
    source: str = "vision"
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- convenience views the Platform Adapter routes on -------------
    @property
    def has_keypad(self) -> bool:
        return any(a.kind is AffordanceKind.KEYPAD_KEY for a in self.affordances)

    @property
    def editable_fields(self) -> list[Affordance]:
        return [a for a in self.affordances if a.kind in EDITABLE_KINDS]

    @property
    def keypad_keys(self) -> list[Affordance]:
        return [a for a in self.affordances if a.kind is AffordanceKind.KEYPAD_KEY]

    @property
    def submit_buttons(self) -> list[Affordance]:
        return [a for a in self.affordances if a.kind is AffordanceKind.SUBMIT]

    def digit_keys_in_order(self) -> list[Affordance]:
        """Keypad digit keys sorted by their value (0-9) — what the
        resolver taps, in code order, to enter a numeric secret."""
        digits = [
            a for a in self.keypad_keys
            if (a.value or "").strip().isdigit()
        ]
        return sorted(digits, key=lambda a: int(a.value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "screen_type": self.screen_type,
            "screen_confidence": self.screen_confidence,
            "affordances": [a.to_dict() for a in self.affordances],
            "source": self.source,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AffordanceMap":
        if not isinstance(d, dict):
            return cls()
        return cls(
            screen_type=d.get("screen_type") or "unknown",
            screen_confidence=float(d.get("screen_confidence") or 0.0),
            affordances=[
                Affordance.from_dict(a)
                for a in (d.get("affordances") or [])
                if isinstance(a, dict)
            ],
            source=d.get("source") or "vision",
            meta=dict(d.get("meta") or {}),
        )

    @property
    def is_pin_entry(self) -> bool:
        """Back-compat bridge for the old boolean while callers migrate
        to ``screen_type``. True for PIN/secret-code screens."""
        st = (self.screen_type or "").lower()
        return "pin" in st or "secret" in st or "код" in st
