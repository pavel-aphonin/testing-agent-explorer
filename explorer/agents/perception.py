"""PER-198: perception agents — Context Identifier / Dynamic Perceiver /
Screen Parser.

Unlike the LLM agents (base.RoleAgent, OpenAI chat protocol), these hit
the perception microservice's REST endpoints (/classify, /compare,
/parse). They still resolve their endpoint via RoleResolver so the
operator controls the wiring through ModuleAssignment, but the wire
format is bespoke per service.

All three degrade gracefully: if the role is unassigned or the service
is down, the method returns ``None`` and the caller falls back to its
legacy behaviour (AX-tree only, no context label, etc.).
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from explorer.role_resolver import ModuleRole

if TYPE_CHECKING:
    from explorer.role_resolver import RoleResolver

logger = logging.getLogger("explorer.agents.perception")


class _PerceptionAgent:
    """Shared endpoint resolution for perception services."""

    role: ModuleRole

    def __init__(self, resolver: "RoleResolver"):
        self._resolver = resolver

    async def _base_url(self) -> str | None:
        endpoint = await self._resolver.resolve(self.role, required=False)
        if endpoint is None or not endpoint.endpoint_url:
            return None
        return endpoint.endpoint_url.rstrip("/")


# ─────────────────────────────────────────────────────────────────────
# Context Identifier — the key module for the PIN bug
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ContextResult:
    label: str
    confidence: float
    all_scores: dict[str, float]

    @property
    def is_pin_entry(self) -> bool:
        """True when the top label looks like a PIN/secret-code screen.

        Handles BOTH the legacy text-classifier labels (e.g. "PIN code
        entry screen") and the new vision screen-type keys (e.g.
        "pin_entry"). Used by scenario_runner to inject the «after N
        digits, the submit button is mandatory» hint into the Planner
        prompt — the behavioural fix the routing-only refactor (PER-196)
        couldn't deliver on its own.
        """
        l = self.label.lower()
        return "pin" in l or "secret" in l or "код" in l


class ContextIdentifierAgent(_PerceptionAgent):
    role = ModuleRole.CONTEXT_IDENTIFIER

    async def classify(
        self,
        screen_text: str,
        candidate_labels: list[str] | None = None,
        *,
        timeout: float = 20.0,
    ) -> ContextResult | None:
        base = await self._base_url()
        if base is None:
            return None
        body: dict = {"text": screen_text}
        if candidate_labels:
            body["candidate_labels"] = candidate_labels
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as c:
                r = await c.post(f"{base}/classify", json=body)
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning("ContextIdentifier.classify failed: %s", exc)
            return None
        return ContextResult(
            label=data["label"],
            confidence=float(data["confidence"]),
            all_scores=data.get("all_scores", {}),
        )

    async def classify_vision(
        self,
        png: bytes,
        candidate_types: dict[str, str] | None = None,
        *,
        timeout: float = 30.0,
    ) -> ContextResult | None:
        """PER-175 Phase B: classify the screen TYPE from the SCREENSHOT
        (SigLIP2 zero-shot), not the AX-tree text.

        This is the blindness fix: on a canvas keypad the AX-tree is empty
        but the image is unambiguous. Returns a ``ContextResult`` whose
        ``label`` is the short screen-type key (e.g. ``pin_entry``).
        ``None`` when the role is unassigned / service down → caller falls
        back to the text classifier or legacy behaviour.
        """
        base = await self._base_url()
        if base is None:
            return None
        body: dict = {"png_b64": base64.b64encode(png).decode()}
        if candidate_types:
            body["candidate_types"] = candidate_types
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as c:
                r = await c.post(f"{base}/classify_vision", json=body)
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning("ContextIdentifier.classify_vision failed: %s", exc)
            return None
        return ContextResult(
            label=data["screen_type"],
            confidence=float(data["confidence"]),
            all_scores=data.get("all_scores", {}),
        )


# ─────────────────────────────────────────────────────────────────────
# Dynamic Perceiver — "did the screen actually change?"
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SimilarityResult:
    similarity: float
    changed: bool


class DynamicPerceiverAgent(_PerceptionAgent):
    role = ModuleRole.DYNAMIC_PERCEIVER

    async def compare(
        self,
        before_png: bytes,
        after_png: bytes,
        *,
        changed_threshold: float = 0.98,
        timeout: float = 20.0,
    ) -> SimilarityResult | None:
        base = await self._base_url()
        if base is None:
            return None
        body = {
            "before_png_b64": base64.b64encode(before_png).decode(),
            "after_png_b64": base64.b64encode(after_png).decode(),
            "changed_threshold": changed_threshold,
        }
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as c:
                r = await c.post(f"{base}/compare", json=body)
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning("DynamicPerceiver.compare failed: %s", exc)
            return None
        return SimilarityResult(
            similarity=float(data["similarity"]),
            changed=bool(data["changed"]),
        )


# ─────────────────────────────────────────────────────────────────────
# Screen Parser — OmniParser-v2 element detection
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ParsedScreen:
    elements: list[dict]  # [{bbox:[x1,y1,x2,y2] normalized, confidence}]
    count: int


class ScreenParserAgent(_PerceptionAgent):
    role = ModuleRole.SCREEN_PARSER

    async def parse(
        self, png: bytes, *, box_threshold: float = 0.05, timeout: float = 60.0
    ) -> ParsedScreen | None:
        base = await self._base_url()
        if base is None:
            return None
        body = {"png_b64": base64.b64encode(png).decode(), "box_threshold": box_threshold}
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as c:
                r = await c.post(f"{base}/parse", json=body)
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning("ScreenParser.parse failed: %s", exc)
            return None
        return ParsedScreen(elements=data.get("elements", []), count=data.get("count", 0))
