"""PER-175 Phase F: VLM-based Context Identifier (Holo2-8B).

The real blindness fix. The SigLIP2 zero-shot classifier gave near-uniform
screen-type scores (~0.09) and OmniParser produced text-less boxes, so on a
canvas PIN screen the pipeline still couldn't tell it was a keypad
(smoke dddd4444). Holo2-8B is a screenshot-native GUI VLM (Qwen3-VL-8B based)
that, in one call, returns BOTH the screen type AND the actionable elements
with their bounding boxes — its UI-QA strength.

This agent POSTs the screenshot to Holo2's OpenAI-compatible
``/v1/chat/completions`` (multimodal content: text instruction + image) and
parses the JSON reply directly into an ``AffordanceMap`` — the same contract
the Platform Adapter already consumes. So swapping perception from
SigLIP→Holo2 is just routing the Context Identifier through this agent; the
rest of the 13-module chain is unchanged.

Degrades to ``None`` when the role is unassigned / the model errors / the
reply doesn't parse, so the runner falls back to the SigLIP path.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import TYPE_CHECKING

import httpx

from explorer.affordances import Affordance, AffordanceKind, AffordanceMap
from explorer.role_resolver import ModuleRole

if TYPE_CHECKING:
    from explorer.role_resolver import RoleResolver

logger = logging.getLogger("explorer.agents.vision_context")


# The instruction Holo2 answers. We ask for a compact JSON the agent maps
# straight into an AffordanceMap. Kinds mirror AffordanceKind so the reply
# needs no translation table. RU+EN because the app under test is Russian.
VISION_SYSTEM = """You are a mobile-UI perception module. Look at the
screenshot and return STRICT JSON describing the screen — nothing else.

Schema:
{
  "screen_type": "pin_entry|sms_entry|login|password_entry|home|payment|transactions|settings|permission_dialog|error_dialog|loading|unknown",
  "confidence": 0.0-1.0,
  "elements": [
    {"kind": "keypad_key|text_field|secure_field|submit|button|back|tab|link|toggle|other",
     "label": "<visible text or digit>",
     "value": "<the digit for a keypad key, else null>",
     "bbox": [x1, y1, x2, y2]}
  ]
}

Rules:
- A numeric on-screen keypad → one "keypad_key" per digit button, with
  "value" set to that digit ("0".."9"). This is critical: a PIN/passcode
  screen is drawn on a canvas with NO text field — report the digit keys.
- A real editable text input → "text_field" (or "secure_field" for password).
- The confirm/continue/login button → "submit".
- bbox in PIXELS of the screenshot you were given. Be precise.
- Return ONLY the JSON object."""

VISION_USER = "Опиши этот экран по схеме. Верни только JSON."


class VisionContextAgent:
    """Holo2-style VLM Context Identifier. Resolves the CONTEXT_IDENTIFIER
    role's endpoint (so the operator can point it at Holo2 via
    ModuleAssignment) and calls it with the screenshot."""

    role = ModuleRole.CONTEXT_IDENTIFIER

    def __init__(self, resolver: "RoleResolver", *, timeout: float = 60.0):
        self._resolver = resolver
        self._timeout = timeout

    async def classify_vision_vlm(self, png: bytes) -> AffordanceMap | None:
        """Return an AffordanceMap from the screenshot via the VLM, or
        ``None`` when unassigned / the model errors / parse fails."""
        endpoint = await self._resolver.resolve(self.role, required=False)
        if endpoint is None or not endpoint.endpoint_url:
            return None
        # Only meaningful for a vision chat model. A pytorch_microservice
        # (SigLIP) assignment is handled by the other agent, not here.
        if not endpoint.supports_vision or endpoint.provider == "pytorch_microservice":
            return None

        url = f"{endpoint.endpoint_url.rstrip('/')}/v1/chat/completions"
        b64 = base64.b64encode(png).decode()
        body = {
            "model": endpoint.model_name,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": VISION_USER},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as c:
                r = await c.post(url, json=body)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("VisionContext.classify_vision_vlm HTTP/shape error: %s", exc)
            return None

        amap = parse_vlm_affordances(content)
        if amap is None:
            logger.warning("VisionContext: could not parse VLM reply: %s", str(content)[:160])
        return amap


def parse_vlm_affordances(content: str) -> AffordanceMap | None:
    """Parse a Holo2 JSON reply into an AffordanceMap. Pure + tolerant of
    ```json fences and a <think> preamble (Holo2 is Thinking-based)."""
    if not content:
        return None
    text = content.strip()
    # strip a reasoning preamble + code fences
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    affs: list[Affordance] = []
    for el in obj.get("elements") or []:
        if not isinstance(el, dict):
            continue
        kind_raw = (el.get("kind") or "other").lower()
        try:
            kind = AffordanceKind(kind_raw)
        except ValueError:
            kind = AffordanceKind.OTHER
        bbox = el.get("bbox")
        bbox = list(bbox) if isinstance(bbox, list) and len(bbox) == 4 else None
        affs.append(Affordance(
            kind=kind,
            label=str(el.get("label") or ""),
            value=(str(el["value"]) if el.get("value") is not None else None),
            bbox=bbox,
            editable=kind in (AffordanceKind.TEXT_FIELD, AffordanceKind.SECURE_FIELD),
            source="vlm",
        ))
    return AffordanceMap(
        screen_type=str(obj.get("screen_type") or "unknown"),
        screen_confidence=float(obj.get("confidence") or 0.0),
        affordances=affs,
        source="vlm",
    )
