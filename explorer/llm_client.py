"""LLM client for AI/Hybrid exploration modes.

Talks to any OpenAI-compatible /v1/chat/completions endpoint (llama-server,
llama-swap, vLLM, OpenAI itself). Used by the engine as a `prior_provider`:
given a freshly-discovered ScreenNode, ask the LLM which interactive
element is most worth tapping next, then turn the answer into a prior
distribution over PUCT actions.

Design choices:
- Output is plain JSON, NOT function calling. Smaller open models like
  Gemma 4 E4B handle JSON-mode poorly with grammars; a one-shot prompt
  + parse-with-fallback is more robust.
- The prompt is small and structured (one element per line) so a 4B
  model has a fighting chance.
- Failures fall back to uniform priors, never crash the run.
- The provider is async so it doesn't block the event loop while a
  llama-server worker chews on tokens.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from explorer.engine import action_id_for
from explorer.models import ScreenNode

logger = logging.getLogger("explorer.llm_client")


# Hard cap on tokens we'll spend per call. The model can hallucinate
# infinitely without this and a single call eats minutes.
MAX_TOKENS = 256

# Hard cap on the number of elements we ask the LLM to rank. Anything
# beyond this is uniform — the prompt would be too noisy and the model
# can't keep more than ~20 elements straight in its head anyway.
MAX_ELEMENTS_IN_PROMPT = 20


SYSTEM_PROMPT = """You are an automated mobile app explorer. \
You are given a list of interactive UI elements visible on the current screen \
of a mobile app. Your job is to assign a *priority score* (0.0 to 1.0) to each \
element, where higher scores mean "this element is more interesting to tap \
next to discover new app states". Prefer:

- Buttons with clear navigational labels ("Login", "Settings", "Continue")
- Tabs and menu items
- Anything that looks like it would open a new screen or perform an action

Avoid:
- Pure decoration (back buttons in flows you've already explored, branding)
- Elements that obviously dismiss things you don't care about

Respond with ONLY a JSON object mapping each element's index (as a string) \
to its score. Do not include explanations. Example:
{"0": 0.9, "1": 0.3, "2": 0.7}"""


class LLMClient:
    """Thin async OpenAI-compatible chat client."""

    def __init__(
        self,
        base_url: str,
        model_name: str = "embeddings",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    async def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = MAX_TOKENS,
        screenshot_b64: str | None = None,
        response_format: dict | None = None,
    ) -> str | None:
        """Single-shot chat completion. Returns the assistant's text, or None on error.

        When ``screenshot_b64`` is provided the user message is sent as
        an OpenAI vision content-list — text + image_url(data:image/png;
        base64,…). Both Gemma 4 E4B and Qwen 3.5 35B-A3B are vision-
        capable through llama-server's mmproj sidecar (PER-22), so this
        unlocks "agent looks at the screen" mode without changing the
        infrastructure.

        PER-111: when ``response_format`` is passed, it goes straight
        into the llama-server payload. llama.cpp compiles JSON schemas
        into GBNF grammars under the hood, so a ``{"type":"json_schema",
        "json_schema":{...}}`` value gives us constrained decoding —
        the model physically cannot emit tokens outside the schema. If
        the server returns 4xx mentioning ``response_format`` /
        ``json_schema`` / ``grammar``, we retry the call ONCE without
        the format so the goal node degrades gracefully on older builds
        instead of failing outright.
        """
        if screenshot_b64:
            user_content: Any = [
                {"type": "text", "text": user},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot_b64}",
                    },
                },
            ]
        else:
            user_content = user

        async def _post(client: httpx.AsyncClient, body: dict) -> str | None:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions", json=body
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        base_payload: dict = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, trust_env=False
            ) as client:
                payload = dict(base_payload)
                if response_format is not None:
                    payload["response_format"] = response_format
                try:
                    return await _post(client, payload)
                except httpx.HTTPStatusError as exc:
                    if response_format is None or not 400 <= exc.response.status_code < 500:
                        raise
                    body_text = exc.response.text.lower()
                    if not any(
                        tok in body_text
                        for tok in ("response_format", "json_schema", "grammar")
                    ):
                        raise
                    logger.warning(
                        "LLM rejected response_format (%s); retrying "
                        "without grammar — output will be plain text JSON.",
                        exc.response.status_code,
                    )
                    return await _post(client, base_payload)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning("LLM chat call failed: %s", exc)
            return None


def _build_user_prompt(elements: list) -> str:
    """Format the elements as a numbered list the LLM can index by."""
    lines = ["The current screen has these interactive elements:", ""]
    for i, el in enumerate(elements):
        kind = el.kind.value if hasattr(el.kind, "value") else str(el.kind)
        label = el.label or "(no label)"
        test_id = f" [test_id={el.test_id}]" if el.test_id else ""
        lines.append(f"{i}. [{kind}] {label!r}{test_id}")
    lines.append("")
    lines.append(
        "Return a JSON object mapping each element index (as a string) to a "
        "priority score between 0.0 and 1.0. Higher = more interesting to tap next."
    )
    return "\n".join(lines)


def _parse_scores(raw: str, n_elements: int) -> dict[int, float] | None:
    """Pull a {idx: score} JSON dict out of the model's response.

    Robust to:
    - extra text before/after the JSON
    - markdown code fences
    - integer keys instead of string keys
    - scores outside [0, 1] (clipped)
    """
    if not raw:
        return None

    # First try a direct JSON parse
    candidates: list[str] = []
    raw = raw.strip()
    if raw.startswith("```"):
        # Strip code fence
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    candidates.append(raw)

    # Then look for the first {...} in the text
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if not isinstance(obj, dict):
                continue
            scores: dict[int, float] = {}
            for k, v in obj.items():
                try:
                    idx = int(k)
                    score = float(v)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < n_elements:
                    scores[idx] = max(0.0, min(1.0, score))
            if scores:
                return scores
        except (json.JSONDecodeError, ValueError):
            continue
    return None


class LLMPriorProvider:
    """Callable that turns a ScreenNode into a {action_id: prior} dict.

    The PUCT selector normalizes whatever we return — we don't have to
    make scores sum to 1, just rank them sensibly.

    ``vision_enabled`` (PER-22): when True and the ScreenNode carries
    a screenshot, send it alongside the textual element list. Visual
    bugs (cropped text, weird colors, hidden buttons) become catchable;
    the model can also disambiguate two elements with identical labels
    by their on-screen position.
    """

    def __init__(self, client: LLMClient, vision_enabled: bool = False) -> None:
        self.client = client
        # ``TA_LLM_VISION`` overrides the per-mode setting at runtime —
        # set to "0"/"false" to debug a vision-related model glitch
        # without touching mode config.
        import os as _os
        env = _os.environ.get("TA_LLM_VISION")
        if env is not None:
            self.vision_enabled = env.strip().lower() in ("1", "true", "yes", "on")
        else:
            self.vision_enabled = vision_enabled

    async def __call__(self, node: ScreenNode) -> dict[str, float]:
        elements = node.interactive_elements[:MAX_ELEMENTS_IN_PROMPT]
        if not elements:
            return {}

        user_prompt = _build_user_prompt(elements)
        # Only attach the screenshot if the mode wants it AND we have
        # one — passing None is cheaper than sending a placeholder.
        screenshot = node.screenshot_b64 if self.vision_enabled else None
        raw = await self.client.chat(SYSTEM_PROMPT, user_prompt, screenshot_b64=screenshot)
        if raw is None:
            logger.info(
                "LLM returned no priors for screen %r — uniform", node.name
            )
            return {}

        scores = _parse_scores(raw, n_elements=len(elements))
        if scores is None:
            logger.warning(
                "Failed to parse LLM scores for screen %r; raw=%r",
                node.name,
                raw[:200],
            )
            return {}

        priors: dict[str, float] = {}
        for idx, score in scores.items():
            el = elements[idx]
            priors[action_id_for(el)] = score

        # Elements not mentioned by the LLM get a small floor so they're
        # still reachable, just deprioritized.
        floor = 0.05
        for el in elements:
            aid = action_id_for(el)
            if aid not in priors:
                priors[aid] = floor

        logger.info(
            "LLM scored %d/%d elements on screen %r",
            len(scores),
            len(elements),
            node.name,
        )
        return priors
