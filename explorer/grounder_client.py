"""Dedicated UI-grounder client (PER-164).

Dense general-purpose VLMs fail at canvas-keypad grounding — they
emit one coordinate and oscillate (see PER-163 retry #2 smoke
comparison: Qwen3-VL 32B, Gemma 4 26B, Qwen 3.6 27B all stuck on
the same point). The fix: route ``tap_at`` decisions with a
``target_description`` arg through a separate model that is
fine-tuned for "screenshot + intent → pixel" — UI-TARS-1.5-7B,
Molmo-7B, ShowUI, etc.

This module is the worker-side glue:

* On first call it pulls the active grounder config from
  ``GET /api/internal/grounder/dispatch`` — endpoint URL, regex
  for parsing, coord-space label, prompt template. Config is
  cached for the worker's lifetime; admin swap requires worker
  restart (same pattern as chat-model config — both are
  infrastructure, not per-run).
* ``locate(screenshot_b64, target_description)`` invokes the
  grounder via OpenAI-compatible chat-completions on its
  configured endpoint, applies the regex to the model's text
  response, returns ``(x, y, coord_space)`` as ints + string,
  or ``None`` on any failure (no grounder configured, HTTP
  error, regex didn't match, etc.). The caller decides whether
  to fall back to the chat-LLM's own coordinate guess (today's
  behaviour) or to fail the action.

Coordinate space is returned alongside the (x, y) tuple so the
caller can apply the correct scaling — the chat-LLM and the
grounder may emit in different spaces (UI-TARS = ``pixels``;
Qwen-VL-as-grounder would be ``normalized_1000``). Centralising
the scale calculation in :func:`ScenarioRunner._dispatch` would
work but coupling the parser to the scaler is uglier; we let the
dispatcher receive the labelled tuple and route.
"""
from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from typing import Optional

import httpx


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GrounderConfig:
    """Snapshot of an active grounder's dispatch contract."""

    name: str
    endpoint_url: str
    tap_at_coord_space: str
    response_format: str
    response_regex: str
    prompt_template: str
    default_temperature: float
    default_top_p: float
    image_min_tokens: Optional[int]
    screenshot_max_dim: Optional[int]


@dataclass(frozen=True)
class GrounderResult:
    """Successful grounder response with the labelled coordinate."""

    x: int
    y: int
    coord_space: str
    raw_text: str
    # PER-199: mean top-1 token probability of the grounding output
    # (0..1), or None when the server didn't return logprobs. Low =
    # the model was unsure where to click.
    confidence: float | None = None


def _grounding_confidence(data: dict) -> float | None:
    """PER-199: mean top-1 token probability across the response.

    Reads the OpenAI-style ``logprobs.content[*].logprob`` list and
    returns ``mean(exp(logprob))``. None when the server didn't return
    logprobs (older llama-server, or the field is absent). Cheap,
    no extra model — this IS the Grounding Verifier.
    """
    import math
    try:
        content = data["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    probs = []
    for tok in content or []:
        lp = tok.get("logprob")
        if lp is not None:
            probs.append(math.exp(lp))
    if not probs:
        return None
    return sum(probs) / len(probs)


class GrounderClient:
    """Async wrapper around the configured grounder llama-server."""

    def __init__(
        self,
        backend_url: str,
        worker_token: str,
        http_timeout_sec: float = 60.0,
    ) -> None:
        self._backend_url = backend_url.rstrip("/")
        self._worker_token = worker_token
        self._timeout = http_timeout_sec
        self._cfg: Optional[GrounderConfig] = None
        self._cfg_fetched = False  # distinguish "not fetched yet" from "fetched, none active"

    async def _ensure_config(self) -> Optional[GrounderConfig]:
        """Pull and cache the active grounder config; None if no grounder is configured."""
        if self._cfg_fetched:
            return self._cfg
        url = f"{self._backend_url}/api/internal/grounder/dispatch"
        headers = {"Authorization": f"Bearer {self._worker_token}"}
        try:
            # trust_env=False: PER-51-style proxy bypass for the same
            # corporate-proxy reasons the worker's BackendClient does
            # — we always hit localhost here.
            async with httpx.AsyncClient(trust_env=False, timeout=self._timeout) as client:
                r = await client.get(url, headers=headers)
            if r.status_code == 404:
                logger.info("No active grounder configured — tap_at will use chat-LLM coords")
                self._cfg = None
                self._cfg_fetched = True
                return None
            r.raise_for_status()
            data = r.json()
            self._cfg = GrounderConfig(
                name=data["name"],
                endpoint_url=data["endpoint_url"].rstrip("/"),
                tap_at_coord_space=data["tap_at_coord_space"],
                response_format=data["response_format"],
                response_regex=data["response_regex"],
                prompt_template=data["prompt_template"],
                default_temperature=float(data.get("default_temperature", 0.0)),
                default_top_p=float(data.get("default_top_p", 1.0)),
                image_min_tokens=data.get("image_min_tokens"),
                screenshot_max_dim=data.get("screenshot_max_dim"),
            )
            self._cfg_fetched = True
            logger.info(
                "Grounder configured: %s @ %s (coord_space=%s, format=%s)",
                self._cfg.name, self._cfg.endpoint_url,
                self._cfg.tap_at_coord_space, self._cfg.response_format,
            )
            return self._cfg
        except Exception:
            logger.exception(
                "Failed to fetch grounder dispatch config from %s — tap_at "
                "will use chat-LLM coords this run", url,
            )
            self._cfg = None
            self._cfg_fetched = True
            return None

    async def locate(
        self,
        screenshot_bytes: bytes,
        target_description: str,
    ) -> Optional[GrounderResult]:
        """Ask the grounder to localise ``target_description`` on ``screenshot_bytes``.

        Returns ``None`` if there is no active grounder, the HTTP
        call fails, the regex doesn't match, or the matched groups
        aren't integer-parseable. The caller should fall back to the
        chat-LLM's own (x, y) in that case.

        ``screenshot_bytes`` is raw PNG/JPEG bytes — base64 encoding
        happens here so callers don't need to know the grounder's
        wire format.
        """
        cfg = await self._ensure_config()
        if cfg is None:
            return None
        if not isinstance(target_description, str) or not target_description.strip():
            logger.warning(
                "grounder.locate called with empty target_description — "
                "skipping (caller should pass the chat-LLM's reasoning hint)",
            )
            return None

        b64 = base64.b64encode(screenshot_bytes).decode("ascii")
        prompt = cfg.prompt_template.replace("{hint}", target_description.strip())
        payload = {
            "model": "grounder",  # llama-server --alias grounder
            "max_tokens": 64,
            "temperature": cfg.default_temperature,
            "top_p": cfg.default_top_p,
            # PER-199: ask llama-server for per-token logprobs so we can
            # compute a confidence on the grounding output (Grounding
            # Verifier = calibration, no separate model). top_logprobs
            # gives us the top-2 gap on each token of the coordinate.
            "logprobs": True,
            "top_logprobs": 2,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
        }
        url = f"{cfg.endpoint_url}/v1/chat/completions"
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=self._timeout) as client:
                r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                or ""
            )
        except Exception:
            logger.exception("Grounder HTTP call to %s failed", url)
            return None

        # PER-199: compute grounding confidence from logprobs. Mean of
        # the per-token top-1 probability across the response; low mean
        # means the model was unsure where to click. Logged for every
        # grounding so we can calibrate a threshold from real data.
        confidence = _grounding_confidence(data)
        if confidence is not None:
            logger.info(
                "[PER-199 verifier] grounding confidence=%.3f target=%r",
                confidence, target_description.strip()[:60],
            )

        if not text:
            logger.warning("Grounder returned empty content (model=%s)", cfg.name)
            return None

        try:
            pattern = re.compile(cfg.response_regex)
        except re.error:
            logger.exception(
                "Grounder response_regex doesn't compile: %r — fix the DB row",
                cfg.response_regex,
            )
            return None

        match = pattern.search(text)
        if not match or len(match.groups()) < 2:
            logger.warning(
                "Grounder response didn't match regex (model=%s, format=%s): %r",
                cfg.name, cfg.response_format, text[:200],
            )
            return None

        try:
            x = int(match.group(1))
            y = int(match.group(2))
        except (TypeError, ValueError):
            logger.warning(
                "Grounder matched regex but groups aren't integers: %r",
                match.groups(),
            )
            return None

        return GrounderResult(
            x=x, y=y,
            coord_space=cfg.tap_at_coord_space,
            raw_text=text,
            confidence=confidence,
        )
