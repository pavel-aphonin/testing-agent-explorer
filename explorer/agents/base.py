"""PER-196: base class for per-role agent wrappers.

A ``RoleAgent`` resolves the assigned model lazily on first use and
talks to its OpenAI-compatible endpoint. Per-role subclasses override
``system_prompt`` and ``parse_response`` and pass their call args in.

The base intentionally does NOT cache the HTTP client — each call uses
a short-lived httpx.AsyncClient. This is cheap on localhost and lets
operators change assignments without restarting the worker (the
RoleResolver's TTL cache governs how soon the change takes effect).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from explorer.role_resolver import ModelEndpoint, ModuleRole, RoleResolver

logger = logging.getLogger("explorer.agents")


# Where llama-swap listens for chat completions. Used as the default
# when an ``LLMModel.endpoint_url`` is NULL (the common case — most
# models live behind llama-swap on localhost:8080).
DEFAULT_LLAMA_SWAP_URL = "http://host.docker.internal:8080"


@dataclass
class RoleAgentResult:
    """Wrap an LLM call result with the role + model that produced it.

    Used to enrich diagnostics — when ``planner.decide()`` returns
    something useful we want to log "Planner (gui-owl-1.5-4b) → tap
    submit_btn" without callers having to thread the model name
    through.
    """

    role: str
    model_name: str
    content: str
    raw: dict[str, Any]


class RoleAgent:
    """Base class for per-role wrappers."""

    role: "ModuleRole"  # set by subclasses

    def __init__(self, resolver: "RoleResolver", llama_swap_url: str = DEFAULT_LLAMA_SWAP_URL):
        self._resolver = resolver
        self._llama_swap_url = llama_swap_url.rstrip("/")

    async def _endpoint(self, *, required: bool = True) -> "ModelEndpoint | None":
        return await self._resolver.resolve(self.role, required=required)

    def _chat_url(self, endpoint: "ModelEndpoint") -> str:
        """Pick the URL to POST chat completions to.

        Order:
            1. Per-model ``endpoint_url`` from the LLMModel passport
               (set when the model lives behind its own server, not
               llama-swap).
            2. Default llama-swap URL — llama-swap routes by
               ``model`` field in the request body.
        """
        return f"{(endpoint.endpoint_url or self._llama_swap_url).rstrip('/')}/v1/chat/completions"

    async def call(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> RoleAgentResult | None:
        """Plain OpenAI chat completion against the resolved endpoint.

        Returns ``None`` when the role is unassigned (caller's
        responsibility to handle — typically by falling back to the
        legacy monolith path). Network / 5xx errors are logged and
        also return ``None`` so a single LLM hiccup doesn't kill the
        run.
        """
        endpoint = await self._endpoint(required=False)
        if endpoint is None:
            logger.debug("RoleAgent[%s]: role unassigned, returning None", self.role.value)
            return None

        body: dict[str, Any] = {
            "model": endpoint.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else endpoint.default_temperature,
            "top_p": endpoint.default_top_p,
            "max_tokens": max_tokens,
        }
        if endpoint.default_top_k is not None:
            body["top_k"] = endpoint.default_top_k
        if endpoint.default_min_p is not None:
            body["min_p"] = endpoint.default_min_p
        if response_format and endpoint.supports_json_schema:
            body["response_format"] = response_format

        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                resp = await client.post(self._chat_url(endpoint), json=body)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning(
                "RoleAgent[%s] HTTP error against %s: %s",
                self.role.value, endpoint.model_name, exc,
            )
            return None
        except Exception as exc:
            logger.warning("RoleAgent[%s] unexpected error: %s", self.role.value, exc)
            return None

        # llama-server returns the OpenAI-compatible shape; pluck the
        # first choice's text. Empty response → return None so callers
        # know to fall back, instead of propagating an empty string
        # that would JSON-parse to nothing useful downstream.
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.warning(
                "RoleAgent[%s] unexpected response shape: %s",
                self.role.value, str(data)[:200],
            )
            return None
        if not content:
            return None

        return RoleAgentResult(
            role=self.role.value,
            model_name=endpoint.model_name,
            content=content,
            raw=data,
        )
