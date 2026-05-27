"""PER-196: Memory agent — text → embedding vector.

Used by the episodic memory layer to retrieve similar past screens.
Talks to llama-server's /v1/embeddings endpoint (not /v1/chat/completions
like the other agents), so this one doesn't go through RoleAgent.call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from explorer.agents.base import DEFAULT_LLAMA_SWAP_URL, RoleAgent
from explorer.role_resolver import ModuleRole

if TYPE_CHECKING:
    from explorer.role_resolver import RoleResolver

logger = logging.getLogger("explorer.agents.memory")


class MemoryAgent(RoleAgent):
    role = ModuleRole.MEMORY

    async def embed(self, text: str, *, timeout: float = 30.0) -> list[float] | None:
        """Return an embedding vector or ``None`` when unassigned /
        the model errored."""
        endpoint = await self._endpoint(required=False)
        if endpoint is None:
            return None

        base = (endpoint.endpoint_url or self._llama_swap_url).rstrip("/")
        url = f"{base}/v1/embeddings"
        body = {"model": endpoint.model_name, "input": text}

        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
            logger.warning("MemoryAgent.embed failed (%s): %s", endpoint.model_name, exc)
            return None
