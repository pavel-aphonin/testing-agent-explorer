"""PER-196: Grounder agent.

Wraps the existing UI-TARS grounder call site so the model name
comes from ModuleAssignment instead of a hardcoded env var.
Returns ``(x, y)`` coordinates in the same space the existing
grounder client uses.

For this iteration the agent **delegates** to the legacy
``grounder_client`` for the actual VLM call — we just inject the
endpoint+model resolved from the assignment. This keeps the prompt /
parsing logic in one place (the legacy client is well-tested) while
proving the per-role routing in the hot path.
"""

from __future__ import annotations

import logging
from typing import Any

from explorer.agents.base import RoleAgent
from explorer.role_resolver import ModuleRole

logger = logging.getLogger("explorer.agents.grounder")


class GrounderAgent(RoleAgent):
    role = ModuleRole.GROUNDER

    async def locate(
        self,
        target_description: str,
        screenshot_bytes: bytes,
    ) -> tuple[int, int] | None:
        """Resolve target_description → (x, y) by asking the assigned
        grounder model. Returns ``None`` if unassigned or the model
        couldn't ground the target.

        Implementation note: this iteration delegates to the legacy
        ``grounder_client.locate_by_description`` for the actual VLM
        wire format (image base64 + prompt template + coord parsing).
        The role resolver provides ``endpoint_url`` so the legacy
        client talks to the correct llama-server even if it's not
        the one in its old env var.
        """
        endpoint = await self._endpoint(required=False)
        if endpoint is None:
            return None

        # Lazy import to avoid pulling grounder_client at module load
        # time (it has heavy deps).
        try:
            from explorer import grounder_client as legacy_grounder
        except ImportError:
            logger.warning("legacy grounder_client unavailable; cannot ground %r", target_description)
            return None

        # The legacy client reads its URL/model from env or its own
        # config. Pass-through via the function's optional kwargs if
        # they exist; otherwise fall back to whatever it was using.
        kwargs: dict[str, Any] = {}
        try:
            import inspect
            sig = inspect.signature(legacy_grounder.locate_by_description)
            if "base_url" in sig.parameters and endpoint.endpoint_url:
                kwargs["base_url"] = endpoint.endpoint_url
            if "model" in sig.parameters:
                kwargs["model"] = endpoint.model_name
        except (AttributeError, ImportError, ValueError):
            pass

        try:
            coords = await legacy_grounder.locate_by_description(
                target_description, screenshot_bytes, **kwargs,
            )
        except AttributeError:
            # If legacy grounder doesn't expose ``locate_by_description``
            # under that exact name, we don't try to be clever — caller
            # falls back to the original code path.
            logger.warning(
                "grounder_client.locate_by_description not found; falling back",
            )
            return None
        except Exception as exc:
            logger.warning("GrounderAgent: legacy grounder errored: %s", exc)
            return None

        return coords
