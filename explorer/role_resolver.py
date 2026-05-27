"""PER-195: per-role LLM endpoint resolver with TTL cache.

Provides ``resolve_role(role)`` that asks the backend which model is
currently assigned to a given ``ModuleRole`` and caches the answer for
``CACHE_TTL_SEC`` seconds.

Why a separate module instead of inlining the lookup in each call site:
    * Sub-second hot paths (planner.decide on every step) shouldn't
      hammer the backend — a 5-min cache amortises the cost of
      operator changes.
    * Tests can monkeypatch ``resolve_role`` to a fake mapping without
      mocking the BackendClient.
    * The ``ModelEndpoint`` dataclass is the **single shape** all
      role-specific call sites read from — they don't know about the
      raw backend response.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from explorer.worker import BackendClient

logger = logging.getLogger("explorer.role_resolver")


CACHE_TTL_SEC = 300  # 5 minutes — see module docstring


class ModuleRole(str, Enum):
    """Mirror of backend ``app.models.module_assignment.ModuleRole``.

    Worker holds its own copy so this module doesn't pull SQLAlchemy
    into the explorer process. Keep the string values in lock-step
    with the backend — they're the URL segment for the API.
    """

    SCREEN_PARSER = "SCREEN_PARSER"
    DYNAMIC_PERCEIVER = "DYNAMIC_PERCEIVER"
    CONTEXT_IDENTIFIER = "CONTEXT_IDENTIFIER"
    PLANNER = "PLANNER"
    GROUNDER = "GROUNDER"
    GROUNDING_VERIFIER = "GROUNDING_VERIFIER"
    MEMORY = "MEMORY"
    REFLECTION = "REFLECTION"
    SAFETY_GUARD = "SAFETY_GUARD"
    REWARD_CRITIC = "REWARD_CRITIC"
    AMBIGUITY_RESOLVER = "AMBIGUITY_RESOLVER"


ALL_MODULE_ROLES: tuple[ModuleRole, ...] = tuple(ModuleRole)


@dataclass
class ModelEndpoint:
    """The flat shape every role-call site reads from.

    Mirrors the backend ``/api/internal/module-assignments/{role}``
    response, minus fields the worker doesn't use yet. New fields
    added on the backend just appear in ``extras`` until a caller
    pulls them up.
    """

    role: ModuleRole
    model_name: str
    family: str
    provider: str
    endpoint_url: str | None
    context_length: int
    supports_vision: bool
    supports_tool_use: bool
    supports_thinking: bool
    supports_json_schema: bool
    thinking_activation: str | None
    thinking_extract_regex: str | None
    default_temperature: float
    default_top_p: float
    default_top_k: int | None
    default_min_p: float | None
    tap_at_coord_space: str
    image_min_tokens: int | None
    screenshot_max_dim: int | None
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_backend_payload(cls, role: ModuleRole, payload: dict) -> "ModelEndpoint":
        """Build an instance from the raw backend dict.

        Unknown fields land in ``extras`` so a backend addition
        doesn't break the worker — caller can `endpoint.extras.get(...)`
        to read them.
        """
        known = {
            "model_name", "family", "provider", "endpoint_url",
            "context_length", "supports_vision", "supports_tool_use",
            "supports_thinking", "supports_json_schema",
            "thinking_activation", "thinking_extract_regex",
            "default_temperature", "default_top_p", "default_top_k",
            "default_min_p", "tap_at_coord_space", "image_min_tokens",
            "screenshot_max_dim",
        }
        extras = {k: v for k, v in payload.items() if k not in known | {"role", "model_id", "gguf_path", "mmproj_path", "supported_roles"}}
        return cls(
            role=role,
            model_name=payload["model_name"],
            family=payload.get("family", "unknown"),
            provider=payload.get("provider", "llama_cpp"),
            endpoint_url=payload.get("endpoint_url"),
            context_length=int(payload.get("context_length", 32768)),
            supports_vision=bool(payload.get("supports_vision", False)),
            supports_tool_use=bool(payload.get("supports_tool_use", False)),
            supports_thinking=bool(payload.get("supports_thinking", False)),
            supports_json_schema=bool(payload.get("supports_json_schema", True)),
            thinking_activation=payload.get("thinking_activation"),
            thinking_extract_regex=payload.get("thinking_extract_regex"),
            default_temperature=float(payload.get("default_temperature", 0.7)),
            default_top_p=float(payload.get("default_top_p", 0.9)),
            default_top_k=payload.get("default_top_k"),
            default_min_p=payload.get("default_min_p"),
            tap_at_coord_space=payload.get("tap_at_coord_space", "points"),
            image_min_tokens=payload.get("image_min_tokens"),
            screenshot_max_dim=payload.get("screenshot_max_dim"),
            extras=extras,
        )


class RoleNotAssignedError(RuntimeError):
    """Raised when a call site needs a role that has no model assigned."""

    def __init__(self, role: ModuleRole):
        super().__init__(
            f"Role {role.value} has no LLM model assigned. "
            f"Assign one via /admin/module-assignments in the SPA."
        )
        self.role = role


class RoleResolver:
    """TTL-cached lookup of role → ModelEndpoint via the backend.

    Lifetime is per-run (constructed in the worker run setup). Cache
    is in-memory only — operator changes propagate after at most
    ``CACHE_TTL_SEC`` seconds, which is fine for the "all 11 hot"
    baseline because changing assignments mid-run is a rare event
    and the eventual-consistency wait is bounded.
    """

    def __init__(self, backend: "BackendClient", ttl_sec: float = CACHE_TTL_SEC):
        self._backend = backend
        self._ttl = ttl_sec
        # role.value → (expires_at_unix, ModelEndpoint | None)
        # None caches the "explicitly unassigned" answer too so we
        # don't re-query a NULL row every step.
        self._cache: dict[str, tuple[float, ModelEndpoint | None]] = {}

    async def resolve(
        self, role: ModuleRole, *, required: bool = True
    ) -> ModelEndpoint | None:
        """Look up the endpoint for ``role``.

        ``required=True`` (default) raises ``RoleNotAssignedError`` on
        a NULL assignment. ``required=False`` returns ``None`` —
        useful for optional roles like GROUNDING_VERIFIER.
        """
        now = time.time()
        cached = self._cache.get(role.value)
        if cached and cached[0] > now:
            endpoint = cached[1]
        else:
            payload = await self._backend.get_module_assignment(role.value)
            endpoint = (
                ModelEndpoint.from_backend_payload(role, payload)
                if payload
                else None
            )
            self._cache[role.value] = (now + self._ttl, endpoint)

        if endpoint is None and required:
            raise RoleNotAssignedError(role)
        return endpoint

    async def probe_inventory(self) -> dict[str, ModelEndpoint | None]:
        """Resolve every role once and return the map.

        Used at run startup to log a clear inventory ("PLANNER →
        gui-owl-1.5-4b, GROUNDER → ui-tars-1.5-7b, …, AMBIGUITY → unset"),
        so an operator can spot a missing assignment before the run
        eats minutes hitting it on a hot path.
        """
        out: dict[str, ModelEndpoint | None] = {}
        for role in ALL_MODULE_ROLES:
            try:
                out[role.value] = await self.resolve(role, required=False)
            except Exception as exc:  # defensive — never fail startup
                logger.warning("probe_inventory(%s) errored: %s", role.value, exc)
                out[role.value] = None
        return out


def format_inventory(inventory: dict[str, ModelEndpoint | None]) -> str:
    """Pretty one-liner per role for the startup log."""
    lines = []
    for role_value, endpoint in inventory.items():
        if endpoint is None:
            lines.append(f"  {role_value:22s} → (unset)")
        else:
            vis = " +vision" if endpoint.supports_vision else ""
            lines.append(
                f"  {role_value:22s} → {endpoint.model_name} "
                f"({endpoint.family}{vis})"
            )
    return "PER-175 role inventory:\n" + "\n".join(lines)
