"""PER-175 full integration: every one of the 13 module-runners builds a
handler (no role raises) and the code-only handlers transform correctly.

These don't hit Redis or any model — they construct a ModuleRunner with a
fake resolver (all roles unassigned) and drive the handler with a payload.
The point: prove all 13 are wired AND that the pure-code stages (Platform
Adapter, ScreenSeeker, Grounding Verifier, side-consumers) do the right
thing with no backend, so a live run can't trip over a missing handler.
"""

from __future__ import annotations

import asyncio

import pytest

from explorer.bus.runner import ModuleRunner, ROLE_WIRING
from explorer.role_resolver import ModuleRole


class _FakeResolver:
    """Every role unassigned → agents return None, handlers degrade."""

    async def resolve(self, role, *, required=True):
        return None


def _runner(role: ModuleRole) -> ModuleRunner:
    r = ModuleRunner(role, "http://localhost:8000", "tok")
    return r


async def _build(role: ModuleRole):
    """Build a role's handler with the resolver monkeypatched to the fake."""
    r = _runner(role)
    # _build_handler constructs its own RoleResolver(backend); patch the
    # class so no network happens.
    import explorer.bus.runner as mod
    import explorer.role_resolver as rr
    orig = rr.RoleResolver
    rr.RoleResolver = lambda *a, **k: _FakeResolver()  # type: ignore
    # also patch the symbol imported into the runner module namespace
    mod.RoleResolver = rr.RoleResolver  # type: ignore
    try:
        return await r._build_handler()
    finally:
        rr.RoleResolver = orig
        mod.RoleResolver = orig


def test_all_13_roles_build_a_handler() -> None:
    """No ModuleRole raises 'No bus handler' — all 13 are implemented."""
    async def _run():
        for role in ROLE_WIRING:
            h = await _build(role)
            assert callable(h), f"{role.value} produced no handler"
    asyncio.run(_run())


def test_platform_adapter_handler_resolves_and_passes_payload() -> None:
    """The adapter stage rewrites actions via the affordance map and keeps
    the rest of the payload (so downstream stages still see the screenshot)."""
    async def _run():
        h = await _build(ModuleRole.PLATFORM_ADAPTER)
        payload = {
            "run_id": "r1", "screenshot_b64": "Zm9v",
            "affordance_map": {
                "screen_type": "login",
                "affordances": [
                    {"kind": "text_field", "label": "Телефон", "editable": True},
                ],
            },
            "actions": [{"intent": "provide_credential", "credential": "phone"}],
            "test_data_keys": ["phone"],
        }
        out = await h(payload)
        assert out is not None
        # intent resolved to a concrete enter_text on the field
        assert out["actions"][0]["action"] == "enter_text"
        # payload preserved for downstream
        assert out["screenshot_b64"] == "Zm9v"
        assert out["run_id"] == "r1"
    asyncio.run(_run())


def test_screen_seeker_attaches_region_for_known_affordance() -> None:
    """ScreenSeeker adds a search_region to a tap whose description matches a
    detected affordance bbox; leaves others untouched; never drops actions."""
    async def _run():
        h = await _build(ModuleRole.SCREEN_SEEKER)
        payload = {
            "affordance_map": {
                "screen_type": "pin_entry",
                "affordances": [
                    {"kind": "keypad_key", "value": "8", "label": "8",
                     "bbox": [10, 20, 50, 60]},
                ],
            },
            "actions": [
                {"action": "tap_at", "action_args": {"target_description": "цифра 8 keypad"}},
                {"action": "wait", "action_args": {"ms": 200}},
            ],
        }
        out = await h(payload)
        assert len(out["actions"]) == 2  # nothing dropped
        assert out["actions"][0]["search_region"] == [10, 20, 50, 60]
        assert "search_region" not in out["actions"][1]
    asyncio.run(_run())


def test_grounding_verifier_flags_low_confidence() -> None:
    async def _run():
        h = await _build(ModuleRole.GROUNDING_VERIFIER)
        payload = {
            "grounded_actions": [
                {"action": "tap_at", "ground_confidence": 0.2, "target_description": "x"},
                {"action": "tap_at", "ground_confidence": 0.9, "target_description": "y"},
                {"action": "tap_at"},  # no confidence → untouched
            ],
        }
        out = await h(payload)
        ga = out["grounded_actions"]
        assert ga[0].get("low_confidence") is True
        assert "low_confidence" not in ga[1]
        assert "low_confidence" not in ga[2]
    asyncio.run(_run())


def test_side_consumers_return_none() -> None:
    """Safety / Reflection / Ambiguity / Memory advance nothing → None."""
    async def _run():
        for role in (ModuleRole.SAFETY_GUARD, ModuleRole.REFLECTION,
                     ModuleRole.AMBIGUITY_RESOLVER, ModuleRole.MEMORY):
            h = await _build(role)
            out = await h({"run_id": "r", "goal_text": "g", "history": [],
                           "actions": []})
            assert out is None, f"{role.value} should be a side-consumer"
    asyncio.run(_run())


def test_screen_parser_handles_missing_screenshot() -> None:
    async def _run():
        h = await _build(ModuleRole.SCREEN_PARSER)
        out = await h({"run_id": "r"})  # no screenshot
        assert out["parsed_boxes"] == []
    asyncio.run(_run())


def test_dynamic_perceiver_first_screen_is_changed() -> None:
    async def _run():
        h = await _build(ModuleRole.DYNAMIC_PERCEIVER)
        out = await h({"run_id": "r"})  # no prior, no screenshot
        assert out["screen_changed"] is True
    asyncio.run(_run())
