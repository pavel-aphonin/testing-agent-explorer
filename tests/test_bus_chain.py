"""PER-175 full integration: structural validation of the 13-module bus chain.

These tests assert the TOPOLOGY (ROLE_WIRING) is a coherent pipeline — no
runtime, no models. They are the guard that "all 13 modules are wired"
stays true as the chain evolves: every module present, every produced
message consumed by someone, exactly one terminal the worker awaits.
"""

from __future__ import annotations

from explorer.bus.envelope import MsgType
from explorer.bus.runner import ROLE_WIRING
from explorer.role_resolver import ALL_MODULE_ROLES, ModuleRole


def test_all_13_modules_have_a_runner() -> None:
    """Every ModuleRole has exactly one bus wiring entry — no module is
    left without a place in the pipeline (the whole point of PER-175)."""
    assert set(ROLE_WIRING.keys()) == set(ALL_MODULE_ROLES)
    assert len(ROLE_WIRING) == 13


def test_chain_starts_at_screen_captured() -> None:
    """Exactly one module consumes the worker's screen.captured (the
    pipeline entry point) — the Screen Parser."""
    entry = [r for r, w in ROLE_WIRING.items() if w.consumes is MsgType.SCREEN_CAPTURED]
    assert entry == [ModuleRole.SCREEN_PARSER]


def test_every_produced_message_is_consumed() -> None:
    """No orphan stages: every message a module produces is consumed by at
    least one other module (except the terminal the worker reads)."""
    produced = {w.produces for w in ROLE_WIRING.values() if w.produces is not None}
    consumed = {w.consumes for w in ROLE_WIRING.values()}
    # ground.verified is the terminal — produced but consumed by the WORKER,
    # not by another runner. Everything else must be consumed within the chain.
    orphans = produced - consumed - {MsgType.GROUND_VERIFIED}
    assert orphans == set(), f"orphan produced messages: {orphans}"


def test_terminal_is_ground_verified() -> None:
    """The chain ends at ground.verified (Grounding Verifier) — that's what
    the worker awaits. No runner consumes it."""
    assert any(w.produces is MsgType.GROUND_VERIFIED for w in ROLE_WIRING.values())
    consumers = [r for r, w in ROLE_WIRING.items() if w.consumes is MsgType.GROUND_VERIFIED]
    # only side-consumer Memory reads it (to record); the worker also reads it
    assert consumers == [ModuleRole.MEMORY]


def test_main_chain_is_unbroken_screen_to_ground() -> None:
    """The linear main chain must connect consume→produce link-by-link from
    screen.captured to ground.verified, in order. Asserted per-link (each
    stage's produce == next stage's consume) so a break names the exact gap.
    """
    # (role, consumes, produces) for each main-chain stage, in order.
    chain = [
        (ModuleRole.SCREEN_PARSER, MsgType.SCREEN_CAPTURED, MsgType.SCREEN_PARSED),
        (ModuleRole.DYNAMIC_PERCEIVER, MsgType.SCREEN_PARSED, MsgType.SCREEN_PERCEIVED),
        (ModuleRole.CONTEXT_IDENTIFIER, MsgType.SCREEN_PERCEIVED, MsgType.CONTEXT_CLASSIFIED),
        (ModuleRole.PLANNER, MsgType.CONTEXT_CLASSIFIED, MsgType.PLAN_PRODUCED),
        (ModuleRole.REWARD_CRITIC, MsgType.PLAN_PRODUCED, MsgType.PLAN_CRITIQUED),
        (ModuleRole.PLATFORM_ADAPTER, MsgType.PLAN_CRITIQUED, MsgType.ACTIONS_RESOLVED),
        (ModuleRole.SCREEN_SEEKER, MsgType.ACTIONS_RESOLVED, MsgType.GROUND_REFINED),
        (ModuleRole.GROUNDER, MsgType.GROUND_REFINED, MsgType.GROUND_PRODUCED),
        (ModuleRole.GROUNDING_VERIFIER, MsgType.GROUND_PRODUCED, MsgType.GROUND_VERIFIED),
    ]
    # 1. each declared stage matches ROLE_WIRING exactly
    for role, consumes, produces in chain:
        w = ROLE_WIRING[role]
        assert w.consumes is consumes, f"{role.value} consumes {w.consumes}, expected {consumes}"
        assert w.produces is produces, f"{role.value} produces {w.produces}, expected {produces}"
    # 2. the links actually connect (this stage's produce feeds the next's consume)
    for (_, _, produces), (_, nxt_consumes, _) in zip(chain, chain[1:]):
        assert produces is nxt_consumes, f"chain break: {produces} != {nxt_consumes}"


def test_side_consumers_do_not_advance_chain() -> None:
    """Safety, Reflection, Ambiguity, Memory read a link but produce
    nothing — they annotate/veto/record without advancing the pipeline."""
    side = {r for r, w in ROLE_WIRING.items() if w.produces is None}
    assert side == {
        ModuleRole.SAFETY_GUARD,
        ModuleRole.REFLECTION,
        ModuleRole.AMBIGUITY_RESOLVER,
        ModuleRole.MEMORY,
    }


def test_consumer_groups_are_unique() -> None:
    """Each module has its own consumer group, so all 13 receive their
    own copy of the stream they read (no accidental shared-group stealing)."""
    groups = [w.group for w in ROLE_WIRING.values()]
    assert len(groups) == len(set(groups))
