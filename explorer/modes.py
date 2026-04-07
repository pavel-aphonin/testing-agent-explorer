"""Three exploration modes — all parameterizations of the same engine.

The engine itself doesn't branch on mode. It reads a ModeConfig and
behaves accordingly. This means new modes can be added without touching
the engine, and bugs in one mode can't accidentally regress the others.

Mode summary:

| Mode    | c_puct | LLM priors      | LLM per step | MC rollouts |
|---------|--------|-----------------|--------------|-------------|
| MC      | 1.4    | no (uniform)    | no           | depth 10    |
| AI      | 0.0    | yes (re-asked)  | yes          | depth 0     |
| Hybrid  | 2.0    | yes (cached)    | no           | depth 5     |

In Hybrid mode the LLM is consulted exactly once per newly-discovered
screen — to assign priors over the visible elements. After that, PUCT
makes all the decisions for that screen for the rest of the run.

In AI mode the LLM is in the driver's seat: it sees fresh elements and
makes a fresh decision on every step, with PUCT used only as a fallback
when the LLM's chosen action is unavailable.

In MC mode the LLM is never called.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExplorationMode(StrEnum):
    MC = "mc"
    AI = "ai"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class ModeConfig:
    """All knobs that change between exploration modes.

    Defaults here describe Hybrid mode. The MODE_CONFIGS dict at the
    bottom of this file overrides them for MC and AI.
    """

    mode: ExplorationMode

    # PUCT exploration constant. Higher = more exploration.
    c_puct: float = 2.0

    # If True, ask the LLM to assign priors over the elements of a new screen.
    # If False, all elements get a uniform prior.
    use_llm_priors: bool = True

    # If True, cache LLM priors per screen_id and never ask again for that screen.
    # If False, ask every time we visit (more expensive, more current).
    llm_priors_cache: bool = True

    # If True, ask the LLM to pick the action on every single step.
    # If False, PUCT picks based on Q + prior + visit count.
    llm_per_step: bool = False

    # How many random rollout steps to take from the new state to estimate
    # its value. 0 disables rollouts (LLM provides value directly in AI mode).
    rollout_depth: int = 5

    # Maximum LLM token budget per step. Hard cap to prevent runaway costs.
    llm_max_tokens_per_call: int = 256

    # If True, also ask the LLM for a semantic name for new screens
    # ("LoginForm", "ProductDetail"). This is one extra LLM call per new
    # screen but makes the resulting graph much more readable.
    llm_screen_naming: bool = True


MODE_CONFIGS: dict[ExplorationMode, ModeConfig] = {
    ExplorationMode.MC: ModeConfig(
        mode=ExplorationMode.MC,
        c_puct=1.4,  # standard UCB1 constant from Auer et al
        use_llm_priors=False,
        llm_priors_cache=True,
        llm_per_step=False,
        rollout_depth=10,
        llm_screen_naming=False,
    ),
    ExplorationMode.AI: ModeConfig(
        mode=ExplorationMode.AI,
        c_puct=0.0,  # greedy on Q + LLM prior; LLM is the policy
        use_llm_priors=True,
        llm_priors_cache=False,  # re-ask each visit (more expensive, more current)
        llm_per_step=True,
        rollout_depth=0,  # LLM provides value through reasoning, not rollouts
        llm_screen_naming=True,
    ),
    ExplorationMode.HYBRID: ModeConfig(
        mode=ExplorationMode.HYBRID,
        c_puct=2.0,
        use_llm_priors=True,
        llm_priors_cache=True,  # one LLM call per new screen, then never again
        llm_per_step=False,  # PUCT picks; LLM only sets priors at first sight
        rollout_depth=5,
        llm_screen_naming=True,
    ),
}


def get_mode_config(mode: ExplorationMode | str) -> ModeConfig:
    """Look up the config for a mode by enum or string name."""
    if isinstance(mode, str):
        mode = ExplorationMode(mode)
    return MODE_CONFIGS[mode]
