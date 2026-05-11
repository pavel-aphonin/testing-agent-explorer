"""Exploration modes — three runtime paths the worker can pick from.

Each mode has ONE clear runtime path. The worker reads
``config["mode"]``, looks up the ModeConfig here, and routes to the
right runtime — there is no mode that aliases another. Adding a new
mode means adding a row to MODE_CONFIGS and a branch in
``worker.execute_one_run``; bugs in one mode can't regress others.

Mode summary:

| Mode    | Runtime path                        | c_puct | LLM priors      | LLM per step | Rollouts |
|---------|-------------------------------------|--------|-----------------|--------------|----------|
| MC      | ExplorationEngine (uniform priors)  | 1.4    | no (uniform)    | no           | depth 10 |
| HYBRID  | ExplorationEngine + LLMPriorProvider| 2.0    | yes (cached)    | no           | depth 5  |
| AI      | LLMExplorationLoop                  | 0.0    | yes (re-asked)  | yes          | depth 0  |

**AI** — the LLM is in the driver's seat: it sees fresh elements +
screenshot + history on every step and picks the next action. Highest
cost (one LLM call per step), best at handling open-ended modals,
form ambiguity, recovering from app errors.

**HYBRID** — the LLM is consulted exactly once per newly-discovered
screen to assign priors over the visible elements. PUCT (the same
selector that drives MC) then makes all decisions for that screen,
informed by those priors. Far fewer LLM calls than AI (cost scales
with unique screens, not steps); cached priors mean revisits reuse
the same evaluation. Trade-off: PUCT can only pick from the actions
visible on the current screen, so HYBRID is weaker than AI at
deciding "I should go back and try the other tab".

**MC** — no LLM at all. PUCT runs with uniform priors and Monte-Carlo
rollouts evaluate each new screen. Cheapest, most reproducible,
useful as a baseline and for environments where the LLM is
unavailable.
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

    # If True, attach the current screenshot to LLM calls so vision-
    # capable models (Gemma 4 / Qwen 3.5) can see the actual UI in
    # addition to the textual a11y tree. Default OFF so the legacy
    # "text-only" behaviour holds for anyone with a non-vision model.
    # Set per-mode below; opt-in via TA_LLM_VISION env var as the
    # global override (truthy → force-on, falsy → force-off).
    vision_enabled: bool = False


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
        vision_enabled=True,  # AI mode: pay the latency, gain visual bug detection
    ),
    ExplorationMode.HYBRID: ModeConfig(
        mode=ExplorationMode.HYBRID,
        c_puct=2.0,
        use_llm_priors=True,
        llm_priors_cache=True,  # one LLM call per new screen, then reuse
        llm_per_step=False,     # PUCT picks; LLM only sets priors at first visit
        rollout_depth=5,
        llm_screen_naming=True,
        vision_enabled=True,    # screenshot fed alongside elements on first visit
    ),
}


def get_mode_config(mode: ExplorationMode | str) -> ModeConfig:
    """Look up the config for a mode by enum or string name."""
    if isinstance(mode, str):
        mode = ExplorationMode(mode)
    return MODE_CONFIGS[mode]
