"""PUCT (Polynomial Upper Confidence Tree) selection for the explorer.

This is the AlphaZero selection rule, adapted for app exploration where
"states" are screen IDs and "actions" are interactive elements on a screen.

    score(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

Where:
    Q(s, a)  — exploitation: average reward from taking action a at state s.
               In our setting, reward = number of *new* screens discovered
               by the resulting transition (1 if it leads to a brand-new
               screen, 0 if it loops back to a known one).
    P(s, a)  — prior: probability that this action is worth trying. In MC
               mode this is uniform 1/n. In AI/Hybrid modes the LLM
               assigns it once when the screen is first encountered.
    N(s)     — total times state s has been visited.
    N(s, a)  — times action a has been picked from state s.
    c_puct   — exploration constant. Higher → more exploration.

Unlike full AlphaZero we do not perform tree-search rollouts inside the
selection step. The engine drives MCTS forward one real step at a time
against the simulator, and `backup()` updates Q and N after each
transition. This makes the algorithm a kind of "online MCTS" or "real
environment MCTS" — the simulator is the world model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ActionStats:
    """Statistics for a single (state, action) pair."""

    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 1.0  # set when the action is first registered

    @property
    def q_value(self) -> float:
        """Average value of this action so far. 0 if untried (FPU = 0)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


@dataclass
class StateStats:
    """All actions and their stats for one state."""

    visit_count: int = 0
    actions: dict[str, ActionStats] = field(default_factory=dict)


class PUCTSelector:
    """A single PUCTSelector instance lives for the duration of one exploration run.

    The engine calls:
        register_state(state_id, action_ids, priors=...)
            once per newly-encountered screen, to set up the action table
        select(state_id) -> action_id
            on every step, to pick what to try next
        backup(state_id, action_id, value)
            after each step, to update Q and N
    """

    def __init__(self, c_puct: float = 2.0):
        self.c_puct = c_puct
        self._states: dict[str, StateStats] = {}

    # --- Registration --------------------------------------------------------

    def register_state(
        self,
        state_id: str,
        action_ids: list[str],
        priors: dict[str, float] | None = None,
    ) -> None:
        """Initialize the action table for a newly seen state.

        If `priors` is None, every action gets a uniform prior of 1/n
        (equivalent to plain UCB1). If priors are provided, missing actions
        fall back to uniform — this is forgiving when an LLM call returned
        only a subset of the elements.
        """
        if state_id in self._states:
            return  # idempotent: already registered

        n = max(len(action_ids), 1)
        uniform = 1.0 / n
        ss = StateStats()
        for action_id in action_ids:
            p = uniform
            if priors is not None and action_id in priors:
                p = max(priors[action_id], 0.0)
            ss.actions[action_id] = ActionStats(prior=p)

        # Normalize priors so they sum to 1 (PUCT assumes a probability dist).
        total = sum(a.prior for a in ss.actions.values())
        if total > 0:
            for a in ss.actions.values():
                a.prior /= total
        else:
            # All priors zero → fall back to uniform.
            for a in ss.actions.values():
                a.prior = uniform

        self._states[state_id] = ss

    def add_action_if_missing(self, state_id: str, action_id: str) -> None:
        """Late-arriving action discovery (e.g., element appeared after a tap).

        Adds the action with a uniform prior derived from the current
        action set. Does not re-normalize existing priors aggressively;
        the small drift is acceptable for our purposes.
        """
        ss = self._states.get(state_id)
        if ss is None or action_id in ss.actions:
            return
        n = len(ss.actions) + 1
        new_prior = 1.0 / n
        ss.actions[action_id] = ActionStats(prior=new_prior)

    # --- Selection -----------------------------------------------------------

    def select(self, state_id: str, exclude: set[str] | None = None) -> str | None:
        """Pick the action with the highest PUCT score, breaking ties by lowest visit count.

        Returns None if the state is unknown or has no actions left to try.
        `exclude` lets the caller skip actions that are known to be unsafe
        right now (e.g., a "Logout" button on the auth screen).
        """
        ss = self._states.get(state_id)
        if ss is None or not ss.actions:
            return None

        excluded = exclude or set()
        sqrt_n = math.sqrt(max(ss.visit_count, 1))

        best_action: str | None = None
        best_score = float("-inf")
        best_visits = float("inf")

        for action_id, stats in ss.actions.items():
            if action_id in excluded:
                continue
            u = self.c_puct * stats.prior * sqrt_n / (1 + stats.visit_count)
            score = stats.q_value + u

            # Tie-breaker: prefer the less-visited action.
            if score > best_score or (
                score == best_score and stats.visit_count < best_visits
            ):
                best_score = score
                best_visits = stats.visit_count
                best_action = action_id

        return best_action

    # --- Backup --------------------------------------------------------------

    def backup(self, state_id: str, action_id: str, value: float) -> None:
        """Record the outcome of taking action_id from state_id.

        `value` is the reward signal: 1.0 if the transition led to a brand-new
        screen, 0.0 if it looped back to a known screen. The engine may also
        pass partial credit for "interesting" things like a modal opening.
        """
        ss = self._states.get(state_id)
        if ss is None or action_id not in ss.actions:
            return
        ss.visit_count += 1
        action_stats = ss.actions[action_id]
        action_stats.visit_count += 1
        action_stats.total_value += value

    # --- Inspection ----------------------------------------------------------

    def stats_for(self, state_id: str) -> StateStats | None:
        return self._states.get(state_id)

    def known_states(self) -> list[str]:
        return list(self._states.keys())
