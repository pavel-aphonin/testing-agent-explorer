"""Go-Explore frontier archive (Ecoffet et al, 2019).

The role of this archive in our PUCT loop:

    PUCT does the local exploration well — it picks the best action for
    the current screen. But it has no global memory of "screens I should
    come back to". Without that, the explorer can spend all its budget
    rummaging in one corner of the app while ignoring an obvious unopened
    Login button on the home screen.

    Go-Explore solves this by maintaining an archive of the most
    interesting *frontier* states — states with untried actions remaining,
    weighted by how rarely they have been visited so far. When the engine
    detects it has stalled (no new screens discovered for K steps), it
    asks the archive for the best frontier state and replays the known
    path to get back there before resuming PUCT exploration.

    score(s) = unexplored_action_count(s) * (1 / visit_count(s)^alpha)

The alpha controls how strongly we punish over-visited states. alpha = 0
means raw unexplored count; alpha = 1 means we strictly prefer rarely
visited states. The default 0.5 is a balance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrontierRecord:
    """A snapshot of one screen's frontier status.

    Updated each time the engine visits the screen, so visit_count and
    explored_actions stay in sync with the live PUCT statistics.
    """

    state_id: str
    total_actions: int
    explored_actions: int
    visit_count: int

    @property
    def unexplored_count(self) -> int:
        return max(self.total_actions - self.explored_actions, 0)


class GoExploreArchive:
    """An online frontier archive. Updated incrementally by the engine."""

    def __init__(self, alpha: float = 0.5):
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self._records: dict[str, FrontierRecord] = {}

    # --- Maintenance ---------------------------------------------------------

    def update(
        self,
        state_id: str,
        total_actions: int,
        explored_actions: int,
        visit_count: int,
    ) -> None:
        """Replace the snapshot for one state.

        Called after each engine step for the *current* screen, with the
        latest counts pulled from the PUCT selector.
        """
        self._records[state_id] = FrontierRecord(
            state_id=state_id,
            total_actions=total_actions,
            explored_actions=explored_actions,
            visit_count=max(visit_count, 1),
        )

    def remove(self, state_id: str) -> None:
        """Drop a state from the archive (e.g., when it has no untried actions left)."""
        self._records.pop(state_id, None)

    # --- Queries -------------------------------------------------------------

    def best_frontier(self, exclude_current: str | None = None) -> FrontierRecord | None:
        """Return the highest-scoring frontier state (excluding `exclude_current`).

        Returns None if there are no remaining frontier states. The engine
        should treat that as "exploration complete" — there is nothing left
        worth visiting.
        """
        best: FrontierRecord | None = None
        best_score = -1.0

        for record in self._records.values():
            if record.state_id == exclude_current:
                continue
            if record.unexplored_count == 0:
                continue
            score = record.unexplored_count * (1.0 / (record.visit_count**self.alpha))
            if score > best_score:
                best_score = score
                best = record

        return best

    def has_frontier(self) -> bool:
        return any(rec.unexplored_count > 0 for rec in self._records.values())

    def size(self) -> int:
        return len(self._records)

    def frontier_size(self) -> int:
        return sum(1 for rec in self._records.values() if rec.unexplored_count > 0)

    def snapshot(self) -> list[FrontierRecord]:
        """Return all records for inspection / serialization."""
        return list(self._records.values())
