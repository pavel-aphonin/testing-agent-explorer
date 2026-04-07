"""Unit tests for the Go-Explore frontier archive."""

from explorer.go_explore import GoExploreArchive


def test_empty_archive_has_no_frontier():
    arc = GoExploreArchive()
    assert arc.has_frontier() is False
    assert arc.best_frontier() is None
    assert arc.size() == 0


def test_update_adds_record():
    arc = GoExploreArchive()
    arc.update("home", total_actions=5, explored_actions=2, visit_count=1)
    assert arc.size() == 1
    assert arc.frontier_size() == 1


def test_state_with_no_unexplored_is_not_frontier():
    arc = GoExploreArchive()
    arc.update("done", total_actions=3, explored_actions=3, visit_count=1)
    assert arc.size() == 1
    assert arc.frontier_size() == 0
    assert arc.best_frontier() is None


def test_best_frontier_picks_highest_unexplored_count():
    arc = GoExploreArchive(alpha=0.0)  # disable visit-count damping
    arc.update("home", total_actions=10, explored_actions=2, visit_count=1)  # 8 unexplored
    arc.update("settings", total_actions=5, explored_actions=2, visit_count=1)  # 3 unexplored
    best = arc.best_frontier()
    assert best is not None
    assert best.state_id == "home"


def test_alpha_dampens_overvisited_states():
    arc = GoExploreArchive(alpha=1.0)  # full damping
    # Same unexplored count, but 'rare' has been visited far less than 'busy'
    arc.update("busy", total_actions=10, explored_actions=5, visit_count=100)
    arc.update("rare", total_actions=10, explored_actions=5, visit_count=2)
    best = arc.best_frontier()
    assert best is not None
    assert best.state_id == "rare"


def test_exclude_current_skips_self():
    arc = GoExploreArchive()
    arc.update("home", total_actions=5, explored_actions=1, visit_count=1)
    arc.update("settings", total_actions=5, explored_actions=1, visit_count=1)
    best = arc.best_frontier(exclude_current="home")
    assert best is not None
    assert best.state_id == "settings"


def test_remove_drops_state():
    arc = GoExploreArchive()
    arc.update("home", total_actions=5, explored_actions=1, visit_count=1)
    arc.remove("home")
    assert arc.size() == 0
    assert arc.best_frontier() is None


def test_update_replaces_existing_record():
    arc = GoExploreArchive()
    arc.update("home", total_actions=5, explored_actions=1, visit_count=1)
    arc.update("home", total_actions=5, explored_actions=4, visit_count=10)
    assert arc.size() == 1
    snapshots = arc.snapshot()
    assert snapshots[0].explored_actions == 4
    assert snapshots[0].visit_count == 10


def test_unexplored_count_clamps_to_zero():
    arc = GoExploreArchive()
    # Pathological case: explored > total (shouldn't happen but should be safe)
    arc.update("weird", total_actions=3, explored_actions=10, visit_count=1)
    assert arc.frontier_size() == 0
