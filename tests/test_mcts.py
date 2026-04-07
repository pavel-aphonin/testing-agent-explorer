"""Unit tests for the PUCT selector.

These verify the algorithmic invariants without touching a simulator:
  - registration is idempotent
  - priors normalize to 1
  - selection prefers high-prior, low-visit actions early on
  - selection shifts toward high-Q actions after backups
  - exclusion masks work
  - tie-breaking prefers less-visited actions
"""

import math

from explorer.mcts import PUCTSelector


def test_register_state_is_idempotent():
    sel = PUCTSelector(c_puct=2.0)
    sel.register_state("s1", ["a", "b", "c"])
    stats_first = sel.stats_for("s1")
    sel.register_state("s1", ["a", "b", "c"])  # second call must not reset
    stats_second = sel.stats_for("s1")
    assert stats_first is stats_second


def test_priors_normalize_to_one():
    sel = PUCTSelector()
    sel.register_state("s1", ["a", "b", "c"], priors={"a": 5.0, "b": 5.0, "c": 5.0})
    ss = sel.stats_for("s1")
    assert ss is not None
    total = sum(action.prior for action in ss.actions.values())
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_uniform_priors_when_none_provided():
    sel = PUCTSelector()
    sel.register_state("s1", ["a", "b", "c", "d"])
    ss = sel.stats_for("s1")
    assert ss is not None
    for action in ss.actions.values():
        assert math.isclose(action.prior, 0.25, abs_tol=1e-9)


def test_missing_action_falls_back_to_uniform():
    sel = PUCTSelector()
    # Provide priors for only two of three actions
    sel.register_state("s1", ["a", "b", "c"], priors={"a": 0.5, "b": 0.5})
    ss = sel.stats_for("s1")
    assert ss is not None
    # c should get uniform 1/3 before normalization, then renormalize
    assert ss.actions["c"].prior > 0


def test_select_first_call_picks_highest_prior():
    sel = PUCTSelector(c_puct=2.0)
    sel.register_state("s1", ["a", "b", "c"], priors={"a": 0.7, "b": 0.2, "c": 0.1})
    # With zero visits everywhere, the score is dominated by the prior term
    chosen = sel.select("s1")
    assert chosen == "a"


def test_backup_increments_visits_and_value():
    sel = PUCTSelector()
    sel.register_state("s1", ["a", "b"])
    sel.backup("s1", "a", 1.0)
    sel.backup("s1", "a", 0.5)
    ss = sel.stats_for("s1")
    assert ss is not None
    assert ss.visit_count == 2
    assert ss.actions["a"].visit_count == 2
    assert math.isclose(ss.actions["a"].q_value, 0.75, abs_tol=1e-9)


def test_high_q_action_eventually_dominates():
    sel = PUCTSelector(c_puct=1.0)
    sel.register_state("s1", ["good", "bad"])
    # Good action consistently rewards 1.0; bad action rewards 0.0
    for _ in range(20):
        chosen = sel.select("s1")
        if chosen == "good":
            sel.backup("s1", "good", 1.0)
        else:
            sel.backup("s1", "bad", 0.0)
    ss = sel.stats_for("s1")
    assert ss is not None
    # The good action must have been picked the majority of the time
    assert ss.actions["good"].visit_count > ss.actions["bad"].visit_count


def test_exclude_mask_skips_actions():
    sel = PUCTSelector()
    sel.register_state("s1", ["safe", "danger"])
    chosen = sel.select("s1", exclude={"danger"})
    assert chosen == "safe"


def test_exclude_all_returns_none():
    sel = PUCTSelector()
    sel.register_state("s1", ["a", "b"])
    assert sel.select("s1", exclude={"a", "b"}) is None


def test_tie_break_prefers_less_visited():
    sel = PUCTSelector(c_puct=0.0)  # disable U term so ties only break on Q
    sel.register_state("s1", ["a", "b"])
    sel.backup("s1", "a", 1.0)
    sel.backup("s1", "a", 1.0)
    sel.backup("s1", "b", 1.0)  # both have q=1.0 but b has fewer visits
    chosen = sel.select("s1")
    # With equal Q values, the less-visited action wins
    assert chosen == "b"


def test_unknown_state_returns_none():
    sel = PUCTSelector()
    assert sel.select("never_seen") is None


def test_add_action_if_missing():
    sel = PUCTSelector()
    sel.register_state("s1", ["a", "b"])
    sel.add_action_if_missing("s1", "c")
    ss = sel.stats_for("s1")
    assert ss is not None
    assert "c" in ss.actions
    # Existing actions are not removed
    assert "a" in ss.actions
    assert "b" in ss.actions
