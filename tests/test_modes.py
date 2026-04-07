"""Unit tests for mode configuration."""

from explorer.modes import (
    MODE_CONFIGS,
    ExplorationMode,
    ModeConfig,
    get_mode_config,
)


def test_all_modes_registered():
    for mode in ExplorationMode:
        assert mode in MODE_CONFIGS


def test_get_by_string():
    cfg = get_mode_config("hybrid")
    assert isinstance(cfg, ModeConfig)
    assert cfg.mode == ExplorationMode.HYBRID


def test_get_by_enum():
    cfg = get_mode_config(ExplorationMode.MC)
    assert cfg.mode == ExplorationMode.MC


def test_mc_mode_has_no_llm():
    cfg = get_mode_config(ExplorationMode.MC)
    assert cfg.use_llm_priors is False
    assert cfg.llm_per_step is False
    assert cfg.rollout_depth > 0  # MC must do rollouts to get value signal


def test_ai_mode_uses_llm_per_step():
    cfg = get_mode_config(ExplorationMode.AI)
    assert cfg.use_llm_priors is True
    assert cfg.llm_per_step is True
    assert cfg.rollout_depth == 0  # LLM provides value, no need for rollouts


def test_hybrid_mode_uses_cached_priors():
    cfg = get_mode_config(ExplorationMode.HYBRID)
    assert cfg.use_llm_priors is True
    assert cfg.llm_priors_cache is True  # cached, not re-asked
    assert cfg.llm_per_step is False  # PUCT picks, LLM only seeds priors
    assert cfg.rollout_depth > 0


def test_invalid_mode_raises():
    try:
        get_mode_config("nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")
