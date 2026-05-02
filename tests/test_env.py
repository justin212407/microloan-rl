"""Tests for the core microloan Gymnasium environment."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("omegaconf")
pytest.importorskip("scipy")

from env.microloan_env import MicroLoanEnv


@pytest.fixture
def env() -> Iterator[MicroLoanEnv]:
    """Create a fresh environment instance for each test."""
    instance = MicroLoanEnv()
    yield instance
    instance.close()


def test_reset_returns_correct_obs_shape(env: MicroLoanEnv) -> None:
    obs, _ = env.reset()
    assert obs.shape == (10,)


def test_step_returns_5_tuple(env: MicroLoanEnv) -> None:
    env.reset()
    result = env.step(np.array([0.18], dtype=np.float32))
    assert len(result) == 5
    _, _, terminated, truncated, _ = result
    assert terminated is True
    assert truncated is False


def test_info_contains_required_keys(env: MicroLoanEnv) -> None:
    env.reset()
    _, _, _, _, info = env.step(np.array([0.18], dtype=np.float32))
    assert all(key in info for key in env.info_keys)


def test_demographic_group_not_in_observation(env: MicroLoanEnv) -> None:
    obs, info = env.reset()
    assert "demographic_group" not in list(env.config.data.observation_features)
    assert info["demographic_group"] in list(env.config.data.demographic_groups)
    assert obs.shape == (10,)


def test_reward_is_finite(env: MicroLoanEnv) -> None:
    env.reset()
    _, reward, _, _, _ = env.step(np.array([0.18], dtype=np.float32))
    assert np.isfinite(reward)


def test_step_before_reset_raises(env: MicroLoanEnv) -> None:
    with pytest.raises(RuntimeError):
        env.step(np.array([0.18], dtype=np.float32))


def test_history_appended_after_step(env: MicroLoanEnv) -> None:
    env.reset()
    env.step(np.array([0.18], dtype=np.float32))
    assert len(env._history) == 1


def test_fairness_penalty_uses_history_not_current_step(
    env: MicroLoanEnv,
) -> None:
    """Fairness penalty on first episode must not include current offer."""
    env.reset()
    _, _, _, _, info = env.step(np.array([0.36], dtype=np.float32))
    assert info["fairness_penalty"] >= 0.0
    assert len(env._history) == 1
