"""Evaluation helpers for pricing agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class PredictAgent(Protocol):
    """Protocol for policy objects compatible with evaluation."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Any:
        """Predict an action from an observation."""


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate evaluation outputs for a policy run.

    Attributes:
        agent_name: Class name of the evaluated policy.
        mean_reward: Mean scalar reward across evaluated episodes.
        mean_profit: Mean raw profit from the environment info dict.
        acceptance_rate: Fraction of episodes where the offer was accepted.
        repayment_rate: Fraction of accepted loans that were repaid.
        mean_rate_offered: Mean annual interest rate proposed by the agent.
        episode_logs: Per-episode logs used for downstream fairness analysis.
    """

    agent_name: str
    mean_reward: float
    mean_profit: float
    acceptance_rate: float
    repayment_rate: float
    mean_rate_offered: float
    episode_logs: pd.DataFrame


def evaluate_agent(agent: PredictAgent, env: Any, n_episodes: int) -> EvaluationResult:
    """Evaluate a pricing agent on a seeded environment.

    Args:
        agent: Policy object exposing a `predict(obs)`-compatible method.
        env: Environment instance implementing Gymnasium reset and step.
        n_episodes: Number of single-step episodes to evaluate.

    Returns:
        EvaluationResult with aggregate metrics and episode logs.
    """
    base_seed = _base_seed(env)
    records: list[dict[str, Any]] = []
    for episode_id in range(n_episodes):
        observation, _ = env.reset(seed=base_seed + episode_id)
        action = _predict_action(agent, observation)
        _, reward, _, _, info = env.step(action)
        records.append(_episode_record(episode_id, reward, info))
    frame = pd.DataFrame(records)
    LOGGER.info("Evaluated %s episodes for %s.", n_episodes, agent.__class__.__name__)
    return EvaluationResult(
        agent_name=agent.__class__.__name__,
        mean_reward=float(frame["reward"].mean()),
        mean_profit=float(frame["profit"].mean()),
        acceptance_rate=float(frame["accepted"].mean()),
        repayment_rate=_repayment_rate(frame),
        mean_rate_offered=float(frame["rate_offered"].mean()),
        episode_logs=frame,
    )


def _base_seed(env: Any) -> int:
    config = getattr(env, "config", None)
    if config is None:
        return 0
    return int(config.project.seed)


def _predict_action(agent: PredictAgent, observation: np.ndarray) -> np.ndarray:
    try:
        prediction = agent.predict(observation, deterministic=True)
    except TypeError:
        prediction = agent.predict(observation)
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    return np.asarray(prediction, dtype=np.float32)


def _episode_record(episode_id: int, reward: float, info: dict[str, Any]) -> dict[str, Any]:
    reward_comps = info["reward_components"]
    return {
        "episode_id": episode_id,
        "reward": float(reward),
        "profit": float(info["profit"]),
        "accepted": bool(info["accepted"]),
        "repaid": bool(info["repaid"]),
        "rate_offered": float(info["rate_offered"]),
        "demographic_group": str(info["demographic_group"]),
        "client_segment": str(info["client_segment"]),
        "risk_segment": str(info["client_segment"]),
        "fairness_penalty": float(info["fairness_penalty"]),
        "reward_profit": float(reward_comps["profit"]),
        "reward_default_penalty": float(reward_comps["default_penalty"]),
        "reward_total": float(reward_comps["total"]),
        "reward_alpha": float(reward_comps["alpha"]),
        "reward_beta": float(reward_comps["beta"]),
        "reward_components": info["reward_components"],
    }


def _repayment_rate(frame: pd.DataFrame) -> float:
    accepted = frame[frame["accepted"]]
    if accepted.empty:
        return 0.0
    return float(accepted["repaid"].mean())
