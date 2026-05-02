"""Baseline pricing agents for the microloan environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class RandomAgentConfig:
    """Configuration for the random baseline agent.

    Attributes:
        seed: Optional random seed for reproducible action sampling.
    """

    seed: int | None = None


@dataclass(frozen=True)
class FixedRateAgentConfig:
    """Configuration for the fixed-rate baseline agent.

    Attributes:
        rate: Flat annual interest rate offered to every client.
    """

    rate: float = 0.18


class RandomAgent:
    """Baseline agent that samples uniformly from the action space."""

    def __init__(
        self,
        action_space: spaces.Box,
        config: RandomAgentConfig | None = None,
    ) -> None:
        """Store action space metadata and initialize the sampler.

        Args:
            action_space: Continuous interest-rate action space.
            config: Optional random-agent configuration.
        """
        self.action_space = action_space
        self.config = config or RandomAgentConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Sample a random valid action.

        Args:
            observation: Current environment observation, unused by this baseline.
            deterministic: Unused flag kept for policy interface compatibility.

        Returns:
            Sampled action array matching the environment action shape.
        """
        del observation, deterministic
        sample = self._rng.uniform(self.action_space.low, self.action_space.high)
        return np.asarray(sample, dtype=np.float32)


class GreedyAgent:
    """Baseline agent that always offers the maximum allowed rate."""

    def __init__(self, action_space: spaces.Box) -> None:
        """Store the compatible action space.

        Args:
            action_space: Continuous interest-rate action space.
        """
        self.action_space = action_space

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Return the maximum allowed rate.

        Args:
            observation: Current environment observation, unused by this baseline.
            deterministic: Unused flag kept for policy interface compatibility.

        Returns:
            Action array fixed at the action-space upper bound.
        """
        del observation, deterministic
        return np.asarray(self.action_space.high, dtype=np.float32).copy()


class FixedRateAgent:
    """Baseline agent that offers one flat rate to every client."""

    def __init__(
        self,
        action_space: spaces.Box,
        config: FixedRateAgentConfig | None = None,
    ) -> None:
        """Store action space metadata and flat-rate configuration.

        Args:
            action_space: Continuous interest-rate action space.
            config: Optional fixed-rate configuration.
        """
        self.action_space = action_space
        self.config = config or FixedRateAgentConfig()

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Return the configured flat interest rate.

        Args:
            observation: Current environment observation, unused by this baseline.
            deterministic: Unused flag kept for policy interface compatibility.

        Returns:
            Action array fixed at the configured rate within action-space bounds.
        """
        del observation, deterministic
        low = float(self.action_space.low[0])
        high = float(self.action_space.high[0])
        rate = float(np.clip(self.config.rate, low, high))
        return np.asarray([rate], dtype=np.float32)
