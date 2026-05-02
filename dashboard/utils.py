"""Shared dashboard data-loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_RESULTS_DIR = Path("results")
AGENT_FILES = {
    "PPO": "ppo_results.csv",
    "Random Agent": "random_results.csv",
    "Greedy Agent": "greedy_results.csv",
    "Fixed Rate Agent": "fixed_rate_results.csv",
}
FILE_AGENTS = {filename: agent for agent, filename in AGENT_FILES.items()}
AGENT_ORDER = ["PPO", "Random Agent", "Greedy Agent", "Fixed Rate Agent"]


def get_results_dir(results_dir: str | Path | None = None) -> Path:
    """Resolve the results directory used by the dashboard."""
    if results_dir is None:
        return DEFAULT_RESULTS_DIR
    return Path(results_dir)


def list_available_agents(results_dir: str | Path | None = None) -> list[str]:
    """List agents that currently have saved result CSVs."""
    directory = get_results_dir(results_dir)
    available = [FILE_AGENTS[path.name] for path in directory.glob("*_results.csv") if path.name in FILE_AGENTS]
    return [agent for agent in AGENT_ORDER if agent in available]


def load_results(
    agent_name: str,
    results_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load saved per-episode results for a single agent."""
    path = result_path(agent_name, results_dir)
    if not path.exists():
        return pd.DataFrame()
    return _normalize_result_frame(pd.read_csv(path))


def load_fairness_audit(results_dir: str | Path | None = None) -> str:
    """Load the saved fairness audit text, if available."""
    path = get_results_dir(results_dir) / "fairness_audit.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def fairness_tradeoff_path(results_dir: str | Path | None = None) -> Path:
    """Return the path to the saved fairness tradeoff image."""
    return get_results_dir(results_dir) / "fairness_tradeoff.png"


def result_path(agent_name: str, results_dir: str | Path | None = None) -> Path:
    """Resolve the CSV path for a named agent."""
    directory = get_results_dir(results_dir)
    if agent_name not in AGENT_FILES:
        raise KeyError(f"Unknown agent '{agent_name}'. Known: {list(AGENT_FILES)}")
    return directory / AGENT_FILES[agent_name]


def _normalize_result_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in ("accepted", "repaid"):
        if column in normalized.columns:
            normalized[column] = normalized[column].map(_as_bool)
    return normalized


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)
