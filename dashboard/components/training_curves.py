"""Training-curve visualizations for PPO results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dashboard.utils import load_results

ROLLING_WINDOW = 50


def render_training_curves(results_dir: str | Path | None = None) -> None:
    """Render PPO reward and fairness curves from saved results."""
    st.caption(
        "Training curves show PPO episode logs only. "
        "Baselines are stateless and have no training trajectory."
    )
    frame = load_results("PPO", results_dir)
    if frame.empty:
        st.info("No PPO results found yet.")
        return
    st.pyplot(_training_figure(frame))


def _training_figure(frame: pd.DataFrame) -> plt.Figure:
    figure, axis_left = plt.subplots(figsize=(12, 5))
    axis_right = axis_left.twinx()
    rolling = frame["reward"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    axis_left.plot(frame["episode_id"], frame["reward"], alpha=0.35, label="Reward")
    axis_left.plot(frame["episode_id"], rolling, linewidth=2.0, label="Rolling Mean")
    axis_right.plot(
        frame["episode_id"],
        frame["fairness_penalty"],
        color="tab:red",
        alpha=0.75,
        label="Fairness Penalty",
    )
    axis_left.set_title("PPO Reward Trajectory")
    axis_left.set_xlabel("Episode")
    axis_left.set_ylabel("Reward")
    axis_right.set_ylabel("Fairness Penalty")
    _attach_legend(figure, axis_left, axis_right)
    figure.tight_layout()
    return figure


def _attach_legend(
    figure: plt.Figure,
    axis_left: plt.Axes,
    axis_right: plt.Axes,
) -> None:
    left_handles, left_labels = axis_left.get_legend_handles_labels()
    right_handles, right_labels = axis_right.get_legend_handles_labels()
    figure.legend(left_handles + right_handles, left_labels + right_labels, loc="upper center", ncol=3)
