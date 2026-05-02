"""Policy heatmap rendering for saved evaluation logs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from dashboard.utils import load_results


def render_policy_heatmap(
    agent_name: str,
    results_dir: str | Path | None = None,
) -> None:
    """Render a mean-rate heatmap by risk segment and demographic group."""
    frame = load_results(agent_name, results_dir)
    if frame.empty:
        st.info(f"No saved results found for {agent_name}.")
        return
    st.pyplot(_heatmap_figure(frame, agent_name))


def _heatmap_figure(frame: pd.DataFrame, agent_name: str) -> plt.Figure:
    pivot = frame.pivot_table(
        values="rate_offered",
        index="demographic_group",
        columns="risk_segment",
        aggfunc="mean",
    )
    figure, axis = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=axis)
    axis.set_title(f"{agent_name} Mean Offered Rate")
    axis.set_xlabel("Risk Segment")
    axis.set_ylabel("Demographic Group")
    figure.tight_layout()
    return figure
