"""Fairness-focused Streamlit panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from dashboard.utils import fairness_tradeoff_path, list_available_agents, load_fairness_audit, load_results
from fairness.metrics import compute_fairness_report


def render_fairness_panel(
    agent_name: str,
    results_dir: str | Path | None = None,
) -> None:
    """Render fairness artifacts and per-group rate distributions."""
    audit_text = load_fairness_audit(results_dir)
    if audit_text:
        st.code(audit_text)
    tradeoff_path = fairness_tradeoff_path(results_dir)
    if tradeoff_path.exists():
        st.image(str(tradeoff_path), caption="Fairness Tradeoff Curve")
    frame = load_results(agent_name, results_dir)
    if frame.empty:
        st.info(f"No saved results found for {agent_name}.")
        return
    st.pyplot(_boxplot_figure(frame, agent_name))


def render_agent_comparison(results_dir: str | Path | None = None) -> None:
    """Render an aggregate comparison table across available agents."""
    rows = [_comparison_row(agent, results_dir) for agent in list_available_agents(results_dir)]
    rows = [row for row in rows if row is not None]
    if not rows:
        st.info("No saved agent results available for comparison.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    if any(row["agent_name"] == "Greedy Agent" for row in rows):
        st.caption("Greedy Agent is marked as equal mistreatment: identical pricing, not equitable pricing.")


def _boxplot_figure(frame: pd.DataFrame, agent_name: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=frame, x="demographic_group", y="rate_offered", ax=axis)
    axis.set_title(f"{agent_name} Rate Distribution by Demographic Group")
    axis.set_xlabel("Demographic Group")
    axis.set_ylabel("Rate Offered")
    figure.tight_layout()
    return figure


def _comparison_row(
    agent_name: str,
    results_dir: str | Path | None,
) -> dict[str, float | str] | None:
    frame = load_results(agent_name, results_dir)
    if frame.empty:
        return None
    report = compute_fairness_report(frame)
    return {
        "agent_name": agent_name,
        "mean_reward": float(frame["reward"].mean()),
        "mean_profit": float(frame["profit"].mean()),
        "acceptance_rate": float(frame["accepted"].mean()),
        "fairness_score": report.fairness_score,
        "controlled_rdg": report.controlled_rate_disparity_gap,
    }
