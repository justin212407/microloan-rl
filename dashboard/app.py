"""Streamlit entry point for the microloan RL dashboard."""

from __future__ import annotations

import streamlit as st

from dashboard.components.fairness_panel import render_agent_comparison, render_fairness_panel
from dashboard.components.policy_heatmap import render_policy_heatmap
from dashboard.components.training_curves import render_training_curves
from dashboard.utils import get_results_dir, list_available_agents

st.set_page_config(page_title="DMP Micro-loan RL Dashboard", layout="wide")


def main() -> None:
    """Render the dashboard shell and delegate tab content to components."""
    results_dir = get_results_dir()
    agents = list_available_agents(results_dir)
    st.title("Dynamic Pricing of Micro-loans via RL")
    if not agents:
        st.warning("No saved result CSVs found in the results directory.")
        return
    agent_name = st.sidebar.selectbox("Agent", agents, index=0)
    tabs = st.tabs(["Training Curves", "Policy Heatmap", "Fairness Panel", "Agent Comparison"])
    with tabs[0]:
        render_training_curves(results_dir)
    with tabs[1]:
        render_policy_heatmap(agent_name, results_dir)
    with tabs[2]:
        render_fairness_panel(agent_name, results_dir)
    with tabs[3]:
        render_agent_comparison(results_dir)


if __name__ == "__main__":
    main()
