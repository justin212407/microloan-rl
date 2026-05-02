"""Fairness audit report generation for evaluated pricing agents."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from agents.evaluate import EvaluationResult
from fairness.metrics import FairnessReport, compute_fairness_report

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentAudit:
    """Computed fairness metrics and labels for one evaluated agent.

    Attributes:
        label: User-facing label for the agent column.
        result: Aggregate evaluation result.
        report: Computed fairness metrics for the episode logs.
        footnote: Optional footnote marker for the audit table.
    """

    label: str
    result: EvaluationResult
    report: FairnessReport
    footnote: str = ""


def compare_agents(
    results: list[EvaluationResult],
    results_dir: str | Path = Path("results"),
) -> str:
    """Compare evaluated agents and save a fairness audit report.

    Args:
        results: Evaluation results for the agents to compare.
        results_dir: Directory where audit artifacts are written.

    Returns:
        Rendered fairness audit table as plain text.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    audits = [_build_agent_audit(result) for result in _ordered_results(results)]
    report_text = _format_audit(audits)
    _write_audit(report_text, results_dir)
    _maybe_plot_tradeoff_curve(results_dir)
    return report_text


def load_evaluation_results(
    results_dir: str | Path = Path("results"),
) -> list[EvaluationResult]:
    """Load saved evaluation CSVs into EvaluationResult objects."""
    paths = sorted(Path(results_dir).glob("*_results.csv"))
    return [_load_result(path) for path in paths if path.name != "sensitivity_analysis.csv"]


def main() -> None:
    """Run the fairness audit against saved evaluation result files."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    results = load_evaluation_results(args.results_dir)
    if not results:
        LOGGER.warning("No evaluation result CSVs found in %s.", args.results_dir)
        return
    report_text = compare_agents(results, args.results_dir)
    LOGGER.info("\n%s", report_text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def _ordered_results(results: list[EvaluationResult]) -> list[EvaluationResult]:
    order = {"RandomAgent": 0, "GreedyAgent": 1, "FixedRateAgent": 2, "PPO": 3}
    return sorted(results, key=lambda result: order.get(result.agent_name, 99))


def _build_agent_audit(result: EvaluationResult) -> AgentAudit:
    label = _agent_label(result.agent_name)
    footnote = "*" if result.agent_name == "GreedyAgent" else ""
    report = compute_fairness_report(result.episode_logs)
    return AgentAudit(label=label, result=result, report=report, footnote=footnote)


def _agent_label(agent_name: str) -> str:
    labels = {
        "RandomAgent": "Random Agent",
        "GreedyAgent": "Greedy Agent",
        "FixedRateAgent": "Fixed Rate Agent",
        "PPO": "PPO",
    }
    return labels.get(agent_name, agent_name)


def _format_audit(audits: list[AgentAudit]) -> str:
    episodes = len(audits[0].result.episode_logs) if audits else 0
    widths = [20] + [_column_width(audit) for audit in audits]
    lines = [
        "=== FAIRNESS AUDIT REPORT ===",
        f"Evaluating: {episodes} episodes per agent",
        "",
        _header_line(audits, widths),
        _divider_line(widths),
        _metric_line("Rate Disparity Gap", widths, audits, "raw_rate_disparity_gap"),
        _metric_line("Ctrl. Rate Gap", widths, audits, "controlled_rate_disparity_gap"),
        _metric_line("Acceptance Gap", widths, audits, "controlled_acceptance_rate_gap"),
        _profit_line(widths, audits),
        _metric_line("Fairness Score", widths, audits, "fairness_score"),
        "",
        "* Greedy always charges the maximum rate to everyone.",
        "  This is equal mistreatment, not equitable pricing.",
    ]
    return "\n".join(lines)


def _column_width(audit: AgentAudit) -> int:
    return max(len(f"{audit.label}{audit.footnote}"), 13)


def _header_line(audits: list[AgentAudit], widths: list[int]) -> str:
    cells = ["".ljust(widths[0])]
    for audit, width in zip(audits, widths[1:]):
        cells.append(f"{audit.label}{audit.footnote}".ljust(width))
    return " | ".join(cells)


def _divider_line(widths: list[int]) -> str:
    return "-+-".join("-" * width for width in widths)


def _metric_line(
    label: str,
    widths: list[int],
    audits: list[AgentAudit],
    field_name: str,
) -> str:
    cells = [label.ljust(widths[0])]
    for audit, width in zip(audits, widths[1:]):
        value = getattr(audit.report, field_name)
        cells.append(f"{value:.3f}".ljust(width))
    return " | ".join(cells)


def _profit_line(widths: list[int], audits: list[AgentAudit]) -> str:
    cells = ["Mean Portfolio Profit".ljust(widths[0])]
    for audit, width in zip(audits, widths[1:]):
        cells.append(f"${audit.result.mean_profit:.2f}".ljust(width))
    return " | ".join(cells)


def _write_audit(report_text: str, results_dir: Path) -> None:
    output_path = results_dir / "fairness_audit.txt"
    output_path.write_text(report_text + "\n", encoding="utf-8")
    LOGGER.info("Saved fairness audit to %s.", output_path)


def _maybe_plot_tradeoff_curve(results_dir: Path) -> None:
    input_path = results_dir / "sensitivity_analysis.csv"
    if not input_path.exists():
        return
    frame = pd.read_csv(input_path)
    _plot_tradeoff_curve(frame, results_dir / "fairness_tradeoff.png")


def _plot_tradeoff_curve(frame: pd.DataFrame, output_path: Path) -> None:
    figure, axis_left = plt.subplots(figsize=(8, 5))
    axis_right = axis_left.twinx()
    profit_line = axis_left.plot(
        frame["beta"],
        frame["mean_profit"],
        marker="o",
        color="tab:blue",
        label="Mean Profit",
    )[0]
    fairness_line = axis_right.plot(
        frame["beta"],
        frame["fairness_score"],
        marker="s",
        color="tab:green",
        label="Fairness Score",
    )[0]
    gap_line = axis_right.plot(
        frame["beta"],
        frame["controlled_rdg"],
        marker="^",
        color="tab:red",
        label="Controlled RDG",
    )[0]
    axis_left.set_xlabel("Beta")
    axis_left.set_ylabel("Mean Profit", color="tab:blue")
    axis_right.set_ylabel("Fairness Score / Controlled RDG")
    axis_left.set_title("Fairness Tradeoff Curve")
    figure.legend(handles=[profit_line, fairness_line, gap_line], loc="upper center", ncol=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    LOGGER.info("Saved fairness tradeoff plot to %s.", output_path)


def _load_result(path: Path) -> EvaluationResult:
    frame = pd.read_csv(path)
    frame = _normalize_result_frame(frame)
    agent_name = _agent_name_from_path(path.stem)
    return EvaluationResult(
        agent_name=agent_name,
        mean_reward=float(frame["reward"].mean()),
        mean_profit=float(frame["profit"].mean()),
        acceptance_rate=float(frame["accepted"].mean()),
        repayment_rate=_repayment_rate(frame),
        mean_rate_offered=float(frame["rate_offered"].mean()),
        episode_logs=frame,
    )


def _agent_name_from_path(stem: str) -> str:
    names = {
        "random_results": "RandomAgent",
        "greedy_results": "GreedyAgent",
        "fixed_rate_results": "FixedRateAgent",
        "ppo_results": "PPO",
    }
    return names.get(stem, stem)


def _repayment_rate(frame: pd.DataFrame) -> float:
    accepted = frame[frame["accepted"]]
    if accepted.empty:
        return 0.0
    return float(accepted["repaid"].mean())


def _normalize_result_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["accepted"] = normalized["accepted"].map(_as_bool)
    normalized["repaid"] = normalized["repaid"].map(_as_bool)
    return normalized


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


if __name__ == "__main__":
    main()
