"""Fairness metrics for microloan pricing episode logs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RISK_SEGMENTS = ("low_risk", "medium_risk", "high_risk")


@dataclass(frozen=True)
class FairnessReport:
    """Full fairness audit report for a set of pricing episodes.

    Attributes:
        mean_rate_by_group: Mean offered rate by demographic group.
        acceptance_rate_by_group: Mean acceptance rate by demographic group.
        repayment_rate_by_group: Mean repayment rate by demographic group.
        mean_rate_by_group_segment: Mean rate pivoted by group and risk segment.
        acceptance_rate_by_group_segment: Acceptance pivoted by group and risk segment.
        raw_rate_disparity_gap: Uncontrolled gap across group mean rates.
        controlled_rate_disparity_gap: Mean per-segment rate disparity gap.
        controlled_acceptance_rate_gap: Mean per-segment acceptance gap.
        repayment_disparity: Standard deviation of repayment rates by group.
        fairness_score: Composite fairness score in [0, 1].
    """

    mean_rate_by_group: dict[str, float]
    acceptance_rate_by_group: dict[str, float]
    repayment_rate_by_group: dict[str, float]
    mean_rate_by_group_segment: pd.DataFrame
    acceptance_rate_by_group_segment: pd.DataFrame
    raw_rate_disparity_gap: float
    controlled_rate_disparity_gap: float
    controlled_acceptance_rate_gap: float
    repayment_disparity: float
    fairness_score: float


def compute_fairness_report(episode_log: pd.DataFrame) -> FairnessReport:
    """Compute all fairness metrics required by the project spec.

    Args:
        episode_log: Episode-level evaluation log with fairness columns.

    Returns:
        FairnessReport containing per-group and summary metrics.
    """
    if "risk_segment" not in episode_log.columns:
        episode_log = episode_log.copy()
        episode_log["risk_segment"] = episode_log["client_segment"]
    mean_rate = _group_mean(episode_log, "rate_offered")
    acceptance = _group_mean(episode_log, "accepted")
    repayment = _repayment_rate_by_group(episode_log)
    pivot_rate = _segment_pivot(episode_log, "rate_offered")
    pivot_acceptance = _segment_pivot(episode_log, "accepted")
    raw_rdg = _series_gap(mean_rate)
    controlled_rdg = _controlled_gap(pivot_rate)
    controlled_arg = _controlled_gap(pivot_acceptance)
    repayment_disparity = _repayment_disparity(repayment)
    fairness_score = _fairness_score(
        controlled_rdg,
        controlled_arg,
        repayment_disparity,
    )
    return FairnessReport(
        mean_rate_by_group=mean_rate,
        acceptance_rate_by_group=acceptance,
        repayment_rate_by_group=repayment,
        mean_rate_by_group_segment=pivot_rate,
        acceptance_rate_by_group_segment=pivot_acceptance,
        raw_rate_disparity_gap=raw_rdg,
        controlled_rate_disparity_gap=controlled_rdg,
        controlled_acceptance_rate_gap=controlled_arg,
        repayment_disparity=repayment_disparity,
        fairness_score=fairness_score,
    )


def _group_mean(frame: pd.DataFrame, column: str) -> dict[str, float]:
    series = frame.groupby("demographic_group")[column].mean()
    return {str(group): float(value) for group, value in series.items()}


def _repayment_rate_by_group(frame: pd.DataFrame) -> dict[str, float]:
    accepted = frame[frame["accepted"]]
    if accepted.empty:
        return {}
    series = accepted.groupby("demographic_group")["repaid"].mean()
    return {str(group): float(value) for group, value in series.items()}


def _segment_pivot(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    pivot = frame.pivot_table(
        values=column,
        index="demographic_group",
        columns="risk_segment",
        aggfunc="mean",
    )
    return pivot.reindex(columns=list(RISK_SEGMENTS))


def _series_gap(values: dict[str, float]) -> float:
    if len(values) < 2:
        return 0.0
    data = list(values.values())
    return float(max(data) - min(data))


def _controlled_gap(pivot: pd.DataFrame) -> float:
    gaps = [_column_gap(pivot[column]) for column in pivot.columns]
    valid_gaps = [gap for gap in gaps if gap is not None]
    if not valid_gaps:
        return 0.0
    return float(np.mean(valid_gaps))


def _column_gap(series: pd.Series) -> float | None:
    values = series.dropna()
    if len(values) < 2:
        return None
    return float(values.max() - values.min())


def _repayment_disparity(values: dict[str, float]) -> float:
    if not values:
        return 0.0
    return float(np.std(list(values.values())))


def _fairness_score(
    controlled_rdg: float,
    controlled_arg: float,
    repayment_disparity: float,
) -> float:
    score = 1.0
    score -= (controlled_rdg / 0.10) * 0.5
    score -= (controlled_arg / 0.10) * 0.3
    score -= (repayment_disparity / 0.05) * 0.2
    return float(max(0.0, score))
