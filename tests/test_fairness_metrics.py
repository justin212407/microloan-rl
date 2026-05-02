"""Tests for fairness metric computation."""

from __future__ import annotations

import pandas as pd

from fairness.metrics import compute_fairness_report


def test_perfect_fairness_score() -> None:
    frame = pd.DataFrame(
        [
            _row(0, "GroupA", "low_risk", 0.12, True, True, 10.0),
            _row(1, "GroupB", "low_risk", 0.12, True, True, 10.0),
            _row(2, "GroupC", "low_risk", 0.12, True, True, 10.0),
            _row(3, "GroupA", "high_risk", 0.28, True, True, 8.0),
            _row(4, "GroupB", "high_risk", 0.28, True, True, 8.0),
            _row(5, "GroupC", "high_risk", 0.28, True, True, 8.0),
        ]
    )
    report = compute_fairness_report(frame)
    assert report.fairness_score > 0.95


def test_biased_agent_low_score() -> None:
    frame = pd.DataFrame(
        [
            _row(0, "GroupA", "low_risk", 0.12, True, True, 10.0),
            _row(1, "GroupB", "low_risk", 0.12, True, True, 10.0),
            _row(2, "GroupC", "low_risk", 0.36, True, True, 18.0),
            _row(3, "GroupA", "medium_risk", 0.20, True, True, 9.0),
            _row(4, "GroupB", "medium_risk", 0.20, True, True, 9.0),
            _row(5, "GroupC", "medium_risk", 0.36, True, True, 18.0),
        ]
    )
    report = compute_fairness_report(frame)
    assert report.fairness_score < 0.30


def test_controlled_rdg_lower_than_raw_when_risk_controlled() -> None:
    frame = pd.DataFrame(
        [
            _row(0, "GroupA", "high_risk", 0.30, True, True, 7.0),
            _row(1, "GroupB", "high_risk", 0.30, True, True, 7.0),
            _row(2, "GroupB", "low_risk", 0.10, True, True, 11.0),
            _row(3, "GroupC", "low_risk", 0.10, True, True, 11.0),
        ]
    )
    report = compute_fairness_report(frame)
    assert report.controlled_rate_disparity_gap < report.raw_rate_disparity_gap


def test_missing_risk_segment_column_handled() -> None:
    frame = pd.DataFrame(
        [
            _row(0, "GroupA", "low_risk", 0.12, True, True, 10.0),
            _row(1, "GroupB", "low_risk", 0.12, True, True, 10.0),
        ]
    ).rename(columns={"risk_segment": "client_segment"})
    assert "risk_segment" not in frame.columns
    report = compute_fairness_report(frame)
    assert "low_risk" in report.mean_rate_by_group_segment.columns
    assert report.raw_rate_disparity_gap == 0.0


def test_repayment_rate_computed_on_accepted_only() -> None:
    frame = pd.DataFrame(
        [
            _row(0, "GroupA", "low_risk", 0.12, True, True, 10.0),
            _row(1, "GroupA", "low_risk", 0.12, False, False, 0.0),
            _row(2, "GroupB", "low_risk", 0.12, True, False, -5.0),
        ]
    )
    report = compute_fairness_report(frame)
    assert report.repayment_rate_by_group["GroupA"] == 1.0
    assert report.repayment_rate_by_group["GroupB"] == 0.0


def _row(
    episode_id: int,
    demographic_group: str,
    risk_segment: str,
    rate_offered: float,
    accepted: bool,
    repaid: bool,
    profit: float,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "demographic_group": demographic_group,
        "risk_segment": risk_segment,
        "rate_offered": rate_offered,
        "accepted": accepted,
        "repaid": repaid,
        "profit": profit,
        "reward": profit / 10.0,
    }
