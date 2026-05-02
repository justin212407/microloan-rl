"""Tests for the standalone repayment and acceptance models."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import pytest

pytest.importorskip("omegaconf")
pytest.importorskip("scipy")

from env.repayment_model import RepaymentModel


@pytest.fixture
def model() -> Iterator[RepaymentModel]:
    """Create a repayment model instance for each test."""
    yield RepaymentModel()


@pytest.fixture
def client() -> dict[str, float | str]:
    """Provide a realistic synthetic client payload."""
    return {
        "monthly_income": 320.0,
        "risk_score": 0.72,
        "region": "urban",
        "credit_score": 0.78,
        "loan_amount_requested": 650.0,
        "loan_purpose": "business",
        "previous_default_rate": 0.08,
    }


def test_acceptance_prob_in_bounds(
    model: RepaymentModel,
    client: dict[str, float | str],
) -> None:
    probability = model.acceptance_prob(client, 0.18)
    assert 0.01 <= probability <= 0.99


def test_higher_rate_lowers_acceptance(
    model: RepaymentModel,
    client: dict[str, float | str],
) -> None:
    high_rate = model.acceptance_prob(client, 0.36)
    low_rate = model.acceptance_prob(client, 0.08)
    assert high_rate < low_rate


def test_higher_credit_score_raises_repayment(
    model: RepaymentModel,
    client: dict[str, float | str],
) -> None:
    lower_credit = deepcopy(client)
    higher_credit = deepcopy(client)
    lower_credit["credit_score"] = 0.25
    higher_credit["credit_score"] = 0.90
    assert model.repayment_prob(higher_credit, 0.18) > model.repayment_prob(lower_credit, 0.18)


def test_rural_lower_acceptance_than_urban(
    model: RepaymentModel,
    client: dict[str, float | str],
) -> None:
    urban = deepcopy(client)
    rural = deepcopy(client)
    urban["region"] = "urban"
    rural["region"] = "rural"
    assert model.acceptance_prob(rural, 0.18) < model.acceptance_prob(urban, 0.18)


def test_sample_acceptance_returns_bool(
    model: RepaymentModel,
    client: dict[str, float | str],
) -> None:
    assert isinstance(model.sample_acceptance(client, 0.18), bool)


def test_sample_repayment_returns_bool(
    model: RepaymentModel,
    client: dict[str, float | str],
) -> None:
    assert isinstance(model.sample_repayment(client, 0.18), bool)
