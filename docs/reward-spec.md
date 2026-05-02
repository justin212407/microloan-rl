# Reward Function Specification

## Overview
The reward function is the most critical design element of this project.
It must encode three competing objectives:
1. **Profitability**: The institution needs to be financially sustainable
2. **Default Risk Management**: Bad loans hurt the portfolio
3. **Fairness**: Pricing should not systematically disadvantage demographic groups

---

## Core Formula

```
reward = profit_component - α * default_penalty - β * fairness_penalty
```

Where `α` and `β` are loaded from `configs/default.yaml`.

---

## Component Definitions

### Profit Component
```python
def profit_component(accepted: bool, repaid: bool, loan_amount: float,
                     interest_rate: float, cost_of_funds: float = 0.05) -> float:
    """
    If not accepted: 0 (no loan made)
    If accepted and repaid: loan_amount * (interest_rate - cost_of_funds)
    If accepted and not repaid: 0 (principal assumed recovered at break-even for simplicity)
    
    cost_of_funds = 5% (fixed, represents MFI's borrowing cost)
    """
    if not accepted:
        return 0.0
    if repaid:
        return loan_amount * (interest_rate - cost_of_funds)
    return 0.0
```

### Default Penalty
```python
def default_penalty(accepted: bool, repaid: bool, loan_amount: float) -> float:
    """
    Penalty for defaults — incentivizes the agent to not offer high rates
    to clients who will default.
    
    If not accepted: 0
    If accepted and repaid: 0
    If accepted and not repaid: loan_amount * 0.5
        (partial loss — assume 50% recovery on default)
    """
    if not accepted or repaid:
        return 0.0
    return loan_amount * 0.5
```

### Fairness Penalty (Step-Level Approximation)
The true group-level fairness penalty requires comparing across clients, which is
tricky in single-step episodes. We use a proxy: deviation from the "fair" rate for
the client's risk segment.

```python
def fairness_penalty(offered_rate: float, client_risk_score: float,
                     demographic_group: str, group_mean_rates: dict) -> float:
    """
    Penalizes the agent when it offers rates that deviate from what clients
    of similar risk profile receive on average, stratified by demographic group.
    
    group_mean_rates: rolling dict of {demographic_group: mean_rate_offered}
                      maintained by the environment across the last 100 episodes.
    
    penalty = max(0, offered_rate - fair_rate_for_risk_tier)
    
    where fair_rate_for_risk_tier is computed from a risk-to-rate mapping:
        low_risk    (score > 0.7):  fair_rate = 0.10–0.15
        medium_risk (0.4–0.7):     fair_rate = 0.16–0.24
        high_risk   (score < 0.4): fair_rate = 0.25–0.32
    
    The penalty is the excess over the fair rate for the risk tier.
    This penalizes the agent for charging MORE than risk justifies,
    which is the core of price discrimination.
    """
    fair_rate = _fair_rate_for_risk(client_risk_score)
    excess = max(0.0, offered_rate - fair_rate)
    
    # Scale by how much the group is already being disadvantaged
    group_mean = group_mean_rates.get(demographic_group, offered_rate)
    overall_mean = sum(group_mean_rates.values()) / max(len(group_mean_rates), 1)
    group_disadvantage = max(0.0, group_mean - overall_mean)
    
    return excess + group_disadvantage


def _fair_rate_for_risk(risk_score: float) -> float:
    """Maps a risk score [0,1] to a fair interest rate."""
    # Linear interpolation: higher risk → higher rate, but bounded
    MIN_RATE, MAX_RATE = 0.10, 0.32
    return MIN_RATE + (1.0 - risk_score) * (MAX_RATE - MIN_RATE)
```

---

## Full Reward Calculation

```python
@dataclass
class RewardComponents:
    profit: float
    default_penalty: float
    fairness_penalty: float
    total: float
    alpha: float
    beta: float


def compute_reward(
    accepted: bool,
    repaid: bool,
    loan_amount: float,
    interest_rate: float,
    client_risk_score: float,
    demographic_group: str,
    group_mean_rates: dict,
    alpha: float,
    beta: float,
) -> RewardComponents:
    p = profit_component(accepted, repaid, loan_amount, interest_rate)
    d = default_penalty(accepted, repaid, loan_amount)
    f = fairness_penalty(interest_rate, client_risk_score, demographic_group, group_mean_rates)
    
    total = p - alpha * d - beta * f
    
    # Normalize to keep rewards in a reasonable range for PPO stability
    # Divide by a reference loan amount (e.g., $500)
    total_normalized = total / 500.0
    
    return RewardComponents(
        profit=p,
        default_penalty=d,
        fairness_penalty=f,
        total=total_normalized,
        alpha=alpha,
        beta=beta,
    )
```

---

## Default Hyperparameters

```yaml
# configs/default.yaml (reward section)
reward:
  alpha: 0.8        # default penalty weight — high, we strongly penalize defaults
  beta: 0.3         # fairness weight — moderate, fairness matters but doesn't dominate
  cost_of_funds: 0.05
  reference_loan_amount: 500.0
```

---

## Sensitivity Analysis — The Tradeoff Curve
Run training with `β ∈ [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]` and plot:
- X-axis: β value
- Y-axis (left): mean portfolio profit per episode
- Y-axis (right): demographic parity gap (max group rate - min group rate)

This curve is the key analytical contribution of the POC. It shows mentors
you understand that fairness is a constraint with a real cost, and you've
quantified that cost empirically.

---

## What This Reward Function Teaches the Agent
- Offering high rates to low-risk clients → penalized via fairness_penalty
- Offering loans to high-risk clients at low rates → penalized via default_penalty
- Not offering loans (no acceptance) → 0 reward, which the agent learns to avoid
- Sweet spot: moderate rates calibrated to risk, consistent across demographic groups