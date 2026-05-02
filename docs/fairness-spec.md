# Fairness Specification

## Why Fairness is Central to This Project
Loan pricing affects people's livelihoods. An RL agent optimizing purely for profit
can learn to discriminate — charging higher rates to certain demographic groups
even when their creditworthiness is the same. This document specifies how we measure,
penalize, and report on fairness in the POC.

---

## Fairness Definitions Used

### 1. Demographic Parity in Pricing
**Definition**: The mean interest rate offered should be equal across demographic groups,
after controlling for risk.

**Metric**: Rate Disparity Gap (RDG)
```
RDG = max(mean_rate[g] for g in groups) - min(mean_rate[g] for g in groups)
```

A perfectly fair agent has `RDG = 0`. We consider `RDG < 0.02` (2 percentage points) acceptable.

**Risk-Controlled Version** (more rigorous):
```
For each risk segment s ∈ {low, medium, high}:
    RDG_s = max(mean_rate[g,s]) - min(mean_rate[g,s])

Controlled_RDG = mean(RDG_s for all s)
```
This is the primary fairness metric. A model can have low raw RDG but high Controlled_RDG
if it's gaming the metric by mixing risk tiers.

---

### 2. Equal Opportunity in Loan Access
**Definition**: Acceptance rates should be equal across demographic groups for clients
of the same risk segment.

**Metric**: Acceptance Rate Gap (ARG)
```
For each risk segment s:
    ARG_s = max(acceptance_rate[g,s]) - min(acceptance_rate[g,s])

Controlled_ARG = mean(ARG_s for all s)
```

A model that rarely approves GroupC clients in the medium-risk segment is being unfair,
even if it occasionally approves high-risk GroupC clients.

---

### 3. Repayment Rate Parity
**Definition**: Given that a loan is made, repayment rates should not vary dramatically by
demographic group (controlling for risk). A large gap suggests the model has miscalibrated
its risk assessment for certain groups.

**Metric**: Repayment Disparity (RD)
```
RD = std(repayment_rate[g] for g in groups)
```

---

## Implementation

### `fairness/metrics.py`

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FairnessReport:
    """Full fairness audit report for a set of episodes."""
    
    # Per-group metrics
    mean_rate_by_group: Dict[str, float]
    acceptance_rate_by_group: Dict[str, float]
    repayment_rate_by_group: Dict[str, float]
    
    # Per-group-per-segment metrics
    mean_rate_by_group_segment: pd.DataFrame       # index: group, columns: risk_segment
    acceptance_rate_by_group_segment: pd.DataFrame
    
    # Summary metrics
    raw_rate_disparity_gap: float                  # RDG (uncontrolled)
    controlled_rate_disparity_gap: float           # Controlled_RDG (primary metric)
    controlled_acceptance_rate_gap: float          # Controlled_ARG
    repayment_disparity: float                     # RD
    
    # Overall fairness score (composite, for dashboard display)
    fairness_score: float                          # [0,1], higher is fairer


def compute_fairness_report(episode_log: pd.DataFrame) -> FairnessReport:
    """
    Compute full fairness report from episode logs.
    
    Args:
        episode_log: DataFrame with columns:
            ['episode_id', 'demographic_group', 'risk_segment',
             'rate_offered', 'accepted', 'repaid', 'profit', 'reward']
    
    Returns:
        FairnessReport with all metrics computed.
    """
    groups = episode_log['demographic_group'].unique()
    segments = ['low_risk', 'medium_risk', 'high_risk']
    
    # Per-group aggregates
    mean_rate_by_group = (
        episode_log.groupby('demographic_group')['rate_offered'].mean().to_dict()
    )
    acceptance_rate_by_group = (
        episode_log.groupby('demographic_group')['accepted'].mean().to_dict()
    )
    repayment_rate_by_group = (
        episode_log[episode_log['accepted'] == True]
        .groupby('demographic_group')['repaid'].mean().to_dict()
    )
    
    # Per-group-per-segment
    pivot_rate = episode_log.pivot_table(
        values='rate_offered', index='demographic_group',
        columns='risk_segment', aggfunc='mean'
    )
    pivot_acceptance = episode_log.pivot_table(
        values='accepted', index='demographic_group',
        columns='risk_segment', aggfunc='mean'
    )
    
    # Summary metrics
    raw_rdg = max(mean_rate_by_group.values()) - min(mean_rate_by_group.values())
    
    segment_rdgs = []
    segment_args = []
    for seg in segments:
        if seg in pivot_rate.columns:
            rates = pivot_rate[seg].dropna()
            if len(rates) > 1:
                segment_rdgs.append(rates.max() - rates.min())
            acc = pivot_acceptance[seg].dropna()
            if len(acc) > 1:
                segment_args.append(acc.max() - acc.min())
    
    controlled_rdg = float(np.mean(segment_rdgs)) if segment_rdgs else 0.0
    controlled_arg = float(np.mean(segment_args)) if segment_args else 0.0
    rd = float(np.std(list(repayment_rate_by_group.values())))
    
    # Composite fairness score: lower gaps → higher score
    fairness_score = max(0.0, 1.0 - (controlled_rdg / 0.10) * 0.5 - (controlled_arg / 0.20) * 0.5)
    
    return FairnessReport(
        mean_rate_by_group=mean_rate_by_group,
        acceptance_rate_by_group=acceptance_rate_by_group,
        repayment_rate_by_group=repayment_rate_by_group,
        mean_rate_by_group_segment=pivot_rate,
        acceptance_rate_by_group_segment=pivot_acceptance,
        raw_rate_disparity_gap=raw_rdg,
        controlled_rate_disparity_gap=controlled_rdg,
        controlled_acceptance_rate_gap=controlled_arg,
        repayment_disparity=rd,
        fairness_score=fairness_score,
    )
```

---

## Fairness Audit Report (`fairness/audit.py`)
Generates a human-readable audit comparing agents:

```
=== FAIRNESS AUDIT REPORT ===
Evaluating: 1000 episodes per agent

                    | Random Agent | Greedy Agent | PPO (β=0.0) | PPO (β=0.3) |
--------------------|--------------|--------------|-------------|-------------|
Rate Disparity Gap  |    0.082     |    0.000*    |    0.061    |    0.021    |
Ctrl. Rate Gap      |    0.079     |    0.000*    |    0.058    |    0.018    |
Acceptance Gap      |    0.031     |    0.000*    |    0.042    |    0.015    |
Mean Portfolio Profit|   $12.10    |   $28.40     |   $24.80    |   $21.30    |
Fairness Score      |    0.41      |    1.00*     |    0.52     |    0.82     |

* Greedy always charges 36% to everyone — technically "fair" (equal mistreatment)
  but not a good policy. Note profit vs PPO.
```

This comparison table is the key narrative of the POC: PPO with fairness (β=0.3)
achieves high fairness WITHOUT sacrificing as much profit as you'd expect.

---

## Fairness Tradeoff Curve (Key Plot)

```python
beta_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# For each beta, train PPO and evaluate on 1000 episodes:
results = {
    beta: {
        'mean_profit': ...,
        'controlled_rdg': ...,
        'fairness_score': ...,
    }
    for beta in beta_values
}

# Plot: Fairness Score vs Mean Profit
# This Pareto frontier is the analytical centerpiece of the POC
```

---

## What NOT to Do
- Do not use raw demographic features in the fairness penalty as a shortcut
- Do not conflate "equal rates for all" with "fairness" — a flat 36% is equally unfair
- Do not compute fairness on <30 episodes per group — results will be noisy
- Do not report only raw RDG — always report Controlled_RDG alongside it