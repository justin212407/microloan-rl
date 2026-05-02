# Synthetic Data Specification

## Overview
Synthetic client data must be grounded in real microfinance literature.
Do NOT use purely random distributions. The numbers below are based on
MIX Market data, World Bank microfinance reports, and published MFI portfolios.

---

## Client Feature Schema

```python
@dataclass
class Client:
    # Identity
    client_id: str                    # UUID
    demographic_group: str            # 'GroupA', 'GroupB', 'GroupC'
    segment: str                      # 'low_risk', 'medium_risk', 'high_risk'
    
    # Financial Features
    monthly_income: float             # USD/month
    loan_amount_requested: float      # USD
    existing_debt_ratio: float        # existing debt / income, [0, 1]
    credit_score: float               # normalized [0, 1]
    savings_balance: float            # USD
    
    # Behavioral Features
    num_previous_loans: int           # 0–10
    previous_default_rate: float      # fraction of previous loans defaulted, [0, 1]
    
    # Contextual Features
    loan_purpose: str                 # 'business', 'agriculture', 'education', 'emergency'
    region: str                       # 'urban', 'semi_urban', 'rural'
    
    # Derived (computed from above)
    risk_score: float                 # composite [0,1], higher = less risky
```

---

## Feature Distributions by Risk Segment

All monetary values in USD. Distributions are approximate — use scipy.stats.

### Low Risk (40% of population)
```
monthly_income:          LogNormal(mean=350, sigma=80),  clipped [150, 800]
loan_amount_requested:   LogNormal(mean=800, sigma=200), clipped [200, 2000]
existing_debt_ratio:     Beta(a=2, b=8)                  → mean ~0.2
credit_score:            Beta(a=8, b=2)                  → mean ~0.8, skewed high
savings_balance:         LogNormal(mean=200, sigma=100), clipped [0, 1000]
num_previous_loans:      Poisson(lambda=4),              clipped [0, 10]
previous_default_rate:   Beta(a=1, b=12)                 → mean ~0.08
```

### Medium Risk (40% of population)
```
monthly_income:          LogNormal(mean=220, sigma=80),  clipped [80, 600]
loan_amount_requested:   LogNormal(mean=500, sigma=150), clipped [100, 1500]
existing_debt_ratio:     Beta(a=4, b=6)                  → mean ~0.4
credit_score:            Beta(a=5, b=5)                  → mean ~0.5, symmetric
savings_balance:         LogNormal(mean=80, sigma=60),   clipped [0, 500]
num_previous_loans:      Poisson(lambda=2),              clipped [0, 8]
previous_default_rate:   Beta(a=2, b=8)                  → mean ~0.2
```

### High Risk (20% of population)
```
monthly_income:          LogNormal(mean=130, sigma=60),  clipped [50, 400]
loan_amount_requested:   LogNormal(mean=300, sigma=100), clipped [100, 1000]
existing_debt_ratio:     Beta(a=6, b=4)                  → mean ~0.6
credit_score:            Beta(a=2, b=8)                  → mean ~0.2, skewed low
savings_balance:         LogNormal(mean=20, sigma=30),   clipped [0, 200]
num_previous_loans:      Poisson(lambda=1),              clipped [0, 5]
previous_default_rate:   Beta(a=4, b=6)                  → mean ~0.4
```

---

## Demographic Groups and Overlap

### CRITICAL DESIGN REQUIREMENT
Demographic groups must NOT perfectly correlate with risk segments.
This makes fairness non-trivial and more realistic.

```
GroupA (50% of population):
    low_risk:    55%
    medium_risk: 35%
    high_risk:   10%

GroupB (30% of population):
    low_risk:    35%
    medium_risk: 40%
    high_risk:   25%

GroupC (20% of population):
    low_risk:    30%
    medium_risk: 40%
    high_risk:   30%
```

This means: GroupA is slightly less risky on average, but there are plenty of
low-risk GroupC clients. A fair agent should give similar rates to similar-risk
clients regardless of group. A biased agent will overcharge GroupC/GroupB.

---

## Loan Purpose Distribution
```
'business':    45%   (most common in microfinance)
'agriculture': 25%
'education':   20%
'emergency':   10%
```

### Effect on repayment (for RepaymentModel):
- business:    multiplier 1.0  (baseline)
- agriculture: multiplier 0.85 (seasonal income risk)
- education:   multiplier 1.05 (higher motivation)
- emergency:   multiplier 0.75 (distress borrowing, higher default risk)

---

## Region Distribution
```
'rural':      40%
'semi_urban': 35%
'urban':      25%
```

### Effect on income:
- rural income is ~30% lower than urban for same risk segment
- rural acceptance probability is lower at same rate (higher rate sensitivity)

---

## Risk Score Computation

```python
def compute_risk_score(client: dict) -> float:
    """
    Composite risk score normalized to [0, 1].
    Higher score = LOWER risk = better creditworthiness.
    
    Weights derived from microfinance credit scoring literature.
    """
    score = (
        0.35 * client['credit_score'] +                            # strongest predictor
        0.20 * (1.0 - client['existing_debt_ratio']) +             # debt burden
        0.15 * min(client['monthly_income'] / 500.0, 1.0) +        # income (capped)
        0.15 * (1.0 - client['previous_default_rate']) +           # repayment history
        0.10 * min(client['savings_balance'] / 300.0, 1.0) +       # savings buffer
        0.05 * min(client['num_previous_loans'] / 5.0, 1.0)        # credit history length
    )
    return float(np.clip(score, 0.0, 1.0))
```

---

## Observation Vector (fed to RL agent)

The agent receives a numpy array of shape `(10,)` — all values normalized to [0,1] or standardized:

```python
OBSERVATION_FEATURES = [
    'monthly_income_normalized',       # / 800.0
    'loan_amount_normalized',          # / 2000.0
    'existing_debt_ratio',             # already [0,1]
    'credit_score',                    # already [0,1]
    'savings_balance_normalized',      # / 1000.0
    'num_previous_loans_normalized',   # / 10.0
    'previous_default_rate',           # already [0,1]
    'risk_score',                      # already [0,1]
    'loan_purpose_encoded',            # ordinal: business=0.8, education=0.9, agriculture=0.6, emergency=0.3
    'region_encoded',                  # ordinal: urban=0.8, semi_urban=0.6, rural=0.4
]
```

### IMPORTANT: The agent does NOT see `demographic_group`
The demographic_group is only used for:
- Computing the fairness penalty in the reward
- Logging to episode_info for fairness audit

This reflects a real-world constraint: the agent should price on creditworthiness
alone, but we audit whether its learned policy produces disparate outcomes.

---

## ClientPool Implementation Notes

```python
class ClientPool:
    def __init__(self, n_clients: int = 1000, seed: int = 42):
        """
        Pre-generate a pool of n_clients at init time.
        sample() draws from this pool with replacement.
        This ensures realistic population-level statistics are maintained.
        """
    
    def sample(self) -> Client:
        """Draw one client uniformly from the pool."""
    
    def sample_batch(self, n: int) -> pd.DataFrame:
        """Draw n clients, return as DataFrame with all features."""
    
    def get_population_stats(self) -> dict:
        """Summary statistics for the entire pool — used in dashboard."""
```