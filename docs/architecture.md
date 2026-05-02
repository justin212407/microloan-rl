# Architecture — DMP Micro-loan RL POC

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                            │
│                                                                 │
│  ┌──────────────┐    action     ┌──────────────────────────┐   │
│  │  PPO Agent   │ ────────────► │   MicroLoanEnv (Gym)     │   │
│  │ (SB3)        │ ◄──────────── │                          │   │
│  └──────────────┘  obs, reward  │  ┌────────────────────┐  │   │
│                                 │  │   ClientPool       │  │   │
│  ┌──────────────┐               │  │  (synthetic pop.)  │  │   │
│  │  Baselines   │               │  └────────────────────┘  │   │
│  │  - Random    │               │  ┌────────────────────┐  │   │
│  │  - Greedy    │               │  │  RepaymentModel    │  │   │
│  └──────────────┘               │  │  (logistic)        │  │   │
│                                 │  └────────────────────┘  │   │
│                                 │  ┌────────────────────┐  │   │
│                                 │  │  RewardFunction    │  │   │
│                                 │  │  (profit+fairness) │  │   │
│                                 │  └────────────────────┘  │   │
│                                 └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                        episode_log (DataFrame)
                                │
              ┌─────────────────┴──────────────────┐
              ▼                                     ▼
   ┌─────────────────────┐             ┌────────────────────────┐
   │   Fairness Module   │             │   Streamlit Dashboard  │
   │                     │             │                        │
   │  - DemographicParity│             │  - Training curves     │
   │  - EqualOpportunity │             │  - Policy heatmap      │
   │  - FairnessReport   │             │  - Fairness panel      │
   └─────────────────────┘             │  - Agent comparison    │
                                       └────────────────────────┘
```

---

## Component Breakdown

### 1. `env/microloan_env.py` — The Core
The heart of the entire POC. A single-step `gymnasium.Env` where:
- **State** = one sampled client's feature vector
- **Action** = interest rate to offer (continuous, 8%–36%)
- **Reward** = profit adjusted for default risk and fairness penalty
- **Episode** = one loan decision (terminates after 1 step)

Why single-step? It makes the RL problem tractable for a POC while still demonstrating the
core pricing decision. Multi-step (portfolio-level) MDP is noted as a future extension.

### 2. `env/client_pool.py` — Synthetic Population
Generates a realistic synthetic population of ~1000 microfinance clients with features
drawn from distributions grounded in real microfinance data (see DATA_SPEC.md).

Key design: demographic groups intentionally overlap with risk tiers to make fairness
a non-trivial problem. The agent must learn to price on creditworthiness, not group membership.

### 3. `env/repayment_model.py` — Behavioral Simulator
Two models:
- **Acceptance model**: `P(accept) = sigmoid(w_0 + w_1*income + w_2*rate + w_3*risk_score)`
- **Repayment model**: `P(repay) = sigmoid(w_0 + w_1*credit_score + w_2*income + w_3*rate + w_4*loan_purpose)`

These are fixed (not learned) — they represent the ground truth of client behavior in this simulation.

### 4. `agents/` — RL and Baselines
- **PPO** (Proximal Policy Optimization) via Stable-Baselines3 — chosen for stability and
  ease of use with continuous action spaces
- **Random baseline**: samples uniformly from action space
- **Greedy baseline**: always prices at maximum rate (36%) — pure profit, no fairness
- **Fixed-rate baseline**: charges a flat 18% to everyone — simple fairness, poor profit

### 5. `fairness/` — Audit Layer
Reads episode logs, computes fairness metrics, generates a report. Completely decoupled
from training — can be run on any agent's episode log.

### 6. `dashboard/` — Streamlit App
Loads pre-computed results and displays them interactively. Does not re-train.
Allows toggling between agent types and adjusting β for sensitivity analysis.

---

## Data Flow

```
ClientPool.sample()
    → client_features (dict)
        → MicroLoanEnv.reset() returns observation (np.ndarray)
            → Agent produces action (float: interest_rate)
                → RepaymentModel.acceptance_prob(client, rate) → Bernoulli sample
                    → if accepted: RepaymentModel.repayment_prob(client, rate) → Bernoulli sample
                        → RewardFunction.compute(profit, default, fairness_penalty)
                            → step() returns (next_obs, reward, terminated, truncated, info)
                                → info logged to EpisodeLogger
                                    → FairnessMetrics.compute(episode_log)
```

---

## Key Design Decisions (Summary)
See DESIGN_DECISIONS.md for full rationale.

| Decision | Choice | Reason |
|---|---|---|
| Episode structure | Single-step MDP | POC tractability |
| Action space | Continuous Box | More realistic than discrete bins |
| RL Algorithm | PPO | Stable, works with continuous actions, well-documented |
| Fairness integration | Reward shaping | Avoids constrained optimization complexity |
| Fairness penalty | Group rate disparity | Measurable, interpretable, relevant to domain |
| Config management | OmegaConf YAML | Clean, supports interpolation, familiar to ML engineers |
| Dashboard | Streamlit | Fast to build, widely known, good for ML demos |

---

## Future Extensions (mention in README to show depth of thinking)
1. **Multi-step portfolio MDP**: Agent manages a full loan book over T timesteps
2. **Real Mifos data integration**: Replace synthetic clients with Mifos API data
3. **Constrained RL**: Replace reward shaping with hard fairness constraints (CVaR, etc.)
4. **Multi-agent**: Competing lenders in a market simulation
5. **Counterfactual fairness**: More sophisticated fairness notion based on causal graphs