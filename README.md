# DMP Micro-loan RL

Fairness-aware reinforcement learning for dynamic pricing of micro-loans in a realistic synthetic microfinance environment.

## Motivation

Microfinance pricing decisions affect whether low-income borrowers can access capital on sustainable terms. A purely profit-maximizing RL agent can learn pricing patterns that look efficient on average while systematically disadvantaging certain demographic groups. This project turns that tension into a measurable engineering problem: train a pricing policy, model the profit-risk-fairness tradeoff explicitly, and audit whether similar-risk clients are treated similarly. For mentors, the core value is not just that PPO can optimize a policy, but that the repo shows production-quality thinking about fairness in a real financial setting.

## Architecture

```text
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

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
make install
make train
make dashboard
```

## Key Design Decisions

### Why Single-Step MDP?

Each episode is one pricing decision for one client. That keeps the proof-of-concept tractable, makes reward attribution crisp, and lets us focus on the core pricing tradeoff before moving to a harder portfolio-level control problem.

### Why PPO Instead of DQN?

The action is a continuous interest rate in `[0.08, 0.36]`. PPO works naturally with continuous control, while vanilla DQN assumes a discrete action space and would force us into artificial rate bins.

### Why Reward Shaping Instead of Constrained RL?

Reward shaping is simpler to explain, faster to prototype, and easier to visualize in a DMP/GSoC proof-of-concept. It lets us expose the fairness tradeoff directly through `beta` without adding the implementation and tuning complexity of constrained RL methods.

### Why Controlled_RDG Instead of Raw RDG?

Raw rate disparity can be misleading because groups can have different mixes of low-, medium-, and high-risk clients. Controlled_RDG compares pricing gaps within each risk segment first, then averages those gaps, so it asks the right question: are similar-risk borrowers being priced similarly across demographic groups? That makes it the primary fairness metric in this repo.

## Results

Placeholder evaluation table:

| Agent | Mean Reward | Mean Profit | Acceptance Rate | Controlled RDG | Fairness Score |
|---|---:|---:|---:|---:|---:|
| Random Agent | TBD | TBD | TBD | TBD | TBD |
| Greedy Agent | TBD | TBD | TBD | TBD | TBD |
| Fixed Rate Agent | TBD | TBD | TBD | TBD | TBD |
| PPO | TBD | TBD | TBD | TBD | TBD |

## Future Extensions

1. Multi-step portfolio MDP where the agent manages a full loan book over time.
2. Real Mifos data integration to replace the synthetic client pool.
3. Constrained RL with hard fairness constraints instead of reward shaping.
4. Multi-agent market simulation with competing lenders.
5. Counterfactual fairness analysis based on causal assumptions.

## Tech Stack

| Layer | Library | Version Constraint |
|---|---|---|
| RL Environment | `gymnasium` | `>=0.29` |
| RL Training | `stable-baselines3` | `>=2.0` |
| Tensor Backend | `torch` | `>=2.0` |
| Data | `numpy`, `pandas` | latest stable |
| Fairness | `fairlearn` | `>=0.10` |
| Visualization | `matplotlib`, `seaborn` | latest stable |
| Dashboard | `streamlit` | `>=1.30` |
| Config | `pyyaml`, `omegaconf` | latest stable |
| Experiment Tracking | `tensorboard` | latest stable |
| Testing | `pytest` | latest stable |
# microloan-rl
