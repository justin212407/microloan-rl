.PHONY: help install train baseline evaluate dashboard test fairness

help:
	@printf "Available targets:\n  install   Install project dependencies\n  train     Train PPO and evaluate baselines\n  baseline  Run baseline agents entry point\n  evaluate  Run evaluation harness entry point\n  dashboard Launch Streamlit dashboard\n  test      Run pytest suite\n  fairness  Generate fairness audit artifacts\n"

install:
	pip install --only-binary :all: -r requirements.txt

train:
	PYTHONPATH=. python agents/train_ppo.py --config configs/default.yaml

baseline:
	PYTHONPATH=. python agents/baselines.py

evaluate:
	PYTHONPATH=. python agents/evaluate.py

fairness:
	PYTHONPATH=. python fairness/audit.py --results-dir results

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/
