"""PPO training entry point for the microloan pricing environment."""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from agents.baselines import FixedRateAgent, GreedyAgent, RandomAgent, RandomAgentConfig
from agents.evaluate import EvaluationResult, evaluate_agent
from env.microloan_env import MicroLoanEnv
from fairness.metrics import compute_fairness_report

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingArtifacts:
    """Filesystem locations used by training and evaluation.

    Attributes:
        model_path: Output path for the trained PPO model zip file.
        ppo_results_path: Output CSV path for PPO episode logs.
        summary_path: Output CSV path for aggregate evaluation metrics.
        results_dir: Directory where evaluation CSV files are written.
        tensorboard_log_dir: Directory used for TensorBoard event files.
    """

    model_path: Path
    ppo_results_path: Path
    summary_path: Path
    results_dir: Path
    tensorboard_log_dir: Path


class TensorBoardCallback(BaseCallback):
    """Record custom rollout metrics for TensorBoard dashboards."""

    def __init__(self) -> None:
        """Initialize the callback with quiet default verbosity."""
        super().__init__(verbose=0)

    def _on_step(self) -> bool:
        """Log custom metrics from the latest environment info dict."""
        info = self._latest_info()
        if info:
            self._record_scalar("rollout/profit", info.get("profit", 0.0))
            self._record_scalar("rollout/rate_offered", info.get("rate_offered", 0.0))
            self._record_scalar(
                "rollout/fairness_penalty",
                info.get("fairness_penalty", 0.0),
            )
        return True

    def _latest_info(self) -> dict[str, Any]:
        infos = self.locals.get("infos", [])
        return infos[0] if infos else {}

    def _record_scalar(self, key: str, value: Any) -> None:
        self.logger.record(key, float(value))


def main() -> None:
    """Run PPO training, evaluation, and artifact persistence."""
    args = _parse_args()
    config = OmegaConf.load(args.config)
    _configure_logging(config)
    artifacts = _prepare_artifacts(config)
    model = _train_ppo(args.config, config, artifacts, artifacts.model_path)
    results = _evaluate_all_agents(args.config, config, model, artifacts)
    _save_summary(results, artifacts.summary_path)
    _log_results(results)
    if bool(config.training.get("run_sensitivity_analysis", False)):
        run_sensitivity_analysis(args.config, config, artifacts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def _configure_logging(config: DictConfig) -> None:
    level_name = str(config.project.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")


def _prepare_artifacts(config: DictConfig) -> TrainingArtifacts:
    training_cfg = config.training
    model_dir = Path(str(training_cfg.get("model_dir", "models")))
    results_dir = Path(str(training_cfg.get("results_dir", "results")))
    tensorboard_dir = Path(str(training_cfg.get("tensorboard_log_dir", "logs/tensorboard")))
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    return TrainingArtifacts(
        model_path=model_dir / str(training_cfg.get("ppo_model_filename", "ppo_microloan.zip")),
        ppo_results_path=results_dir / str(training_cfg.get("ppo_results_filename", "ppo_results.csv")),
        summary_path=results_dir / str(training_cfg.get("evaluation_summary_filename", "evaluation_summary.csv")),
        results_dir=results_dir,
        tensorboard_log_dir=tensorboard_dir,
    )


def _train_ppo(
    config_path: str,
    config: DictConfig,
    artifacts: TrainingArtifacts,
    model_path: Path | None,
) -> PPO:
    env = _training_env(config_path, config)
    try:
        model = _build_model(env, config, artifacts)
        model.learn(
            total_timesteps=int(config.training.total_timesteps),
            callback=TensorBoardCallback(),
        )
        _save_model(model, model_path)
        return model
    finally:
        env.close()


def _training_env(config_path: str, config: DictConfig) -> Monitor:
    env = MicroLoanEnv(config_path=config_path)
    info_keys = [str(key) for key in config.env.info_keys]
    return Monitor(env, info_keywords=info_keys)


def _build_model(
    env: Monitor,
    config: DictConfig,
    artifacts: TrainingArtifacts,
) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=float(config.training.ppo_learning_rate),
        seed=int(config.project.seed),
        tensorboard_log=str(artifacts.tensorboard_log_dir),
        verbose=0,
    )


def _save_model(model: PPO, model_path: Path | None) -> None:
    if model_path is None:
        return
    model.save(str(model_path))
    LOGGER.info("Saved PPO model to %s.", model_path)


def _evaluate_all_agents(
    config_path: str,
    config: DictConfig,
    model: PPO,
    artifacts: TrainingArtifacts,
) -> list[EvaluationResult]:
    agents = _baseline_agents(config_path, config) + [("ppo", model)]
    results: list[EvaluationResult] = []
    for slug, agent in agents:
        result = _evaluate_single_agent(slug, agent, config_path, config)
        _save_result_logs(slug, result, artifacts)
        results.append(result)
    return results


def _baseline_agents(config_path: str, config: DictConfig) -> list[tuple[str, Any]]:
    env = MicroLoanEnv(config_path=config_path)
    try:
        seed = int(config.project.seed)
        action_space = env.action_space
        return [
            ("random", RandomAgent(action_space, RandomAgentConfig(seed=seed))),
            ("greedy", GreedyAgent(action_space)),
            ("fixed_rate", FixedRateAgent(action_space)),
        ]
    finally:
        env.close()


def _evaluate_single_agent(
    slug: str,
    agent: Any,
    config_path: str,
    config: DictConfig,
) -> EvaluationResult:
    env = MicroLoanEnv(config_path=config_path)
    try:
        return evaluate_agent(agent, env, int(config.training.evaluation_episodes))
    finally:
        env.close()


def _save_result_logs(
    slug: str,
    result: EvaluationResult,
    artifacts: TrainingArtifacts,
) -> None:
    output_path = artifacts.ppo_results_path if slug == "ppo" else artifacts.results_dir / f"{slug}_results.csv"
    result.episode_logs.to_csv(output_path, index=False)
    LOGGER.info("Saved %s episode logs to %s.", result.agent_name, output_path)


def _save_summary(results: list[EvaluationResult], output_path: Path) -> None:
    rows = [_summary_row(result) for result in results]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    LOGGER.info("Saved evaluation summary to %s.", output_path)


def _summary_row(result: EvaluationResult) -> dict[str, Any]:
    return {
        "agent_name": result.agent_name,
        "mean_reward": result.mean_reward,
        "mean_profit": result.mean_profit,
        "acceptance_rate": result.acceptance_rate,
        "repayment_rate": result.repayment_rate,
        "mean_rate_offered": result.mean_rate_offered,
    }


def _log_results(results: list[EvaluationResult]) -> None:
    for result in results:
        LOGGER.info(
            "%s mean_reward=%.4f mean_profit=%.4f",
            result.agent_name,
            result.mean_reward,
            result.mean_profit,
        )


def run_sensitivity_analysis(
    config_path: str,
    config: DictConfig,
    artifacts: TrainingArtifacts,
) -> pd.DataFrame:
    """Train PPO across beta values and record fairness tradeoffs."""
    try:
        rows = [
            _sensitivity_row(config, artifacts, float(beta))
            for beta in config.reward.sensitivity_beta_values
        ]
        frame = pd.DataFrame(rows)
        output_path = artifacts.results_dir / "sensitivity_analysis.csv"
        frame.to_csv(output_path, index=False)
        LOGGER.info("Saved sensitivity analysis to %s.", output_path)
        return frame
    finally:
        _cleanup_sensitivity_configs(artifacts)


def _sensitivity_row(
    config: DictConfig,
    artifacts: TrainingArtifacts,
    beta: float,
) -> dict[str, float]:
    beta_path, beta_config = _write_beta_config(config, artifacts, beta)
    LOGGER.info(
        "Sensitivity run beta=%.2f config beta=%.4f",
        beta,
        float(beta_config.reward.beta),
    )
    merged = OmegaConf.merge(
        beta_config,
        {
            "training": {
                "total_timesteps": int(
                    config.training.get(
                        "sensitivity_timesteps",
                        config.training.total_timesteps,
                    )
                )
            }
        },
    )
    OmegaConf.save(config=merged, f=beta_path)
    model_path = artifacts.results_dir / f"ppo_beta_{beta:.2f}.zip"
    model = _train_ppo(str(beta_path), merged, artifacts, model_path=model_path)
    result = _evaluate_single_agent("ppo", model, str(beta_path), merged)
    report = compute_fairness_report(result.episode_logs)
    LOGGER.info(
        "beta=%.2f mean_profit=%.4f fairness_score=%.4f controlled_rdg=%.4f",
        beta,
        result.mean_profit,
        report.fairness_score,
        report.controlled_rate_disparity_gap,
    )
    return _sensitivity_metrics(beta, result, report)


def _write_beta_config(
    config: DictConfig,
    artifacts: TrainingArtifacts,
    beta: float,
) -> tuple[Path, DictConfig]:
    config_dir = artifacts.results_dir / "sensitivity_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    merged = OmegaConf.merge(config, {"reward": {"beta": beta}})
    path = config_dir / f"temp_beta_{beta:.2f}.yaml"
    OmegaConf.save(config=merged, f=path)
    LOGGER.debug("Saved sensitivity config for beta %.2f to %s.", beta, path)
    return path, merged


def _sensitivity_metrics(
    beta: float,
    result: EvaluationResult,
    report: Any,
) -> dict[str, float]:
    return {
        "beta": beta,
        "mean_profit": result.mean_profit,
        "fairness_score": report.fairness_score,
        "controlled_rdg": report.controlled_rate_disparity_gap,
    }


def _cleanup_sensitivity_configs(artifacts: TrainingArtifacts) -> None:
    config_dir = artifacts.results_dir / "sensitivity_configs"
    if config_dir.exists():
        shutil.rmtree(config_dir)
        LOGGER.debug("Cleaned up sensitivity temp configs.")


if __name__ == "__main__":
    main()
