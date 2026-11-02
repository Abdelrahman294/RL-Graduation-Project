from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from .data import TemporalDataset, prepare_temporal_data, write_manifest
from .environment import TemporalSpoofingEnv
from .metrics import classification_metrics, predict_segments


class TemporalValidationCallback(BaseCallback):
    def __init__(self, validation_segments, history_length: int, eval_freq: int,
                 output_dir: Path, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.validation_segments = validation_segments
        self.history_length = history_length
        self.eval_freq = eval_freq
        self.output_dir = output_dir
        self.best_macro_f1 = -1.0

    def _evaluate_and_save_if_best(self) -> None:
        truth, predictions, switch_rate = predict_segments(
            self.model, self.validation_segments, self.history_length,
        )
        metrics = classification_metrics(truth, predictions, switch_rate)
        metrics["timesteps"] = int(self.num_timesteps)
        print(
            f"\nValidation at {self.num_timesteps:,} steps: "
            f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
            f"prediction_switch_rate={switch_rate:.4f}",
            flush=True,
        )
        if metrics["macro_f1"] > self.best_macro_f1:
            self.best_macro_f1 = metrics["macro_f1"]
            self.model.save(self.output_dir / "best_model")
            (self.output_dir / "best_validation_metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8",
            )
            print("Saved new best validation checkpoint.", flush=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            self._evaluate_and_save_if_best()
        return True

    def _on_training_end(self) -> None:
        if self.best_macro_f1 < 0:
            self._evaluate_and_save_if_best()


def train_model(config: dict) -> tuple[TemporalDataset, Path]:
    seed = int(config["seed"])
    set_random_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    dataset = prepare_temporal_data(config)
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(dataset.scaler, output / "minmax_scaler.joblib")
    write_manifest(output / "data_manifest.json", dataset.split_manifest)

    training_cfg = config["training"]
    history_length = int(config["temporal"]["history_length"])
    env = make_vec_env(
        lambda: TemporalSpoofingEnv(
            dataset.train,
            history_length=history_length,
            episode_steps=int(config["temporal"]["episode_steps"]),
        ),
        n_envs=1,
        seed=seed,
    )
    callback = TemporalValidationCallback(
        validation_segments=dataset.validation,
        history_length=history_length,
        eval_freq=int(training_cfg["eval_freq"]),
        output_dir=output,
    )
    model = DQN(
        "MlpPolicy", env,
        learning_rate=float(training_cfg["learning_rate"]),
        gamma=float(training_cfg["gamma"]),
        buffer_size=int(training_cfg["buffer_size"]),
        learning_starts=int(training_cfg["learning_starts"]),
        batch_size=int(training_cfg["batch_size"]),
        train_freq=int(training_cfg["train_freq"]),
        target_update_interval=int(training_cfg["target_update_interval"]),
        exploration_fraction=float(training_cfg["exploration_fraction"]),
        exploration_initial_eps=float(training_cfg["exploration_initial_eps"]),
        exploration_final_eps=float(training_cfg["exploration_final_eps"]),
        policy_kwargs={"net_arch": training_cfg["net_arch"], "activation_fn": torch.nn.ReLU},
        seed=seed,
        verbose=1,
        device="auto",
    )
    print(
        f"Temporal observation: {history_length} epochs x {dataset.feature_dim} values "
        f"= {history_length * dataset.feature_dim} inputs; "
        f"episode length={int(config['temporal']['episode_steps'])}.",
        flush=True,
    )
    model.learn(
        total_timesteps=int(training_cfg["total_timesteps"]),
        callback=callback,
        progress_bar=False,
        tb_log_name="temporal_dqn",
    )
    model.save(output / "final_model")
    env.close()

    selected_path = output / "best_model.zip"
    if not selected_path.exists():
        selected_path = output / "final_model.zip"
    print(f"Training finished. Selected model: {selected_path}", flush=True)
    return dataset, selected_path
