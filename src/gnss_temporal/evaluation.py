from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from stable_baselines3 import DQN

from .data import CLASS_NAMES, prepare_temporal_data
from .metrics import classification_metrics, full_report, predict_segments


def evaluate_model(config: dict, model_path: Path | None = None) -> dict:
    output = config["output_dir"]
    scaler_path = output / "minmax_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError("Training artifacts are missing; run src/main.py first.")
    import joblib
    scaler = joblib.load(scaler_path)
    dataset = prepare_temporal_data(config, scaler=scaler)

    if model_path is None:
        model_path = output / "best_model.zip"
        if not model_path.exists():
            model_path = output / "final_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}")

    model = DQN.load(str(model_path), device="auto")
    history_length = int(config["temporal"]["history_length"])
    truth, predictions, switch_rate = predict_segments(model, dataset.test, history_length)
    metrics = classification_metrics(truth, predictions, switch_rate)
    metrics["model"] = str(model_path)
    metrics["recordings"] = len(dataset.test)
    output.mkdir(parents=True, exist_ok=True)
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report, matrix = full_report(truth, predictions)
    pd.DataFrame(report).transpose().to_csv(output / "classification_report.csv")
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)), yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted label", ylabel="True label",
        title="Temporal DQN — chronological held-out test blocks",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for row in range(len(CLASS_NAMES)):
        for col in range(len(CLASS_NAMES)):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center",
                    color="white" if matrix[row, col] > 0.55 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label="Recall")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    print("\nChronological held-out test metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nPer-class results:")
    print(classification_report(
        truth, predictions, labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, zero_division=0,
    ))
    print("\nOutputs:", output)
    return metrics
