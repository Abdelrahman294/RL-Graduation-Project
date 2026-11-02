from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from .data import CLASS_NAMES, SequenceSegment


def make_history_windows(segment: SequenceSegment, history_length: int) -> np.ndarray:
    if len(segment.x) < history_length:
        raise ValueError(f"{segment.recording} has fewer epochs than the history window.")
    return np.asarray([
        segment.x[end - history_length + 1:end + 1].reshape(-1)
        for end in range(history_length - 1, len(segment.x))
    ], dtype=np.float32)


def predict_segments(model, segments: list[SequenceSegment],
                     history_length: int) -> tuple[np.ndarray, np.ndarray, float]:
    truth_parts = []
    prediction_parts = []
    total_switches = 0
    total_adjacent_pairs = 0
    for segment in segments:
        windows = make_history_windows(segment, history_length)
        predictions, _ = model.predict(windows, deterministic=True)
        predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
        truth_parts.append(np.full(len(predictions), segment.label, dtype=np.int64))
        prediction_parts.append(predictions)
        if len(predictions) > 1:
            total_switches += int(np.count_nonzero(predictions[1:] != predictions[:-1]))
            total_adjacent_pairs += len(predictions) - 1

    truth = np.concatenate(truth_parts)
    predictions = np.concatenate(prediction_parts)
    switch_rate = total_switches / total_adjacent_pairs if total_adjacent_pairs else 0.0
    return truth, predictions, float(switch_rate)


def classification_metrics(truth: np.ndarray, predictions: np.ndarray,
                           switch_rate: float = 0.0) -> dict:
    labels = np.arange(len(CLASS_NAMES))
    return {
        "epochs": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": float(precision_score(
            truth, predictions, labels=labels, average="macro", zero_division=0,
        )),
        "macro_recall": float(recall_score(
            truth, predictions, labels=labels, average="macro", zero_division=0,
        )),
        "macro_f1": float(f1_score(
            truth, predictions, labels=labels, average="macro", zero_division=0,
        )),
        "adjacent_prediction_switch_rate": float(switch_rate),
    }


def full_report(truth: np.ndarray, predictions: np.ndarray) -> tuple[dict, np.ndarray]:
    report = classification_report(
        truth, predictions, labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(
        truth, predictions, labels=np.arange(len(CLASS_NAMES)), normalize="true",
    )
    return report, matrix
