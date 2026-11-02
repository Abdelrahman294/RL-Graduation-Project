from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

CLASS_NAMES = ["cs", "ds1", "ds2", "ds3", "ds4", "ds7", "ds8"]
CHANNELS_PER_ROW = 8
FIELDS_PER_CHANNEL = 11
# Receiver time is used for chronology checks, not as an absolute input feature.
FEATURE_NAMES = [
    "acq_doppler_hz", "acq_doppler_step", "fs", "prompt_i", "prompt_q",
    "cn0_db_hz", "carrier_doppler_hz", "pseudorange_m",
]
FEATURE_OFFSETS = list(range(2, 10))


@dataclass
class RawRecording:
    name: str
    label: int
    features: np.ndarray  # [source row, channel, feature]
    prns: np.ndarray  # [source row, channel]
    channel_times_ms: np.ndarray  # [source row, channel]


@dataclass
class SequenceSegment:
    recording: str
    label: int
    x: np.ndarray  # chronological observations from one uninterrupted channel/PRN track
    start_row: int
    end_row: int
    channel_id: int
    prn: int


@dataclass
class TemporalDataset:
    train: list[SequenceSegment]
    validation: list[SequenceSegment]
    test: list[SequenceSegment]
    scaler: MinMaxScaler
    prn_slots: list[int]
    feature_dim: int
    split_manifest: dict


def _read_recording(path: Path, name: str, label: int,
                    drop_initial_rows: int) -> RawRecording:
    frame = pd.read_csv(path, header=None, skip_blank_lines=True, skipinitialspace=True)
    expected_fields = CHANNELS_PER_ROW * FIELDS_PER_CHANNEL
    while frame.shape[1] > expected_fields:
        if not frame.iloc[:, -1].isna().all():
            raise ValueError(
                f"{path.name} has {frame.shape[1]} columns; expected {expected_fields} "
                "plus at most an empty trailing field."
            )
        frame = frame.iloc[:, :-1]
    if frame.shape[1] != expected_fields:
        raise ValueError(f"{path.name} has {frame.shape[1]} columns; expected {expected_fields}.")
    if len(frame) <= drop_initial_rows:
        raise ValueError(f"{path.name} has too few rows after warm-up trimming.")

    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    values = values[drop_initial_rows:]
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains missing or non-numeric values after warm-up trimming.")

    rows = len(values)
    features = np.empty((rows, CHANNELS_PER_ROW, len(FEATURE_OFFSETS)), dtype=np.float64)
    prns = np.empty((rows, CHANNELS_PER_ROW), dtype=np.int64)
    times = np.empty((rows, CHANNELS_PER_ROW), dtype=np.float64)
    for channel in range(CHANNELS_PER_ROW):
        base = channel * FIELDS_PER_CHANNEL
        prn_values = values[:, base + 1]
        if not np.equal(prn_values, np.floor(prn_values)).all():
            raise ValueError(f"{path.name} contains a non-integer PRN value.")
        prns[:, channel] = prn_values.astype(np.int64)
        features[:, channel, :] = values[:, [base + offset for offset in FEATURE_OFFSETS]]
        times[:, channel] = values[:, base + 10]

    # RX time is recorded independently per tracking channel. Repeated values are valid
    # because the CSV time field is quantized; source-row order breaks ties.
    for channel in range(CHANNELS_PER_ROW):
        if np.any(np.diff(times[:, channel]) < -1e-9):
            raise ValueError(
                f"{path.name} receiver time goes backwards in channel {channel}; "
                "the source row order cannot be used as that track's chronology."
            )
    return RawRecording(name, label, features, prns, times)


def _split_bounds(length: int, split: dict, embargo: int, history_length: int,
                  episode_steps: int, recording: str) -> dict[str, tuple[int, int]]:
    fractions = [float(split[key]) for key in ("train", "validation", "test")]
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Train, validation, and test split fractions must sum to 1.")

    train_end = int(length * fractions[0])
    validation_end = int(length * (fractions[0] + fractions[1]))
    bounds = {
        "train": (0, train_end),
        "validation": (train_end + embargo, validation_end),
        "test": (validation_end + embargo, length),
    }
    for name, (start, end) in bounds.items():
        minimum = history_length - 1 + episode_steps if name == "train" else history_length
        if end - start < minimum:
            raise ValueError(
                f"{recording} {name} block has {end-start} rows; needs at least {minimum}."
            )
    return bounds


def _make_track_segments(recording: RawRecording, scaled_features: np.ndarray,
                         common_prns: set[int], start: int, end: int,
                         min_length: int) -> list[SequenceSegment]:
    segments = []
    for channel in range(CHANNELS_PER_ROW):
        run_start = None
        run_prn = None
        for row in range(start, end + 1):
            current_prn = int(recording.prns[row, channel]) if row < end else None
            valid = current_prn in common_prns if current_prn is not None else False
            if run_start is not None and (not valid or current_prn != run_prn):
                if row - run_start >= min_length:
                    segments.append(SequenceSegment(
                        recording=recording.name,
                        label=recording.label,
                        x=scaled_features[run_start:row, channel, :].copy(),
                        start_row=run_start,
                        end_row=row,
                        channel_id=channel,
                        prn=int(run_prn),
                    ))
                run_start = None
                run_prn = None
            if valid and run_start is None:
                run_start = row
                run_prn = current_prn
    return segments


def prepare_temporal_data(config: dict, scaler: MinMaxScaler | None = None) -> TemporalDataset:
    recordings_config = config["recordings"]
    if list(recordings_config) != CLASS_NAMES:
        raise ValueError(f"Recording classes must appear in this order: {CLASS_NAMES}")
    raw = []
    for label, (name, filename) in enumerate(recordings_config.items()):
        path = config["data_dir"] / filename
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        raw.append(_read_recording(path, name, label, int(config["drop_initial_rows"])))

    history_length = int(config["temporal"]["history_length"])
    episode_steps = int(config["temporal"]["episode_steps"])
    embargo = config["split"].get("embargo_epochs")
    embargo = history_length - 1 if embargo is None else int(embargo)
    if history_length < 1 or episode_steps < 1 or embargo < history_length - 1:
        raise ValueError("History/episode lengths must be positive and embargo must be at least history_length - 1.")

    prn_slots = [int(prn) for prn in config["common_prns"]]
    if not prn_slots or len(set(prn_slots)) != len(prn_slots):
        raise ValueError("common_prns must be a non-empty list of unique PRNs.")
    common_prns = set(prn_slots)
    split_bounds = {
        recording.name: _split_bounds(
            len(recording.features), config["split"], embargo, history_length,
            episode_steps, recording.name,
        )
        for recording in raw
    }

    if scaler is None:
        training_features = []
        for recording in raw:
            train_end = split_bounds[recording.name]["train"][1]
            prn_mask = np.isin(recording.prns[:train_end], prn_slots)
            if not prn_mask.any():
                raise ValueError(f"No configured common PRNs found in {recording.name}'s training block.")
            training_features.append(recording.features[:train_end][prn_mask])
        scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=True)
        scaler.fit(np.concatenate(training_features, axis=0))

    segments: dict[str, list[SequenceSegment]] = {key: [] for key in ("train", "validation", "test")}
    manifest_recordings = {}
    min_train_track = history_length - 1 + episode_steps
    for recording in raw:
        scaled = scaler.transform(recording.features.reshape(-1, len(FEATURE_NAMES)))
        scaled = scaled.reshape(recording.features.shape).astype(np.float32)
        manifest_recordings[recording.name] = {
            "label": recording.label,
            "rows_after_warmup": int(len(recording.features)),
            "time_start_ms": float(recording.channel_times_ms.min()),
            "time_end_ms": float(recording.channel_times_ms.max()),
            "channel_time_order": "non-decreasing per channel; duplicate quantized times retain source-row order",
            "splits": {},
        }
        for split_name, (start, end) in split_bounds[recording.name].items():
            min_length = min_train_track if split_name == "train" else history_length
            track_segments = _make_track_segments(
                recording, scaled, common_prns, start, end, min_length,
            )
            if not track_segments:
                raise ValueError(
                    f"{recording.name} has no sufficiently long common-PRN tracks in {split_name}."
                )
            segments[split_name].extend(track_segments)
            manifest_recordings[recording.name]["splits"][split_name] = {
                "start_row_inclusive": int(start),
                "end_row_exclusive": int(end),
                "rows": int(end - start),
                "track_segments": len(track_segments),
                "track_epochs": int(sum(len(segment.x) for segment in track_segments)),
            }

    for split_name in ("train", "validation", "test"):
        present_labels = {segment.label for segment in segments[split_name]}
        expected_labels = set(range(len(CLASS_NAMES)))
        if present_labels != expected_labels:
            missing = [CLASS_NAMES[label] for label in sorted(expected_labels - present_labels)]
            raise ValueError(f"{split_name} is missing track data for classes: {missing}")

    manifest = {
        "class_names": CLASS_NAMES,
        "feature_names": FEATURE_NAMES,
        "receiver_time_input": False,
        "receiver_time_usage": "per-channel monotonic-order validation; source row resolves duplicate timestamps",
        "sequence_unit": "one contiguous source channel and PRN track",
        "common_prns": prn_slots,
        "excluded_recording_specific_prns": {
            rec.name: sorted(set(rec.prns.ravel().tolist()) - common_prns) for rec in raw
        },
        "history_length": history_length,
        "embargo_rows": embargo,
        "scaler_fit_scope": "training rows, configured common PRNs only",
        "recordings": manifest_recordings,
    }
    return TemporalDataset(
        train=segments["train"], validation=segments["validation"], test=segments["test"],
        scaler=scaler, prn_slots=prn_slots,
        feature_dim=len(FEATURE_NAMES),
        split_manifest=manifest,
    )


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
