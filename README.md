# Temporal GNSS Spoofing DQN

This is a separate temporal extension of the paper reproduction in `../gnss_dqn_reproduction`. It keeps the original implementation untouched. The seven CSV recordings required by this project are included in the repository under `data/`, so a clone or Codespace has the data in the expected location.

## Data and temporal design

The CSV contains eight channel blocks per source row, but the receiver timestamps are recorded independently per channel. Their values can repeat because of timestamp quantization, so the implementation does not assume that all eight channels in one row are synchronized.

Each model stream is one uninterrupted channel/PRN track. Rows remain in source order within each channel; a stream is cut when that channel changes PRNs or encounters a PRN excluded from the shared set. The seven configured PRNs `[3, 6, 7, 13, 16, 19, 23]` are present in every scenario, while scenario-specific PRNs are excluded so satellite presence alone cannot identify the class. Absolute receiver time is checked for non-decreasing order within each channel but is not given to the model.

Each DQN state contains 10 consecutive observations from one track (8 signal features per observation, 80 input values). Episodes last 128 decisions. At reset, the environment randomly chooses a class, one of its training tracks, and a valid starting position; within an episode, observations proceed in order and the agent receives a class reward at each step.

Each source file is divided chronologically into train, validation, and test blocks. A 9-row embargo separates the blocks. Histories cannot cross a split boundary, and MinMax scaling is fitted on training observations from the shared PRNs only.

This dataset has one file per class, so the chronological test blocks cover later portions of the same captures, not independent capture runs. The files provide scenario labels rather than exact spoofing-onset annotations; results therefore report per-observation class metrics and within-track prediction-switch rate, not attack-onset detection latency.

## Run

From this folder in PowerShell, using the Python environment where the dependencies are installed:

```powershell
py src/main.py
```

The entry point checks and prints dependency availability before importing the project modules, then trains and evaluates the temporal DQN. It automatically uses CUDA when PyTorch detects it. The initial configuration uses 100,000 steps as a temporal-model pilot; increase `training.total_timesteps` in `configs/default.yaml` after reviewing validation results.

The run saves best-validation and final checkpoints plus the scaler and split manifest under `artifacts/`. Evaluation prints accuracy, macro precision/recall/F1, adjacent prediction-switch rate, and per-class results. It writes `test_metrics.json`, `classification_report.csv`, and `confusion_matrix.png`.

## Main files

- `configs/default.yaml`: paths, shared PRNs, history size, time split, episode length, and DQN parameters.
- `src/main.py`: dependency availability report and full run entry point.
- `src/gnss_temporal/data.py`: per-channel timestamp checks, PRN-track segmentation, chronological splits, and train-only scaling.
- `src/gnss_temporal/environment.py`: multi-step environment over contiguous track windows.
- `src/gnss_temporal/metrics.py`: chronological history windows and prediction-switch metric.
- `src/gnss_temporal/training.py`: DQN setup and periodic validation.
- `src/gnss_temporal/evaluation.py`: final chronological test evaluation and reports.

