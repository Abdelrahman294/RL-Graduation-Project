from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data import SequenceSegment


class TemporalSpoofingEnv(gym.Env):
    """Multi-step classification over one contiguous channel/PRN track."""

    metadata = {"render_modes": []}

    def __init__(self, segments: list[SequenceSegment], history_length: int,
                 episode_steps: int, shaped_classes: tuple[int, ...] = (5, 6)):
        super().__init__()
        if not segments:
            raise ValueError("At least one training segment is required.")
        self.history_length = int(history_length)
        self.episode_steps = int(episode_steps)
        self.shaped_classes = set(shaped_classes)
        self.feature_dim = int(segments[0].x.shape[1])
        self.segments_by_class = {
            label: [segment for segment in segments if segment.label == label]
            for label in sorted({segment.label for segment in segments})
        }
        min_length = self.history_length - 1 + self.episode_steps
        if any(
            segment.x.ndim != 2
            or segment.x.shape[1] != self.feature_dim
            or len(segment.x) < min_length
            for segment in segments
        ):
            raise ValueError(
                f"Every training track must have at least {min_length} observations "
                "and the same feature dimension."
            )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.history_length * self.feature_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(7)
        self.segment: SequenceSegment | None = None
        self.history: np.ndarray | None = None
        self.index = 0
        self.steps_in_episode = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        labels = list(self.segments_by_class)
        label = labels[int(self.np_random.integers(0, len(labels)))]
        candidates = self.segments_by_class[label]
        self.segment = candidates[int(self.np_random.integers(0, len(candidates)))]
        minimum_start = self.history_length - 1
        maximum_start = len(self.segment.x) - self.episode_steps
        self.index = int(self.np_random.integers(minimum_start, maximum_start + 1))
        history_start = self.index - self.history_length + 1
        self.history = self.segment.x[history_start:self.index + 1].copy()
        self.steps_in_episode = 0
        info = {
            "recording": self.segment.recording,
            "channel_id": self.segment.channel_id,
            "prn": self.segment.prn,
            "epoch_index": self.segment.start_row + self.index,
            "label": self.segment.label,
        }
        return self.history.reshape(-1).copy(), info

    def step(self, action: int):
        if self.segment is None or self.history is None:
            raise RuntimeError("Call reset() before step().")
        action = int(np.asarray(action).item())
        truth = int(self.segment.label)
        correct = action == truth
        magnitude = 2.0 if truth in self.shaped_classes else 1.0
        reward = magnitude if correct else -magnitude

        self.steps_in_episode += 1
        next_index = self.index + 1
        terminated = (
            self.steps_in_episode >= self.episode_steps
            or next_index >= len(self.segment.x)
        )
        info = {
            "recording": self.segment.recording,
            "channel_id": self.segment.channel_id,
            "prn": self.segment.prn,
            "epoch_index": self.segment.start_row + self.index,
            "label": truth,
            "action": action,
            "correct": correct,
        }
        if not terminated:
            self.index = next_index
            self.history[:-1] = self.history[1:]
            self.history[-1] = self.segment.x[self.index]
        return self.history.reshape(-1).copy(), float(reward), terminated, False, info
