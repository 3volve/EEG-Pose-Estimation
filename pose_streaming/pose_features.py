"""Normalized upper-body pose features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pose_async import PoseResult


UPPER_BODY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24]

_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24
_POSITION_DIM = len(UPPER_BODY_LANDMARKS) * 3


def feature_dim(include_velocity: bool = True) -> int:
    return _POSITION_DIM * (2 if include_velocity else 1)


@dataclass(frozen=True, slots=True)
class PoseFeatureFrame:
    timestamp_ms: int
    received_time_s: float
    vector: NDArray[np.float32]
    pose_detected: bool
    confidence: float


class PoseFeatureExtractor:
    """Convert trusted pose results into normalized upper-body features."""

    def __init__(
        self,
        include_velocity: bool = True,
        use_world_landmarks: bool = True,
        min_scale: float = 1e-6,
    ) -> None:
        self.include_velocity = include_velocity
        self.use_world_landmarks = use_world_landmarks
        self.min_scale = min_scale
        self._previous_positions: NDArray[np.float32] | None = None
        self._previous_timestamp_ms: int | None = None

    def reset(self) -> None:
        self._previous_positions = None
        self._previous_timestamp_ms = None

    def extract(self, result: PoseResult) -> PoseFeatureFrame:
        if not result.pose_detected:
            self.reset()
            return self._empty_frame(result)

        landmarks = (
            result.world_landmarks
            if self.use_world_landmarks and result.world_landmarks
            else result.landmarks
        )
        coordinates = np.asarray(
            [(landmark.x, landmark.y, landmark.z) for landmark in landmarks],
            dtype=np.float32,
        )
        root = (
            coordinates[_LEFT_HIP] + coordinates[_RIGHT_HIP]
        ) / 2.0
        scale = np.linalg.norm(
            coordinates[_LEFT_SHOULDER] - coordinates[_RIGHT_SHOULDER]
        )
        if not np.isfinite(scale) or scale < self.min_scale:
            self.reset()
            return self._empty_frame(result)

        positions = (
            coordinates[UPPER_BODY_LANDMARKS] - root
        ) / scale
        if not np.isfinite(positions).all():
            self.reset()
            return self._empty_frame(result)

        flat_positions = positions.reshape(_POSITION_DIM)
        if self.include_velocity:
            if self._previous_positions is None:
                velocity = np.zeros_like(flat_positions)
            else:
                assert self._previous_timestamp_ms is not None
                elapsed_s = (
                    result.timestamp_ms - self._previous_timestamp_ms
                ) / 1000.0
                assert elapsed_s > 0
                velocity = (
                    flat_positions - self._previous_positions
                ) / elapsed_s
            vector = np.concatenate((flat_positions, velocity))
        else:
            vector = flat_positions

        self._previous_positions = flat_positions.copy()
        self._previous_timestamp_ms = result.timestamp_ms
        return PoseFeatureFrame(
            timestamp_ms=result.timestamp_ms,
            received_time_s=result.received_time_s,
            vector=vector.astype(np.float32, copy=False),
            pose_detected=True,
            confidence=_landmark_confidence(landmarks),
        )

    def _empty_frame(self, result: PoseResult) -> PoseFeatureFrame:
        return PoseFeatureFrame(
            timestamp_ms=result.timestamp_ms,
            received_time_s=result.received_time_s,
            vector=np.zeros(
                feature_dim(self.include_velocity),
                dtype=np.float32,
            ),
            pose_detected=False,
            confidence=0.0,
        )


def _landmark_confidence(landmarks: list) -> float:
    values = [
        value
        for index in UPPER_BODY_LANDMARKS
        for value in (
            landmarks[index].visibility,
            landmarks[index].presence,
        )
        if value is not None
    ]
    return float(np.mean(values)) if values else 1.0
