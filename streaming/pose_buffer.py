from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from config import PAIRING_MAX_POSE_GAP_S

if TYPE_CHECKING:
    from pose_encoding import PoseLatentFrame


@dataclass(frozen=True, slots=True)
class InterpolatedPoseLatent:
    target_time_s: float
    latent: NDArray[np.float32]
    pose_confidence: float
    pose_reconstruction_error: float
    interpolation_confidence: float


class PoseLatentBuffer:
    """Keep recent pose latents and interpolate truth labels by monotonic camera capture time."""

    def __init__(
        self,
        *,
        max_gap_s: float = PAIRING_MAX_POSE_GAP_S,
        max_frames: int = 256,
    ) -> None:
        self.max_gap_s = max_gap_s
        self._frames: deque["PoseLatentFrame"] = deque(maxlen=max_frames)
        self._latest_timestamp_ms: int | None = None

    def add(self, frame: "PoseLatentFrame | None") -> None:
        if frame is None:
            return
        if frame.timestamp_ms == self._latest_timestamp_ms:
            return
        self._frames.append(frame)
        self._latest_timestamp_ms = frame.timestamp_ms

    @property
    def latest_time_s(self) -> float | None:
        if not self._frames:
            return None
        return self._frames[-1].timestamp_ms / 1000.0

    def latent_at(self, target_time_s: float) -> InterpolatedPoseLatent | None:
        if len(self._frames) < 2:
            return None

        before: PoseLatentFrame | None = None
        after: PoseLatentFrame | None = None
        for frame in self._frames:
            if frame.timestamp_ms / 1000.0 <= target_time_s:
                before = frame
                continue
            after = frame
            break

        if before is None or after is None:
            return None

        gap_s = (after.timestamp_ms - before.timestamp_ms) / 1000.0
        if gap_s <= 0 or gap_s > self.max_gap_s:
            return None

        alpha = (target_time_s - before.timestamp_ms / 1000.0) / gap_s
        latent = ((1.0 - alpha) * before.latent + alpha * after.latent).astype(
            np.float32,
            copy=False,
        )
        pose_confidence = float(
            (1.0 - alpha) * before.confidence + alpha * after.confidence
        )
        reconstruction_error = float(
            (1.0 - alpha) * before.reconstruction_error
            + alpha * after.reconstruction_error
        )
        gap_confidence = max(0.0, 1.0 - gap_s / self.max_gap_s)
        error_confidence = 1.0 / (1.0 + reconstruction_error)
        interpolation_confidence = (
            pose_confidence * gap_confidence * error_confidence
        )
        return InterpolatedPoseLatent(
            target_time_s=target_time_s,
            latent=latent,
            pose_confidence=pose_confidence,
            pose_reconstruction_error=reconstruction_error,
            interpolation_confidence=float(interpolation_confidence),
        )
