"""Polling wrapper that exposes live pose autoencoder latents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from config import (
    DEFAULT_DEVICE,
    POSE_INCLUDE_VELOCITY,
    POSE_LIVE_MEAN_WINDOW,
    POSE_LIVE_MEDIAN_WINDOW,
    POSE_LIVE_SMOOTHING,
    POSE_USE_WORLD_LANDMARKS,
)
from streaming.pose import PoseFeatureExtractor, feature_dim

from .pose_autoencoder import PoseAutoencoder, load_checkpoint


@dataclass(frozen=True, slots=True)
class PoseLatentFrame:
    timestamp_ms: int
    received_time_s: float
    feature_vector: NDArray[np.float32]
    latent: NDArray[np.float32]
    reconstruction: NDArray[np.float32]
    reconstruction_error: float
    pose_detected: bool
    confidence: float


class PoseLatentStream:
    """Poll an existing pose estimator and encode each new result."""

    def __init__(
        self,
        pose_estimator,
        autoencoder: PoseAutoencoder,
        feature_extractor: PoseFeatureExtractor,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self.pose_estimator = pose_estimator
        self.autoencoder = autoencoder.to(device)
        self.autoencoder.eval()
        self.feature_extractor = feature_extractor
        self.device = torch.device(device)
        assert self.autoencoder.input_dim == feature_dim(
            self.feature_extractor.include_velocity
        )
        self._latest: PoseLatentFrame | None = None
        self._latest_timestamp_ms: int | None = None

    @classmethod
    def from_checkpoint(
        cls,
        pose_estimator,
        checkpoint_path: str | Path,
        include_velocity: bool = POSE_INCLUDE_VELOCITY,
        use_world_landmarks: bool = POSE_USE_WORLD_LANDMARKS,
        smooth_positions: bool = POSE_LIVE_SMOOTHING,
        device: str = DEFAULT_DEVICE,
    ) -> PoseLatentStream:
        model, _ = load_checkpoint(checkpoint_path, map_location=device)
        extractor = PoseFeatureExtractor(
            include_velocity=include_velocity,
            use_world_landmarks=use_world_landmarks,
            smooth_positions=smooth_positions,
            median_window=POSE_LIVE_MEDIAN_WINDOW,
            mean_window=POSE_LIVE_MEAN_WINDOW,
        )
        expected_dim = feature_dim(include_velocity)
        if model.input_dim != expected_dim:
            raise ValueError(
                f"Checkpoint expects {model.input_dim} features, but the "
                f"extractor produces {expected_dim}"
            )
        return cls(pose_estimator, model, extractor, device)

    def get_latest(self) -> PoseLatentFrame | None:
        result = self.pose_estimator.get_latest()
        if result is None:
            return self._latest
        if result.timestamp_ms != self._latest_timestamp_ms:
            self._latest = self._process(result)
            self._latest_timestamp_ms = result.timestamp_ms
        return self._latest

    def get_nowait(self) -> PoseLatentFrame | None:
        result = self.pose_estimator.get_nowait()
        if result is None:
            return None
        if result.timestamp_ms == self._latest_timestamp_ms:
            return self._latest
        self._latest = self._process(result)
        self._latest_timestamp_ms = result.timestamp_ms
        return self._latest

    def _process(self, result) -> PoseLatentFrame:
        feature_frame = self.feature_extractor.extract(result)
        features = torch.from_numpy(feature_frame.vector).to(self.device)
        with torch.no_grad():
            latent = self.autoencoder.encode(features)
            reconstruction = self.autoencoder.decode(latent)

        latent_array = latent.cpu().numpy().astype(np.float32, copy=False)
        reconstruction_array = (
            reconstruction.cpu().numpy().astype(np.float32, copy=False)
        )
        error = float(
            np.mean(np.abs(reconstruction_array - feature_frame.vector))
        )
        return PoseLatentFrame(
            timestamp_ms=feature_frame.timestamp_ms,
            received_time_s=feature_frame.received_time_s,
            feature_vector=feature_frame.vector,
            latent=latent_array,
            reconstruction=reconstruction_array,
            reconstruction_error=error,
            pose_detected=feature_frame.pose_detected,
            confidence=feature_frame.confidence,
        )
