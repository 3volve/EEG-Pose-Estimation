from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import (
    DEFAULT_DEVICE,
    EEG_ADAPTATION_MODE_SESSION,
    EEG_CALIBRATION_MAX_POSE_RECONSTRUCTION_ERROR,
    EEG_CALIBRATION_MIN_INTERPOLATION_CONFIDENCE,
    EEG_CALIBRATION_MIN_POSE_CONFIDENCE,
    EEG_MODEL_LR,
    EEG_ONLINE_CALIBRATION_BATCH_SIZE,
    EEG_ONLINE_CALIBRATION_MAX_SAMPLES,
    EEG_ONLINE_CALIBRATION_MIN_BATCH_SIZE,
    EEG_ONLINE_CALIBRATION_STEPS_PER_UPDATE,
    EEG_ONLINE_CALIBRATION_UPDATE_EVERY,
    EEG_STILLNESS_ALLOWED_PREDICTED_VELOCITY,
    EEG_STILLNESS_TARGET_VELOCITY_THRESHOLD,
)
from streaming.dataset import save_paired_frames
from streaming.records import PairedTrainingFrame

from .model import (
    EegPoseVAE,
    preprocessing_signature_from_config,
    context_from_history,
    save_model,
    set_trainable_scope,
    standardize_pose_latents,
    transform_eeg_for_model,
    unstandardize_pose_latents,
)
from .personalization import ReadinessMetrics, ReadinessResult, score_readiness


@dataclass(frozen=True, slots=True)
class OnlineCalibrationStatus:
    trusted_samples: int
    skipped_samples: int
    update_count: int
    readiness: ReadinessResult
    latest_loss: float | None


class OnlineEegCalibrator:
    """Batched-online session adaptation from trusted paired EEG/pose frames."""

    def __init__(
        self,
        model: EegPoseVAE,
        pose_decoder: nn.Module,
        *,
        device: str | torch.device = DEFAULT_DEVICE,
        adaptation_mode: str = EEG_ADAPTATION_MODE_SESSION,
        learning_rate: float = EEG_MODEL_LR,
        batch_size: int = EEG_ONLINE_CALIBRATION_BATCH_SIZE,
        min_batch_size: int = EEG_ONLINE_CALIBRATION_MIN_BATCH_SIZE,
        update_every: int = EEG_ONLINE_CALIBRATION_UPDATE_EVERY,
        steps_per_update: int = EEG_ONLINE_CALIBRATION_STEPS_PER_UPDATE,
        max_samples: int = EEG_ONLINE_CALIBRATION_MAX_SAMPLES,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.pose_decoder = pose_decoder.to(self.device)
        self.pose_decoder.eval()
        for parameter in self.pose_decoder.parameters():
            parameter.requires_grad_(False)

        set_trainable_scope(self.model, adaptation_mode)
        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        assert trainable_parameters
        self.optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.update_every = update_every
        self.steps_per_update = steps_per_update
        self.max_samples = max_samples
        self.beta = self.model.config.beta
        self.reconstruction_weight = self.model.config.reconstruction_weight

        self._feature_history: list[np.ndarray] = []
        self._eeg_contexts: list[np.ndarray] = []
        self._target_latents: list[np.ndarray] = []
        self._trusted_frames: list[PairedTrainingFrame] = []
        self._decoded_pose_errors: list[float] = []
        self._predicted_velocity: list[float] = []
        self._target_velocity: list[float] = []
        self._trusted_since_update = 0
        self.skipped_samples = 0
        self.update_count = 0
        self.latest_loss: float | None = None

    @property
    def trusted_sample_count(self) -> int:
        return len(self._trusted_frames)

    def observe(self, frame: PairedTrainingFrame) -> bool:
        eeg_features = transform_eeg_for_model(frame.eeg[None, :, :], self.model.config)[0]
        self._feature_history.append(eeg_features)
        self._feature_history = self._feature_history[
            -self.model.config.context_packet_count :
        ]
        context = context_from_history(
            self._feature_history,
            self.model.config.context_packet_count,
        )

        if not self._is_trusted(frame):
            self.skipped_samples += 1
            return False

        self._eeg_contexts.append(context)
        self._target_latents.append(frame.pose_latent.astype(np.float32, copy=False))
        self._trusted_frames.append(frame)
        self._trim_buffers()
        self._record_model_metrics(context, frame.pose_latent)
        self._trusted_since_update += 1

        if (
            self.trusted_sample_count >= self.min_batch_size
            and self._trusted_since_update >= self.update_every
        ):
            self.train_update()
        return True

    def train_update(self) -> float:
        sample_count = len(self._eeg_contexts)
        assert sample_count >= self.min_batch_size
        self.model.train()
        losses = []
        for _ in range(self.steps_per_update):
            indices = torch.randperm(sample_count)[
                : min(self.batch_size, sample_count)
            ].tolist()
            eeg = torch.from_numpy(
                np.stack([self._eeg_contexts[i] for i in indices])
            ).to(self.device)
            target_raw_np = np.stack([self._target_latents[i] for i in indices])
            target_raw = torch.from_numpy(target_raw_np).to(self.device)
            target_model = torch.from_numpy(
                standardize_pose_latents(target_raw_np, self.model.config)
            ).to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            predicted, reconstruction, mean, log_variance = self.model(eeg)
            prediction_loss = self.loss_fn(predicted, target_model)
            reconstruction_target = self.model.adapt_eeg_features(eeg).detach()
            reconstruction_loss = nn.functional.mse_loss(
                reconstruction,
                reconstruction_target,
            )
            kl_loss = -0.5 * torch.mean(
                1.0 + log_variance - mean.pow(2) - log_variance.exp()
            )
            loss = (
                prediction_loss
                + self.reconstruction_weight * reconstruction_loss
                + self.beta * kl_loss
            )
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))

        self.model.eval()
        self._trusted_since_update = 0
        self.update_count += 1
        self.latest_loss = float(np.mean(losses))
        return self.latest_loss

    def status(self) -> OnlineCalibrationStatus:
        return OnlineCalibrationStatus(
            trusted_samples=self.trusted_sample_count,
            skipped_samples=self.skipped_samples,
            update_count=self.update_count,
            readiness=self.readiness(),
            latest_loss=self.latest_loss,
        )

    def readiness(self) -> ReadinessResult:
        if not self._trusted_frames:
            metrics = ReadinessMetrics(
                trusted_sample_count=0,
                mean_pose_confidence=0.0,
                mean_interpolation_confidence=0.0,
                mean_pose_reconstruction_error=float("inf"),
            )
            return score_readiness(metrics)

        pose_confidence = [frame.pose_confidence for frame in self._trusted_frames]
        interpolation_confidence = [
            frame.interpolation_confidence for frame in self._trusted_frames
        ]
        reconstruction_error = [
            frame.pose_reconstruction_error for frame in self._trusted_frames
        ]
        metrics = ReadinessMetrics(
            trusted_sample_count=len(self._trusted_frames),
            mean_pose_confidence=float(np.mean(pose_confidence)),
            mean_interpolation_confidence=float(np.mean(interpolation_confidence)),
            mean_pose_reconstruction_error=float(np.mean(reconstruction_error)),
            decoded_pose_error=(
                float(np.mean(self._decoded_pose_errors))
                if self._decoded_pose_errors
                else None
            ),
            stationary_false_positive_score=self._stationary_false_positive_score(),
            movement_response_score=self._movement_response_score(),
        )
        return score_readiness(metrics)

    def save(self, session_dir: str | Path) -> None:
        session_path = Path(session_dir)
        session_path.mkdir(parents=True, exist_ok=True)
        save_model(session_path / "session_model.pt", self.model)
        if self._trusted_frames:
            save_paired_frames(
                session_path / "trusted_calibration_samples.npz",
                self._trusted_frames,
                metadata={
                    **preprocessing_signature_from_config(self.model.config),
                    "sample_kind": "online_trusted_calibration",
                },
            )
        status = self.status()
        (session_path / "online_status.json").write_text(
            json.dumps(
                {
                    "trusted_samples": status.trusted_samples,
                    "skipped_samples": status.skipped_samples,
                    "update_count": status.update_count,
                    "latest_loss": status.latest_loss,
                    "readiness_score": status.readiness.score,
                    "ready": status.readiness.ready,
                    "readiness_metrics": asdict(status.readiness.metrics),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _is_trusted(self, frame: PairedTrainingFrame) -> bool:
        return (
            frame.pose_confidence >= EEG_CALIBRATION_MIN_POSE_CONFIDENCE
            and frame.interpolation_confidence >= EEG_CALIBRATION_MIN_INTERPOLATION_CONFIDENCE
            and frame.pose_reconstruction_error <= EEG_CALIBRATION_MAX_POSE_RECONSTRUCTION_ERROR
        )

    def _trim_buffers(self) -> None:
        if len(self._trusted_frames) <= self.max_samples:
            return
        extra = len(self._trusted_frames) - self.max_samples
        del self._eeg_contexts[:extra]
        del self._target_latents[:extra]
        del self._trusted_frames[:extra]
        del self._decoded_pose_errors[:extra]
        del self._predicted_velocity[:extra]
        del self._target_velocity[:extra]

    def _record_model_metrics(
        self,
        eeg_context: np.ndarray,
        target_latent: np.ndarray,
    ) -> None:
        self.model.eval()
        eeg = torch.from_numpy(eeg_context[None, ...]).to(self.device)
        target = torch.from_numpy(target_latent[None, ...]).to(self.device)
        with torch.no_grad():
            predicted_model = self.model.predict_pose_latent(eeg)
            predicted = unstandardize_pose_latents(
                predicted_model,
                self.model.config,
            )
            predicted_features = self.pose_decoder.decode(predicted)
            target_features = self.pose_decoder.decode(target)

        predicted_np = predicted_features.cpu().numpy()
        target_np = target_features.cpu().numpy()
        position_error = np.linalg.norm(
            predicted_np[:, :24].reshape(-1, 8, 3)
            - target_np[:, :24].reshape(-1, 8, 3),
            axis=2,
        ).mean()
        self._decoded_pose_errors.append(float(position_error))
        self._predicted_velocity.append(_mean_decoded_velocity(predicted_np))
        self._target_velocity.append(_mean_decoded_velocity(target_np))

    def _stationary_false_positive_score(self) -> float:
        if not self._target_velocity:
            return 0.0
        predicted = np.asarray(self._predicted_velocity, dtype=np.float32)
        target = np.asarray(self._target_velocity, dtype=np.float32)
        still = target < EEG_STILLNESS_TARGET_VELOCITY_THRESHOLD
        if not np.any(still):
            return 0.0
        excess = np.maximum(
            0.0,
            predicted[still] - EEG_STILLNESS_ALLOWED_PREDICTED_VELOCITY,
        )
        return float(np.mean(excess))

    def _movement_response_score(self) -> float:
        if len(self._target_velocity) < 2:
            return 0.0
        predicted = np.asarray(self._predicted_velocity, dtype=np.float32)
        target = np.asarray(self._target_velocity, dtype=np.float32)
        if np.std(predicted) <= 1e-8 or np.std(target) <= 1e-8:
            return 0.0
        return max(0.0, float(np.corrcoef(predicted, target)[0, 1]))


def _mean_decoded_velocity(features: np.ndarray) -> float:
    if features.shape[1] < 48:
        return 0.0
    velocity = features[:, 24:48].reshape(-1, 8, 3)
    return float(np.linalg.norm(velocity, axis=2).mean())
