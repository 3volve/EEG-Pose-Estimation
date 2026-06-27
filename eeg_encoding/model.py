from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    ROOT_DIR,
    DEFAULT_DEVICE,
    EEG_CONTEXT_PACKET_COUNT,
    EEG_DECODED_POSITION_LOSS_WEIGHT,
    EEG_DECODED_VELOCITY_LOSS_WEIGHT,
    EEG_MODEL_BATCH_SIZE,
    EEG_MODEL_BETA,
    EEG_MODEL_EPOCHS,
    EEG_MODEL_HIDDEN_DIM,
    EEG_MODEL_LATENT_DIM,
    EEG_MODEL_LR,
    EEG_MODEL_RECONSTRUCTION_WEIGHT,
    EEG_MODEL_SEED,
    EEG_MODEL_VAL_SPLIT,
    EEG_USE_BAND_ADAPTER,
    EEG_POSITION_LANDMARK_WEIGHTS,
    EEG_STANDARDIZE_POSE_LATENTS,
    EEG_STILLNESS_ALLOWED_PREDICTED_VELOCITY,
    EEG_STILLNESS_LOSS_WEIGHT,
    EEG_STILLNESS_TARGET_VELOCITY_THRESHOLD,
    EEG_VALIDATE_BY_RUN,
    EEG_VELOCITY_LANDMARK_WEIGHTS,
    EEG_WAVELET,
    EEG_WAVELET_LEVEL,
    EEG_WAVELET_MODE,
    EEG_WAVELET_STANDARDIZE_INPUT,
    EEG_ADAPTATION_MODE_ADAPTER_HEAD,
    EEG_ADAPTATION_MODE_ADAPTER_ONLY,
    EEG_ADAPTATION_MODE_PROFILE_CORE,
    EEG_ADAPTATION_MODE_PROFILE_ENCODER,
    EEG_ADAPTATION_MODE_PROFILE_FULL,
    EEG_ADAPTATION_MODE_PROFILE_HEAD,
    EEG_ADAPTATION_MODE_SESSION,
    EEG_ADAPTATION_MODE_SESSION_DEEP,
    POSE_ENCODING_MODEL,
)
from streaming.eeg import EegPacket

from .records import PredictedPoseLatentFrame


@dataclass(frozen=True, slots=True)
class EegPoseModelConfig:
    n_channels: int
    n_samples: int
    eeg_feature_count: int
    pose_latent_dim: int
    context_packet_count: int = EEG_CONTEXT_PACKET_COUNT
    model_latent_dim: int = EEG_MODEL_LATENT_DIM
    hidden_dim: int = EEG_MODEL_HIDDEN_DIM
    beta: float = EEG_MODEL_BETA
    reconstruction_weight: float = EEG_MODEL_RECONSTRUCTION_WEIGHT
    standardize_pose_latents: bool = EEG_STANDARDIZE_POSE_LATENTS
    pose_latent_mean: tuple[float, ...] | None = None
    pose_latent_std: tuple[float, ...] | None = None
    use_wavelet: bool = True
    wavelet: str = EEG_WAVELET
    wavelet_level: int = EEG_WAVELET_LEVEL
    wavelet_mode: str = EEG_WAVELET_MODE
    standardize_input: bool = EEG_WAVELET_STANDARDIZE_INPUT
    use_band_adapter: bool = EEG_USE_BAND_ADAPTER
    wavelet_band_lengths: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class EegTrainingSplitReport:
    n_samples: int
    pose_mae: float
    pose_rmse: float
    baseline_pose_mae: float
    mae_vs_baseline: float
    pose_r2: float
    cosine_similarity: float
    eeg_reconstruction_mse: float
    kl_loss: float


@dataclass(frozen=True, slots=True)
class EegTrainingReport:
    dataset_path: str
    checkpoint_path: str | None
    output_path: str
    pose_checkpoint_path: str | None
    epochs: int
    batch_size: int
    learning_rate: float
    context_packet_count: int
    standardize_pose_latents: bool
    validate_by_run: bool
    decoded_position_weight: float
    decoded_velocity_weight: float
    stillness_weight: float
    adaptation_mode: str | None
    train: EegTrainingSplitReport
    validation: EegTrainingSplitReport | None


class EegBandAdapter(nn.Module):
    """Per-channel, per-wavelet-band affine adapter for calibration drift."""

    def __init__(self, n_channels: int, band_lengths: tuple[int, ...]) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.band_lengths = band_lengths
        self.scale = nn.Parameter(torch.ones(n_channels, len(band_lengths)))
        self.bias = nn.Parameter(torch.zeros(n_channels, len(band_lengths)))

    def forward(self, features: Tensor) -> Tensor:
        assert features.ndim in (3, 4)
        adapted_chunks = []
        start = 0
        for band_index, band_length in enumerate(self.band_lengths):
            stop = start + band_length
            chunk = features[..., start:stop]
            if features.ndim == 3:
                scale = self.scale[:, band_index].view(1, -1, 1)
                bias = self.bias[:, band_index].view(1, -1, 1)
                adapted_chunks.append(chunk * scale + bias)
            else:
                scale = self.scale[:, band_index].view(1, 1, -1, 1)
                bias = self.bias[:, band_index].view(1, 1, -1, 1)
                adapted_chunks.append(chunk * scale + bias)
            start = stop
        assert start == features.shape[-1]
        return torch.cat(adapted_chunks, dim=-1)


class EegPoseVAE(nn.Module):
    """Compact VAE-shaped EEG model that predicts pose autoencoder latents."""

    def __init__(self, config: EegPoseModelConfig) -> None:
        super().__init__()
        self.config = config
        band_lengths = config.wavelet_band_lengths or (config.eeg_feature_count,)
        self.band_adapter = (
            EegBandAdapter(config.n_channels, band_lengths)
            if config.use_band_adapter
            else nn.Identity()
        )
        input_dim = (
            config.context_packet_count
            * config.n_channels
            * config.eeg_feature_count
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.latent_mean = nn.Linear(config.hidden_dim, config.model_latent_dim)
        self.latent_log_variance = nn.Linear(
            config.hidden_dim,
            config.model_latent_dim,
        )
        self.pose_head = nn.Linear(config.model_latent_dim, config.pose_latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.model_latent_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, input_dim),
        )

    def adapt_eeg_features(self, eeg_features: Tensor) -> Tensor:
        return self.band_adapter(eeg_features)

    def forward(self, eeg_features: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        adapted_features = self.adapt_eeg_features(eeg_features)
        flat = adapted_features.flatten(start_dim=1)
        hidden = self.encoder(flat)
        mean = self.latent_mean(hidden)
        log_variance = torch.clamp(self.latent_log_variance(hidden), -12.0, 8.0)
        std = torch.exp(0.5 * log_variance)
        z = mean + std * torch.randn_like(std)
        pose_latent = self.pose_head(z)
        reconstruction = self.decoder(z).reshape_as(adapted_features)
        return pose_latent, reconstruction, mean, log_variance

    def predict_pose_latent(self, eeg_features: Tensor) -> Tensor:
        adapted_features = self.adapt_eeg_features(eeg_features)
        flat = adapted_features.flatten(start_dim=1)
        hidden = self.encoder(flat)
        mean = self.latent_mean(hidden)
        return self.pose_head(mean)


def set_trainable_scope(model: EegPoseVAE, mode: str | None) -> None:
    if mode is None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if mode == EEG_ADAPTATION_MODE_PROFILE_CORE:
        _set_module_trainable(model.encoder, True)
        _set_module_trainable(model.latent_mean, True)
        _set_module_trainable(model.latent_log_variance, True)
        _set_module_trainable(model.pose_head, True)
        return

    _set_module_trainable(model.band_adapter, True)

    if mode == EEG_ADAPTATION_MODE_ADAPTER_ONLY:
        return

    if mode == EEG_ADAPTATION_MODE_ADAPTER_HEAD:
        _set_module_trainable(model.pose_head, True)
        return

    _set_module_trainable(model.latent_mean, True)
    _set_module_trainable(model.latent_log_variance, True)
    _set_module_trainable(model.pose_head, True)

    if mode == EEG_ADAPTATION_MODE_SESSION_DEEP:
        _set_encoder_linear_trainable(model, linear_index=1, trainable=True)
    elif mode == EEG_ADAPTATION_MODE_PROFILE_ENCODER:
        _set_module_trainable(model.encoder, True)
    elif mode == EEG_ADAPTATION_MODE_PROFILE_FULL:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif mode == EEG_ADAPTATION_MODE_PROFILE_HEAD:
        pass
    elif mode == EEG_ADAPTATION_MODE_SESSION:
        pass
    else:
        raise ValueError(f"Unknown EEG adaptation mode: {mode}")


def trainable_parameter_names(model: EegPoseVAE) -> tuple[str, ...]:
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def reset_band_adapter_identity(model: EegPoseVAE) -> None:
    if isinstance(model.band_adapter, EegBandAdapter):
        with torch.no_grad():
            model.band_adapter.scale.fill_(1.0)
            model.band_adapter.bias.zero_()


def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def _set_encoder_linear_trainable(
    model: EegPoseVAE,
    *,
    linear_index: int,
    trainable: bool,
) -> None:
    seen = 0
    for module in model.encoder:
        if isinstance(module, nn.Linear):
            if seen == linear_index:
                _set_module_trainable(module, trainable)
                return
            seen += 1
    raise AssertionError(f"Encoder does not have linear layer index {linear_index}")


class EegPosePredictor:
    def __init__(
        self,
        model: EegPoseVAE,
        *,
        device: str | torch.device = DEFAULT_DEVICE,
        training_report: EegTrainingReport | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.training_report = training_report
        self._feature_history: list[np.ndarray] = []

    def predict(self, packet: EegPacket) -> PredictedPoseLatentFrame:
        eeg_features = transform_eeg_for_model(
            packet.samples[None, :, :],
            self.model.config,
        )
        self._feature_history.append(eeg_features[0])
        self._feature_history = self._feature_history[
            -self.model.config.context_packet_count :
        ]
        context_features = context_from_history(
            self._feature_history,
            self.model.config.context_packet_count,
        )
        eeg = torch.from_numpy(context_features[None, ...]).to(self.device)
        with torch.no_grad():
            predicted = self.model.predict_pose_latent(eeg)
        predicted = unstandardize_pose_latents(predicted, self.model.config)
        return PredictedPoseLatentFrame(
            packet_id=packet.packet_id,
            target_time_s=packet.end_time_s,
            predicted_latent=predicted.cpu().numpy()[0].astype(np.float32, copy=False),
        )


def transform_eeg_for_model(
    raw_eeg: np.ndarray,
    config: EegPoseModelConfig,
) -> np.ndarray:
    assert raw_eeg.ndim == 3
    assert raw_eeg.shape[1] == config.n_channels
    assert raw_eeg.shape[2] == config.n_samples

    eeg = raw_eeg.astype(np.float32, copy=False)
    if config.standardize_input:
        mean = eeg.mean(axis=2, keepdims=True)
        std = eeg.std(axis=2, keepdims=True)
        eeg = (eeg - mean) / np.where(std > 1e-6, std, 1.0)

    if not config.use_wavelet:
        return eeg

    import pywt

    transformed = np.empty(
        (eeg.shape[0], eeg.shape[1], config.eeg_feature_count),
        dtype=np.float32,
    )
    for sample_index in range(eeg.shape[0]):
        for channel_index in range(eeg.shape[1]):
            coefficients = pywt.wavedec(
                eeg[sample_index, channel_index],
                config.wavelet,
                mode=config.wavelet_mode,
                level=config.wavelet_level,
            )
            transformed[sample_index, channel_index] = np.concatenate(
                [coefficient.astype(np.float32, copy=False) for coefficient in coefficients]
            )
    return transformed


def transformed_eeg_feature_count(
    *,
    n_samples: int,
    use_wavelet: bool,
    wavelet: str,
    wavelet_level: int,
    wavelet_mode: str,
) -> int:
    if not use_wavelet:
        return n_samples

    import pywt

    coefficients = pywt.wavedec(
        np.zeros(n_samples, dtype=np.float32),
        wavelet,
        mode=wavelet_mode,
        level=wavelet_level,
    )
    return sum(len(coefficient) for coefficient in coefficients)


def transformed_eeg_band_lengths(
    *,
    n_samples: int,
    use_wavelet: bool,
    wavelet: str,
    wavelet_level: int,
    wavelet_mode: str,
) -> tuple[int, ...]:
    if not use_wavelet:
        return (n_samples,)

    import pywt

    coefficients = pywt.wavedec(
        np.zeros(n_samples, dtype=np.float32),
        wavelet,
        mode=wavelet_mode,
        level=wavelet_level,
    )
    return tuple(len(coefficient) for coefficient in coefficients)


def context_from_history(
    feature_history: list[np.ndarray],
    context_packet_count: int,
) -> np.ndarray:
    assert feature_history
    missing_count = context_packet_count - len(feature_history)
    if missing_count > 0:
        padded = [feature_history[0]] * missing_count + feature_history
    else:
        padded = feature_history[-context_packet_count:]
    return np.stack(padded, axis=0).astype(np.float32, copy=False)


def build_context_windows(
    eeg_features: np.ndarray,
    *,
    context_packet_count: int,
) -> np.ndarray:
    assert eeg_features.ndim == 3
    windows = [
        context_from_history(
            [eeg_features[j] for j in range(max(0, i - context_packet_count + 1), i + 1)],
            context_packet_count,
        )
        for i in range(len(eeg_features))
    ]
    return np.stack(windows, axis=0).astype(np.float32, copy=False)


def standardize_pose_latents(
    pose_latents: np.ndarray,
    config: EegPoseModelConfig,
) -> np.ndarray:
    if not config.standardize_pose_latents:
        return pose_latents.astype(np.float32, copy=False)
    assert config.pose_latent_mean is not None
    assert config.pose_latent_std is not None
    mean = np.asarray(config.pose_latent_mean, dtype=np.float32)
    std = np.asarray(config.pose_latent_std, dtype=np.float32)
    return ((pose_latents - mean[None, :]) / std[None, :]).astype(
        np.float32,
        copy=False,
    )


def unstandardize_pose_latents(
    pose_latents: Tensor,
    config: EegPoseModelConfig,
) -> Tensor:
    if not config.standardize_pose_latents:
        return pose_latents
    assert config.pose_latent_mean is not None
    assert config.pose_latent_std is not None
    mean = torch.tensor(
        config.pose_latent_mean,
        dtype=pose_latents.dtype,
        device=pose_latents.device,
    )
    std = torch.tensor(
        config.pose_latent_std,
        dtype=pose_latents.dtype,
        device=pose_latents.device,
    )
    return pose_latents * std + mean


def decoded_pose_training_loss(
    predicted_latent: Tensor,
    target_latent: Tensor,
    pose_decoder: nn.Module,
    *,
    position_weight: float,
    velocity_weight: float,
    stillness_weight: float,
    target_velocity_threshold: float,
    allowed_predicted_velocity: float,
    position_landmark_weights: Tensor,
    velocity_landmark_weights: Tensor,
) -> Tensor:
    if position_weight <= 0 and velocity_weight <= 0 and stillness_weight <= 0:
        return predicted_latent.new_zeros(())

    with torch.no_grad():
        target_features = pose_decoder.decode(target_latent)
    predicted_features = pose_decoder.decode(predicted_latent)

    total = predicted_latent.new_zeros(())
    predicted_positions = predicted_features[:, :24].reshape(-1, 8, 3)
    target_positions = target_features[:, :24].reshape(-1, 8, 3)
    if position_weight > 0:
        position_error = nn.functional.smooth_l1_loss(
            predicted_positions,
            target_positions,
            reduction="none",
        )
        total = total + position_weight * torch.mean(
            position_error * position_landmark_weights
        )

    if predicted_features.shape[1] >= 48:
        predicted_velocity = predicted_features[:, 24:48].reshape(-1, 8, 3)
        target_velocity = target_features[:, 24:48].reshape(-1, 8, 3)
        if velocity_weight > 0:
            velocity_error = nn.functional.smooth_l1_loss(
                predicted_velocity,
                target_velocity,
                reduction="none",
            )
            total = total + velocity_weight * torch.mean(
                velocity_error * velocity_landmark_weights
            )
        if stillness_weight > 0:
            target_velocity_mag = torch.linalg.vector_norm(
                target_velocity,
                dim=2,
            ).mean(dim=1)
            predicted_velocity_mag = torch.linalg.vector_norm(
                predicted_velocity,
                dim=2,
            ).mean(dim=1)
            still_mask = target_velocity_mag < target_velocity_threshold
            if torch.any(still_mask):
                excess_motion = torch.relu(
                    predicted_velocity_mag[still_mask] - allowed_predicted_velocity
                )
                total = total + stillness_weight * torch.mean(excess_motion ** 2)

    return total


def _load_pose_decoder(
    pose_checkpoint: str | Path | None,
    *,
    pose_latent_dim: int,
    device: str | torch.device,
) -> nn.Module | None:
    if pose_checkpoint is None:
        return None

    from pose_encoding.pose_autoencoder import load_checkpoint

    pose_decoder, _ = load_checkpoint(pose_checkpoint, map_location=device)
    if pose_decoder.latent_dim != pose_latent_dim:
        raise ValueError(
            "Pose decoder checkpoint does not match the paired pose latents: "
            f"checkpoint expects {pose_decoder.latent_dim}, "
            f"dataset has {pose_latent_dim}."
        )
    for parameter in pose_decoder.parameters():
        parameter.requires_grad_(False)
    pose_decoder.eval()
    return pose_decoder


def _landmark_weight_tensor(
    values: tuple[float, ...],
    *,
    device: str | torch.device,
) -> Tensor:
    weights = torch.tensor(values, dtype=torch.float32, device=device)
    assert weights.shape == (8,)
    return weights.reshape(1, 8, 1)


def train_model(
    dataset_path: str | Path | list[str | Path],
    out_path: str | Path,
    *,
    epochs: int = EEG_MODEL_EPOCHS,
    batch_size: int = EEG_MODEL_BATCH_SIZE,
    learning_rate: float = EEG_MODEL_LR,
    model_latent_dim: int = EEG_MODEL_LATENT_DIM,
    hidden_dim: int = EEG_MODEL_HIDDEN_DIM,
    beta: float = EEG_MODEL_BETA,
    reconstruction_weight: float = EEG_MODEL_RECONSTRUCTION_WEIGHT,
    context_packet_count: int = EEG_CONTEXT_PACKET_COUNT,
    standardize_pose_latents_for_training: bool = EEG_STANDARDIZE_POSE_LATENTS,
    validate_by_run: bool = EEG_VALIDATE_BY_RUN,
    pose_checkpoint: str | Path | None = POSE_ENCODING_MODEL,
    decoded_position_weight: float = EEG_DECODED_POSITION_LOSS_WEIGHT,
    decoded_velocity_weight: float = EEG_DECODED_VELOCITY_LOSS_WEIGHT,
    stillness_weight: float = EEG_STILLNESS_LOSS_WEIGHT,
    stillness_target_velocity_threshold: float = EEG_STILLNESS_TARGET_VELOCITY_THRESHOLD,
    stillness_allowed_predicted_velocity: float = EEG_STILLNESS_ALLOWED_PREDICTED_VELOCITY,
    wavelet: str = EEG_WAVELET,
    wavelet_level: int = EEG_WAVELET_LEVEL,
    wavelet_mode: str = EEG_WAVELET_MODE,
    standardize_input: bool = EEG_WAVELET_STANDARDIZE_INPUT,
    use_band_adapter: bool = EEG_USE_BAND_ADAPTER,
    adaptation_mode: str | None = None,
    device: str | torch.device = DEFAULT_DEVICE,
    seed: int = EEG_MODEL_SEED,
    val_split: float = EEG_MODEL_VAL_SPLIT,
    checkpoint_path: str | Path | None = None,
) -> EegPosePredictor:
    if not isinstance(dataset_path, list):
        dataset_path = [dataset_path]

    out_path = _resolve_model_path(out_path)
    checkpoint_path = (
        _resolve_model_path(checkpoint_path)
        if checkpoint_path is not None
        else None
    )
    eeg_parts = []
    pose_latent_parts = []
    group_parts = []
    resolved_dataset_paths = []
    for resolved_path in _resolve_dataset_paths(dataset_path):
        resolved_dataset_paths.append(str(resolved_path))
        with np.load(resolved_path) as dataset:
            eeg_parts.append(np.asarray(dataset["eeg"], dtype=np.float32))
            pose_latent_parts.append(np.asarray(dataset["pose_latent"], dtype=np.float32))
            group_parts.append(
                np.full(len(dataset["eeg"]), resolved_path.name, dtype=object)
            )
    raw_eeg = np.concatenate(eeg_parts, axis=0)
    pose_latent = np.concatenate(pose_latent_parts, axis=0)
    group_ids = np.concatenate(group_parts, axis=0)
    train_indices, validation_indices = (
        _split_indices_by_group(group_ids, val_split, seed)
        if validate_by_run
        else _split_indices(len(group_ids), val_split, seed)
    )

    torch.manual_seed(seed)
    if checkpoint_path is None:
        feature_count = transformed_eeg_feature_count(
            n_samples=raw_eeg.shape[2],
            use_wavelet=True,
            wavelet=wavelet,
            wavelet_level=wavelet_level,
            wavelet_mode=wavelet_mode,
        )
        band_lengths = transformed_eeg_band_lengths(
            n_samples=raw_eeg.shape[2],
            use_wavelet=True,
            wavelet=wavelet,
            wavelet_level=wavelet_level,
            wavelet_mode=wavelet_mode,
        )
        config = EegPoseModelConfig(
            n_channels=raw_eeg.shape[1],
            n_samples=raw_eeg.shape[2],
            eeg_feature_count=feature_count,
            pose_latent_dim=pose_latent.shape[1],
            context_packet_count=context_packet_count,
            model_latent_dim=model_latent_dim,
            hidden_dim=hidden_dim,
            beta=beta,
            reconstruction_weight=reconstruction_weight,
            standardize_pose_latents=standardize_pose_latents_for_training,
            pose_latent_mean=(
                tuple(float(value) for value in pose_latent[train_indices].mean(axis=0))
                if standardize_pose_latents_for_training
                else None
            ),
            pose_latent_std=(
                tuple(
                    float(value)
                    for value in np.maximum(
                        pose_latent[train_indices].std(axis=0),
                        1e-6,
                    )
                )
                if standardize_pose_latents_for_training
                else None
            ),
            use_wavelet=True,
            wavelet=wavelet,
            wavelet_level=wavelet_level,
            wavelet_mode=wavelet_mode,
            standardize_input=standardize_input,
            use_band_adapter=use_band_adapter,
            wavelet_band_lengths=band_lengths,
        )
        model = EegPoseVAE(config).to(device)
    else:
        model = _load_raw_model(checkpoint_path, device)
        config = model.config
        if (
            config.n_channels != raw_eeg.shape[1]
            or config.n_samples != raw_eeg.shape[2]
            or config.pose_latent_dim != pose_latent.shape[1]
        ):
            raise ValueError(
                "Checkpoint shape does not match the paired dataset: "
                f"checkpoint expects EEG ({config.n_channels}, {config.n_samples}) "
                f"and pose latent {config.pose_latent_dim}; dataset has EEG "
                f"({raw_eeg.shape[1]}, {raw_eeg.shape[2]}) and pose latent {pose_latent.shape[1]}."
            )
    transformed_parts = [
        build_context_windows(
            transform_eeg_for_model(raw_part, config),
            context_packet_count=config.context_packet_count,
        )
        for raw_part in eeg_parts
    ]
    eeg = np.concatenate(transformed_parts, axis=0)
    pose_latent_model_target = standardize_pose_latents(pose_latent, config)
    pose_decoder = _load_pose_decoder(
        pose_checkpoint,
        pose_latent_dim=pose_latent.shape[1],
        device=device,
    )
    position_landmark_weights = _landmark_weight_tensor(
        EEG_POSITION_LANDMARK_WEIGHTS,
        device=device,
    )
    velocity_landmark_weights = _landmark_weight_tensor(
        EEG_VELOCITY_LANDMARK_WEIGHTS,
        device=device,
    )

    set_trainable_scope(model, adaptation_mode)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert trainable_parameters
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    data = TensorDataset(
        torch.from_numpy(eeg[train_indices]),
        torch.from_numpy(pose_latent_model_target[train_indices]),
        torch.from_numpy(pose_latent[train_indices]),
    )
    loader = DataLoader(
        data,
        batch_size=min(batch_size, len(data)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    for _ in range(epochs):
        model.train()
        for batch_eeg, batch_pose_latent_target, batch_pose_latent_raw in loader:
            batch_eeg = batch_eeg.to(device)
            batch_pose_latent_target = batch_pose_latent_target.to(device)
            batch_pose_latent_raw = batch_pose_latent_raw.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted, reconstruction, mean, log_variance = model(batch_eeg)
            prediction_loss = loss_fn(predicted, batch_pose_latent_target)
            reconstruction_target = model.adapt_eeg_features(batch_eeg).detach()
            reconstruction_loss = nn.functional.mse_loss(
                reconstruction,
                reconstruction_target,
            )
            predicted_raw = unstandardize_pose_latents(predicted, config)
            decoded_loss = (
                decoded_pose_training_loss(
                    predicted_raw,
                    batch_pose_latent_raw,
                    pose_decoder,
                    position_weight=decoded_position_weight,
                    velocity_weight=decoded_velocity_weight,
                    stillness_weight=stillness_weight,
                    target_velocity_threshold=stillness_target_velocity_threshold,
                    allowed_predicted_velocity=stillness_allowed_predicted_velocity,
                    position_landmark_weights=position_landmark_weights,
                    velocity_landmark_weights=velocity_landmark_weights,
                )
                if pose_decoder is not None
                else predicted.new_zeros(())
            )
            kl_loss = -0.5 * torch.mean(
                1.0 + log_variance - mean.pow(2) - log_variance.exp()
            )
            loss = (
                prediction_loss
                + decoded_loss
                + reconstruction_weight * reconstruction_loss
                + beta * kl_loss
            )
            loss.backward()
            optimizer.step()

    save_model(out_path, model)
    train_report = _evaluate_split(
        model,
        eeg[train_indices],
        pose_latent[train_indices],
        config=config,
        baseline_pose_latent=pose_latent[train_indices].mean(axis=0),
        device=device,
    )
    validation_report = (
        _evaluate_split(
            model,
            eeg[validation_indices],
            pose_latent[validation_indices],
            config=config,
            baseline_pose_latent=pose_latent[train_indices].mean(axis=0),
            device=device,
        )
        if len(validation_indices) > 0
        else None
    )
    report = EegTrainingReport(
        dataset_path=";".join(resolved_dataset_paths),
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        output_path=str(out_path),
        pose_checkpoint_path=(
            str(pose_checkpoint)
            if pose_decoder is not None and pose_checkpoint is not None
            else None
        ),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        context_packet_count=config.context_packet_count,
        standardize_pose_latents=config.standardize_pose_latents,
        validate_by_run=validate_by_run,
        decoded_position_weight=decoded_position_weight if pose_decoder is not None else 0.0,
        decoded_velocity_weight=decoded_velocity_weight if pose_decoder is not None else 0.0,
        stillness_weight=stillness_weight if pose_decoder is not None else 0.0,
        adaptation_mode=adaptation_mode,
        train=train_report,
        validation=validation_report,
    )
    return EegPosePredictor(model, device=device, training_report=report)


def save_model(path: str | Path, model: EegPoseVAE) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = model.config
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "n_channels": config.n_channels,
                "n_samples": config.n_samples,
                "eeg_feature_count": config.eeg_feature_count,
                "pose_latent_dim": config.pose_latent_dim,
                "context_packet_count": config.context_packet_count,
                "model_latent_dim": config.model_latent_dim,
                "hidden_dim": config.hidden_dim,
                "beta": config.beta,
                "reconstruction_weight": config.reconstruction_weight,
                "standardize_pose_latents": config.standardize_pose_latents,
                "pose_latent_mean": config.pose_latent_mean,
                "pose_latent_std": config.pose_latent_std,
                "use_wavelet": config.use_wavelet,
                "wavelet": config.wavelet,
                "wavelet_level": config.wavelet_level,
                "wavelet_mode": config.wavelet_mode,
                "standardize_input": config.standardize_input,
                "use_band_adapter": config.use_band_adapter,
                "wavelet_band_lengths": config.wavelet_band_lengths,
            },
        },
        output_path,
    )


def load_model(
    path: str | Path,
    *,
    device: str | torch.device = DEFAULT_DEVICE,
) -> EegPosePredictor:
    model = _load_raw_model(path, device)
    return EegPosePredictor(model, device=device)


def _load_raw_model(
    path: str | Path,
    device: str | torch.device = DEFAULT_DEVICE,
) -> EegPoseVAE:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = EegPoseModelConfig(**checkpoint["config"])
    model = EegPoseVAE(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def _resolve_dataset_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    direct_path = ROOT_DIR / path
    if direct_path.exists():
        return direct_path
    return ROOT_DIR / "eeg_encoding" / "data" / path


def _resolve_dataset_paths(paths: list[str | Path]) -> list[Path]:
    resolved_paths: list[Path] = []
    for path in paths:
        resolved_path = _resolve_dataset_path(path)
        matches = sorted(glob.glob(str(resolved_path)))
        if matches:
            resolved_paths.extend(Path(match) for match in matches)
        else:
            resolved_paths.append(resolved_path)
    return resolved_paths


def _resolve_model_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    direct_path = ROOT_DIR / path
    if direct_path.exists() or path.parent != Path("."):
        return direct_path
    return ROOT_DIR / "eeg_encoding" / "models" / path


def _split_indices(
    n_samples: int,
    val_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_samples)
    if n_samples < 3 or val_split <= 0:
        return indices, np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    validation_count = min(max(1, round(n_samples * val_split)), n_samples - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def _split_indices_by_group(
    group_ids: np.ndarray,
    val_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(group_ids))
    unique_groups = np.unique(group_ids)
    if len(unique_groups) < 2 or val_split <= 0:
        return _split_indices(len(group_ids), val_split, seed)

    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(unique_groups)
    validation_group_count = min(
        max(1, round(len(unique_groups) * val_split)),
        len(unique_groups) - 1,
    )
    validation_groups = set(shuffled_groups[:validation_group_count])
    validation_mask = np.asarray(
        [group_id in validation_groups for group_id in group_ids],
        dtype=bool,
    )
    return indices[~validation_mask], indices[validation_mask]


def _evaluate_split(
    model: EegPoseVAE,
    eeg: np.ndarray,
    pose_latent: np.ndarray,
    *,
    config: EegPoseModelConfig,
    baseline_pose_latent: np.ndarray,
    device: str | torch.device,
) -> EegTrainingSplitReport:
    model.eval()
    eeg_tensor = torch.from_numpy(eeg).to(device)
    with torch.no_grad():
        predicted_tensor = model.predict_pose_latent(eeg_tensor)
        predicted = unstandardize_pose_latents(
            predicted_tensor,
            config,
        ).cpu().numpy()
        _, reconstruction, mean, log_variance = model(eeg_tensor)
        reconstruction_np = reconstruction.cpu().numpy()
        reconstruction_target = model.adapt_eeg_features(eeg_tensor).cpu().numpy()
        kl_loss = -0.5 * torch.mean(
            1.0 + log_variance - mean.pow(2) - log_variance.exp()
        )

    error = predicted - pose_latent
    baseline_error = baseline_pose_latent[None, :] - pose_latent
    target_variance = np.sum((pose_latent - pose_latent.mean(axis=0)) ** 2)
    residual_variance = np.sum(error * error)
    norms = np.linalg.norm(predicted, axis=1) * np.linalg.norm(pose_latent, axis=1)
    valid_cosine = norms > 1e-8
    cosine = np.sum(predicted * pose_latent, axis=1)
    cosine_similarity = (
        float(np.mean(cosine[valid_cosine] / norms[valid_cosine]))
        if np.any(valid_cosine)
        else 0.0
    )

    pose_mae = float(np.mean(np.abs(error)))
    baseline_pose_mae = float(np.mean(np.abs(baseline_error)))
    return EegTrainingSplitReport(
        n_samples=len(eeg),
        pose_mae=pose_mae,
        pose_rmse=float(np.sqrt(np.mean(error * error))),
        baseline_pose_mae=baseline_pose_mae,
        mae_vs_baseline=float(pose_mae / baseline_pose_mae) if baseline_pose_mae > 0 else float("inf"),
        pose_r2=(
            float(1.0 - residual_variance / target_variance)
            if target_variance > 0
            else 0.0
        ),
        cosine_similarity=cosine_similarity,
        eeg_reconstruction_mse=float(
            np.mean((reconstruction_np - reconstruction_target) ** 2)
        ),
        kl_loss=float(kl_loss.cpu()),
    )


def format_training_report(report: EegTrainingReport) -> str:
    lines = [
        "EEG encoding training report",
        f"  dataset: {report.dataset_path}",
        f"  output: {report.output_path}",
    ]
    if report.checkpoint_path is not None:
        lines.append(f"  initialized from: {report.checkpoint_path}")
    if report.pose_checkpoint_path is not None:
        lines.append(f"  pose decoder: {report.pose_checkpoint_path}")
    if report.adaptation_mode is not None:
        lines.append(f"  adaptation_mode: {report.adaptation_mode}")
    lines.extend(
        [
            f"  epochs: {report.epochs}",
            f"  batch_size: {report.batch_size}",
            f"  learning_rate: {report.learning_rate:g}",
            f"  context_packets: {report.context_packet_count}",
            f"  standardize_pose_latents: {report.standardize_pose_latents}",
            f"  validate_by_run: {report.validate_by_run}",
            "  decoded loss weights: "
            f"position={report.decoded_position_weight:g}, "
            f"velocity={report.decoded_velocity_weight:g}, "
            f"stillness={report.stillness_weight:g}",
            _format_split_report("train", report.train),
        ]
    )
    if report.validation is not None:
        lines.append(_format_split_report("validation", report.validation))
    else:
        lines.append("  validation: skipped; dataset too small or val_split <= 0")
    return "\n".join(lines)


def _format_split_report(name: str, report: EegTrainingSplitReport) -> str:
    return (
        f"  {name}: n={report.n_samples}, "
        f"pose_mae={report.pose_mae:.6f}, "
        f"pose_rmse={report.pose_rmse:.6f}, "
        f"baseline_mae={report.baseline_pose_mae:.6f}, "
        f"mae/baseline={report.mae_vs_baseline:.3f}, "
        f"r2={report.pose_r2:.3f}, "
        f"cosine={report.cosine_similarity:.3f}, "
        f"eeg_recon_mse={report.eeg_reconstruction_mse:.6f}, "
        f"kl={report.kl_loss:.6f}"
    )
