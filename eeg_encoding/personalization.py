from __future__ import annotations

import copy
import csv
import glob
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DEFAULT_DEVICE,
    EEG_ADAPTATION_MODE_ADAPTER_HEAD,
    EEG_ADAPTATION_MODE_PROFILE_CORE,
    EEG_ADAPTATION_MODE_SESSION,
    EEG_CALIBRATION_MAX_POSE_RECONSTRUCTION_ERROR,
    EEG_CALIBRATION_MAX_DECODED_POSE_ERROR,
    EEG_CALIBRATION_MAX_STATIONARY_FALSE_POSITIVE_SCORE,
    EEG_CALIBRATION_MIN_INTERPOLATION_CONFIDENCE,
    EEG_CALIBRATION_MIN_MOVEMENT_RESPONSE_SCORE,
    EEG_CALIBRATION_MIN_POSE_CONFIDENCE,
    EEG_CALIBRATION_MIN_TRUSTED_SAMPLES,
    EEG_CALIBRATION_READY_SCORE,
    EEG_DECODED_POSITION_LOSS_WEIGHT,
    EEG_DECODED_VELOCITY_LOSS_WEIGHT,
    EEG_MODEL_BATCH_SIZE,
    EEG_MODEL_BETA,
    EEG_MODEL_EPOCHS,
    EEG_MODEL_LR,
    EEG_MODEL_RECONSTRUCTION_WEIGHT,
    EEG_POSITION_LANDMARK_WEIGHTS,
    EEG_PROFILE_BUILD_INNER_EPOCHS,
    EEG_PROFILE_BUILD_MOVEMENT_TOLERANCE,
    EEG_PROFILE_BUILD_QUERY_EPOCHS,
    EEG_PROFILE_BUILD_QUERY_FRACTION,
    EEG_PROFILE_BUILD_REGRESSION_TOLERANCE,
    EEG_PROFILES_ROOT,
    EEG_STILLNESS_ALLOWED_PREDICTED_VELOCITY,
    EEG_STILLNESS_LOSS_WEIGHT,
    EEG_STILLNESS_TARGET_VELOCITY_THRESHOLD,
    EEG_VALIDATE_BY_RUN,
    EEG_VELOCITY_LANDMARK_WEIGHTS,
    POSE_ENCODING_MODEL,
    ROOT_DIR,
)

from .model import (
    EegPoseVAE,
    EegTrainingReport,
    build_context_windows,
    decoded_pose_training_loss,
    format_training_report,
    reset_band_adapter_identity,
    save_model,
    set_trainable_scope,
    standardize_pose_latents,
    train_model,
    transform_eeg_for_model,
    unstandardize_pose_latents,
    _load_pose_decoder,
    _load_raw_model,
)


@dataclass(frozen=True, slots=True)
class ProfilePaths:
    root: Path
    profile_model: Path
    history_root: Path
    sessions_root: Path


@dataclass(frozen=True, slots=True)
class ReadinessMetrics:
    trusted_sample_count: int
    mean_pose_confidence: float
    mean_interpolation_confidence: float
    mean_pose_reconstruction_error: float
    decoded_pose_error: float | None = None
    stationary_false_positive_score: float | None = None
    movement_response_score: float | None = None


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    score: float
    ready: bool
    metrics: ReadinessMetrics


@dataclass(frozen=True, slots=True)
class ProfileSession:
    name: str
    source_paths: tuple[Path, ...]
    eeg: np.ndarray
    pose_latent: np.ndarray
    pose_confidence: np.ndarray
    interpolation_confidence: np.ndarray
    pose_reconstruction_error: np.ndarray
    support_indices: np.ndarray
    query_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class PostAdaptationMetrics:
    session: str
    query_samples: int
    pose_mae: float
    decoded_pose_error: float
    stationary_false_positive_score: float
    movement_response_score: float
    readiness_score: float
    ready: bool


@dataclass(frozen=True, slots=True)
class ProfileBuildReport:
    user_id: str
    profile_model: str
    proposed_model: str
    committed: bool
    start_checkpoint: str
    history_dir: str
    session_count: int
    old_metrics: dict[str, float] | None
    new_metrics: dict[str, float]
    old_session_metrics: tuple[PostAdaptationMetrics, ...] | None
    session_metrics: tuple[PostAdaptationMetrics, ...]


def profile_paths(
    user_id: str,
    *,
    profiles_root: str | Path = EEG_PROFILES_ROOT,
) -> ProfilePaths:
    root = Path(profiles_root) / user_id
    return ProfilePaths(
        root=root,
        profile_model=root / "profile_model.pt",
        history_root=root / "profile_history",
        sessions_root=root / "sessions",
    )


def trusted_calibration_mask(
    archive: np.lib.npyio.NpzFile | dict[str, np.ndarray],
    *,
    min_pose_confidence: float = EEG_CALIBRATION_MIN_POSE_CONFIDENCE,
    min_interpolation_confidence: float = EEG_CALIBRATION_MIN_INTERPOLATION_CONFIDENCE,
    max_pose_reconstruction_error: float = EEG_CALIBRATION_MAX_POSE_RECONSTRUCTION_ERROR,
) -> np.ndarray:
    return (
        (archive["pose_confidence"] >= min_pose_confidence)
        & (archive["interpolation_confidence"] >= min_interpolation_confidence)
        & (archive["pose_reconstruction_error"] <= max_pose_reconstruction_error)
    )


def readiness_from_archive(
    archive_path: str | Path,
    *,
    decoded_pose_error: float | None = None,
    stationary_false_positive_score: float | None = None,
    movement_response_score: float | None = None,
) -> ReadinessResult:
    with np.load(archive_path) as archive:
        trusted_mask = trusted_calibration_mask(archive)
        trusted_count = int(np.sum(trusted_mask))
        if trusted_count > 0:
            pose_confidence = archive["pose_confidence"][trusted_mask]
            interpolation_confidence = archive["interpolation_confidence"][trusted_mask]
            pose_reconstruction_error = archive["pose_reconstruction_error"][trusted_mask]
            metrics = ReadinessMetrics(
                trusted_sample_count=trusted_count,
                mean_pose_confidence=float(np.mean(pose_confidence)),
                mean_interpolation_confidence=float(np.mean(interpolation_confidence)),
                mean_pose_reconstruction_error=float(np.mean(pose_reconstruction_error)),
                decoded_pose_error=decoded_pose_error,
                stationary_false_positive_score=stationary_false_positive_score,
                movement_response_score=movement_response_score,
            )
        else:
            metrics = ReadinessMetrics(
                trusted_sample_count=0,
                mean_pose_confidence=0.0,
                mean_interpolation_confidence=0.0,
                mean_pose_reconstruction_error=float("inf"),
                decoded_pose_error=decoded_pose_error,
                stationary_false_positive_score=stationary_false_positive_score,
                movement_response_score=movement_response_score,
            )
    return score_readiness(metrics)


def score_readiness(metrics: ReadinessMetrics) -> ReadinessResult:
    enough_trusted_samples = (
        metrics.trusted_sample_count >= EEG_CALIBRATION_MIN_TRUSTED_SAMPLES
    )
    if metrics.decoded_pose_error is None:
        return ReadinessResult(score=0.0, ready=False, metrics=metrics)

    decoded_score = max(
        0.0,
        1.0
        - metrics.decoded_pose_error
        / max(EEG_CALIBRATION_MAX_DECODED_POSE_ERROR, 1e-6),
    )
    stationary_score = (
        max(
            0.0,
            1.0
            - metrics.stationary_false_positive_score
            / max(EEG_CALIBRATION_MAX_STATIONARY_FALSE_POSITIVE_SCORE, 1e-6),
        )
        if metrics.stationary_false_positive_score is not None
        else 0.0
    )
    movement_score = (
        min(
            metrics.movement_response_score
            / max(EEG_CALIBRATION_MIN_MOVEMENT_RESPONSE_SCORE, 1e-6),
            1.0,
        )
        if metrics.movement_response_score is not None
        else 0.0
    )
    score = float(np.mean([decoded_score, stationary_score, movement_score]))
    return ReadinessResult(
        score=score,
        ready=enough_trusted_samples and score >= EEG_CALIBRATION_READY_SCORE,
        metrics=metrics,
    )


def train_session_model(
    *,
    user_id: str,
    base_checkpoint: str | Path,
    calibration_data: str | Path | list[str | Path],
    session_id: str | None = None,
    profiles_root: str | Path = EEG_PROFILES_ROOT,
    epochs: int = EEG_MODEL_EPOCHS,
    batch_size: int = EEG_MODEL_BATCH_SIZE,
    learning_rate: float = EEG_MODEL_LR,
    adaptation_mode: str = EEG_ADAPTATION_MODE_SESSION,
    pose_checkpoint: str | Path | None = POSE_ENCODING_MODEL,
    validate_by_run: bool = EEG_VALIDATE_BY_RUN,
    device: str = DEFAULT_DEVICE,
) -> EegTrainingReport:
    paths = profile_paths(user_id, profiles_root=profiles_root)
    session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = paths.sessions_root / session_id
    session_model_path = session_root / "session_model.pt"
    predictor = train_model(
        calibration_data,
        session_model_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_path=base_checkpoint,
        adaptation_mode=adaptation_mode,
        pose_checkpoint=pose_checkpoint,
        validate_by_run=validate_by_run,
        device=device,
    )
    assert predictor.training_report is not None
    _write_json(
        session_root / "readiness_training_report.json",
        {"training_report": format_training_report(predictor.training_report)},
    )
    return predictor.training_report


def should_commit_profile_update(
    old_metrics: dict[str, float] | None,
    new_metrics: dict[str, float],
) -> bool:
    if old_metrics is None:
        return True
    tolerance = EEG_PROFILE_BUILD_REGRESSION_TOLERANCE
    movement_tolerance = EEG_PROFILE_BUILD_MOVEMENT_TOLERANCE
    if (
        "decoded_pose_error" in old_metrics
        and "decoded_pose_error" in new_metrics
        and new_metrics["decoded_pose_error"] > old_metrics["decoded_pose_error"] + tolerance
    ):
        return False
    if (
        "stationary_false_positive_score" in old_metrics
        and "stationary_false_positive_score" in new_metrics
        and new_metrics["stationary_false_positive_score"]
        > old_metrics["stationary_false_positive_score"] + tolerance
    ):
        return False
    if (
        "movement_response_score" in old_metrics
        and "movement_response_score" in new_metrics
        and new_metrics["movement_response_score"]
        < old_metrics["movement_response_score"] - movement_tolerance
    ):
        return False
    if (
        "readiness_score" in old_metrics
        and "readiness_score" in new_metrics
        and new_metrics["readiness_score"] < old_metrics["readiness_score"] - tolerance
    ):
        return False
    return True


def profile_start_checkpoint(
    user_id: str,
    fallback_checkpoint: str | Path,
    *,
    profiles_root: str | Path = EEG_PROFILES_ROOT,
) -> Path:
    paths = profile_paths(user_id, profiles_root=profiles_root)
    return paths.profile_model if paths.profile_model.exists() else Path(fallback_checkpoint)


def install_profile_model(
    user_id: str,
    source_model: str | Path,
    *,
    profiles_root: str | Path = EEG_PROFILES_ROOT,
    update_kind: str,
) -> Path:
    paths = profile_paths(user_id, profiles_root=profiles_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    history_root = paths.history_root / timestamp
    history_root.mkdir(parents=True, exist_ok=True)
    paths.root.mkdir(parents=True, exist_ok=True)
    source_model = Path(source_model)
    history_model = history_root / "profile_model.pt"
    shutil.copy2(source_model, history_model)
    shutil.copy2(source_model, paths.profile_model)
    _write_json(
        history_root / "profile_update.json",
        {
            "update_kind": update_kind,
            "source_model": str(source_model),
            "profile_model": str(paths.profile_model),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return paths.profile_model


def build_profile_model(
    *,
    user_id: str,
    new_session_data: str | Path | list[str | Path],
    base_checkpoint: str | Path,
    profiles_root: str | Path = EEG_PROFILES_ROOT,
    epochs: int = EEG_MODEL_EPOCHS,
    inner_epochs: int = EEG_PROFILE_BUILD_INNER_EPOCHS,
    query_epochs: int = EEG_PROFILE_BUILD_QUERY_EPOCHS,
    batch_size: int = EEG_MODEL_BATCH_SIZE,
    learning_rate: float = EEG_MODEL_LR,
    inner_adaptation_mode: str = EEG_ADAPTATION_MODE_ADAPTER_HEAD,
    pose_checkpoint: str | Path | None = POSE_ENCODING_MODEL,
    query_fraction: float = EEG_PROFILE_BUILD_QUERY_FRACTION,
    device: str | torch.device = DEFAULT_DEVICE,
) -> ProfileBuildReport:
    paths = profile_paths(user_id, profiles_root=profiles_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    history_root = paths.history_root / timestamp
    history_root.mkdir(parents=True, exist_ok=True)
    paths.root.mkdir(parents=True, exist_ok=True)

    _write_profile_session_archives(
        new_session_data,
        history_root=paths.history_root,
        batch_id=timestamp,
    )
    session_roots = _profile_session_roots(paths.history_root)
    sessions = [
        load_profile_session(root, query_fraction=query_fraction)
        for root in session_roots
    ]

    start_checkpoint = profile_start_checkpoint(
        user_id,
        base_checkpoint,
        profiles_root=profiles_root,
    )
    old_metrics = None
    old_session_metrics = None
    if paths.profile_model.exists():
        old_session_metrics = evaluate_profile_post_adaptation(
            paths.profile_model,
            sessions,
            inner_adaptation_mode=inner_adaptation_mode,
            inner_epochs=inner_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            pose_checkpoint=pose_checkpoint,
            device=device,
        )
        old_metrics = summarize_post_adaptation_metrics(old_session_metrics)

    proposed_model = _load_raw_model(start_checkpoint, device)
    reset_band_adapter_identity(proposed_model)
    pose_decoder = _load_pose_decoder(
        pose_checkpoint,
        pose_latent_dim=sessions[0].pose_latent.shape[1],
        device=device,
    )
    for _ in range(epochs):
        for session in sessions:
            session_model = copy.deepcopy(proposed_model)
            reset_band_adapter_identity(session_model)
            _train_model_on_session_split(
                session_model,
                session,
                session.support_indices,
                mode=inner_adaptation_mode,
                epochs=inner_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                pose_decoder=pose_decoder,
                device=device,
            )
            _train_model_on_session_split(
                session_model,
                session,
                session.query_indices,
                mode=EEG_ADAPTATION_MODE_PROFILE_CORE,
                epochs=query_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                pose_decoder=pose_decoder,
                device=device,
            )
            _copy_profile_core_state(session_model, proposed_model)
            reset_band_adapter_identity(proposed_model)

    proposed_model_path = history_root / "proposed_profile_model.pt"
    reset_band_adapter_identity(proposed_model)
    save_model(proposed_model_path, proposed_model)
    session_metrics = evaluate_profile_post_adaptation(
        proposed_model_path,
        sessions,
        inner_adaptation_mode=inner_adaptation_mode,
        inner_epochs=inner_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pose_checkpoint=pose_checkpoint,
        device=device,
    )
    new_metrics = summarize_post_adaptation_metrics(session_metrics)
    committed = should_commit_profile_update(old_metrics, new_metrics)
    if committed:
        shutil.copy2(proposed_model_path, paths.profile_model)
        _write_json(
            paths.root / "profile_metrics.json",
            {
                "metrics": new_metrics,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

    report = ProfileBuildReport(
        user_id=user_id,
        profile_model=str(paths.profile_model),
        proposed_model=str(proposed_model_path),
        committed=committed,
        start_checkpoint=str(start_checkpoint),
        history_dir=str(history_root),
        session_count=len(sessions),
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        old_session_metrics=tuple(old_session_metrics) if old_session_metrics else None,
        session_metrics=tuple(session_metrics),
    )
    _write_profile_build_report(history_root, report)
    return report


def load_profile_session(
    path: str | Path | list[str | Path],
    *,
    query_fraction: float = EEG_PROFILE_BUILD_QUERY_FRACTION,
) -> ProfileSession:
    source_paths, name = _profile_session_source_paths(path)
    parts = [_load_profile_arrays(source_path) for source_path in source_paths]
    lengths = [len(part["eeg"]) for part in parts]
    eeg = np.concatenate([part["eeg"] for part in parts], axis=0)
    support_indices, query_indices = _support_query_indices_for_files(
        lengths,
        query_fraction=query_fraction,
    )
    return ProfileSession(
        name=name,
        source_paths=tuple(source_paths),
        eeg=eeg,
        pose_latent=np.concatenate([part["pose_latent"] for part in parts], axis=0),
        pose_confidence=np.concatenate([part["pose_confidence"] for part in parts], axis=0),
        interpolation_confidence=np.concatenate(
            [part["interpolation_confidence"] for part in parts],
            axis=0,
        ),
        pose_reconstruction_error=np.concatenate(
            [part["pose_reconstruction_error"] for part in parts],
            axis=0,
        ),
        support_indices=support_indices,
        query_indices=query_indices,
    )


def evaluate_profile_post_adaptation(
    checkpoint: str | Path,
    sessions: list[ProfileSession],
    *,
    inner_adaptation_mode: str = EEG_ADAPTATION_MODE_ADAPTER_HEAD,
    inner_epochs: int = EEG_PROFILE_BUILD_INNER_EPOCHS,
    batch_size: int = EEG_MODEL_BATCH_SIZE,
    learning_rate: float = EEG_MODEL_LR,
    pose_checkpoint: str | Path | None = POSE_ENCODING_MODEL,
    device: str | torch.device = DEFAULT_DEVICE,
) -> list[PostAdaptationMetrics]:
    metrics = []
    pose_decoder = _load_pose_decoder(
        pose_checkpoint,
        pose_latent_dim=sessions[0].pose_latent.shape[1],
        device=device,
    )
    for session in sessions:
        model = _load_raw_model(checkpoint, device)
        reset_band_adapter_identity(model)
        _train_model_on_session_split(
            model,
            session,
            session.support_indices,
            mode=inner_adaptation_mode,
            epochs=inner_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            pose_decoder=pose_decoder,
            device=device,
        )
        metrics.append(
            _evaluate_query_metrics(
                model,
                session,
                session.query_indices,
                pose_decoder=pose_decoder,
                device=device,
            )
        )
    return metrics


def summarize_post_adaptation_metrics(
    metrics: list[PostAdaptationMetrics],
) -> dict[str, float]:
    return {
        "pose_mae": float(np.mean([metric.pose_mae for metric in metrics])),
        "decoded_pose_error": float(
            np.mean([metric.decoded_pose_error for metric in metrics])
        ),
        "stationary_false_positive_score": float(
            np.mean([metric.stationary_false_positive_score for metric in metrics])
        ),
        "movement_response_score": float(
            np.mean([metric.movement_response_score for metric in metrics])
        ),
        "readiness_score": float(
            np.mean([metric.readiness_score for metric in metrics])
        ),
    }


def _write_profile_session_archives(
    data: str | Path | list[str | Path],
    *,
    history_root: Path,
    batch_id: str,
) -> list[Path]:
    session_groups = _resolve_profile_session_groups(data)
    session_roots = []
    for index, (name, source_paths) in enumerate(session_groups):
        session_root = history_root / f"{batch_id}_{index:02d}_{_safe_name(name)}"
        session_root.mkdir(parents=True, exist_ok=True)
        session_roots.append(session_root)
        copied_paths = []
        for source_index, source_path in enumerate(source_paths):
            copied_path = session_root / f"source_{source_index:02d}.npz"
            shutil.copy2(source_path, copied_path)
            copied_paths.append(copied_path)
        _write_merged_profile_archive(
            copied_paths,
            session_root / "paired_profile_session.npz",
        )
    return session_roots


def _write_merged_profile_archive(source_paths: list[Path], out_path: Path) -> None:
    if len(source_paths) == 1:
        shutil.copy2(source_paths[0], out_path)
        return

    arrays: dict[str, list[np.ndarray]] = {}
    metadata: dict[str, np.ndarray] = {}
    for path in source_paths:
        with np.load(path) as archive:
            n_samples = len(archive["eeg"])
            for key in archive.files:
                value = np.asarray(archive[key])
                if value.shape[:1] == (n_samples,):
                    arrays.setdefault(key, []).append(value)
                elif key not in metadata:
                    metadata[key] = value
    merged = {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}
    merged.update(metadata)
    np.savez_compressed(out_path, **merged)


def _resolve_profile_session_groups(
    data: str | Path | list[str | Path],
) -> list[tuple[str, list[Path]]]:
    if not isinstance(data, list):
        data = [data]
    groups: list[tuple[str, list[Path]]] = []
    for item in data:
        raw_path = Path(item)
        path = raw_path if raw_path.is_absolute() else ROOT_DIR / raw_path
        if path.is_dir():
            groups.append((path.name, sorted(path.glob("*.npz"))))
            continue
        has_glob = any(marker in str(raw_path) for marker in ("*", "?", "["))
        matches = sorted(glob.glob(str(path)))
        if has_glob and matches:
            by_parent: dict[Path, list[Path]] = {}
            for match in matches:
                match_path = Path(match)
                by_parent.setdefault(match_path.parent, []).append(match_path)
            groups.extend(
                (parent.name, paths)
                for parent, paths in sorted(by_parent.items(), key=lambda item: str(item[0]))
            )
        else:
            source_path = Path(matches[0]) if matches else path
            groups.append((source_path.stem, [source_path]))
    return groups


def _profile_session_roots(history_root: Path) -> list[Path]:
    if not history_root.exists():
        return []
    return sorted(
        path
        for path in history_root.iterdir()
        if path.is_dir() and (path / "paired_profile_session.npz").exists()
    )


def _profile_session_source_paths(
    path: str | Path | list[str | Path],
) -> tuple[list[Path], str]:
    if isinstance(path, list):
        paths = [Path(item) for item in path]
        return paths, paths[0].parent.name if paths else "session"
    path = Path(path)
    if path.is_dir():
        source_paths = sorted(path.glob("source_*.npz"))
        if source_paths:
            return source_paths, path.name
        return [path / "paired_profile_session.npz"], path.name
    if path.name == "paired_profile_session.npz":
        source_paths = sorted(path.parent.glob("source_*.npz"))
        if source_paths:
            return source_paths, path.parent.name
        return [path], path.parent.name
    return [path], path.stem


def _load_profile_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {
            "eeg": np.asarray(archive["eeg"], dtype=np.float32),
            "pose_latent": np.asarray(archive["pose_latent"], dtype=np.float32),
            "pose_confidence": np.asarray(archive["pose_confidence"], dtype=np.float32),
            "interpolation_confidence": np.asarray(
                archive["interpolation_confidence"],
                dtype=np.float32,
            ),
            "pose_reconstruction_error": np.asarray(
                archive["pose_reconstruction_error"],
                dtype=np.float32,
            ),
        }


def _support_query_indices(
    n_samples: int,
    *,
    query_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_samples)
    if n_samples < 3:
        return indices, indices
    query_count = min(max(1, round(n_samples * query_fraction)), n_samples - 1)
    return indices[:-query_count], indices[-query_count:]


def _support_query_indices_for_files(
    lengths: list[int],
    *,
    query_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(lengths) < 2:
        return _support_query_indices(lengths[0], query_fraction=query_fraction)

    starts = np.cumsum([0, *lengths[:-1]])
    file_indices = list(range(len(lengths)))
    query_file_count = min(
        max(1, round(len(lengths) * query_fraction)),
        len(lengths) - 1,
    )
    support_files = file_indices[:-query_file_count]
    query_files = file_indices[-query_file_count:]
    support = np.concatenate(
        [np.arange(starts[i], starts[i] + lengths[i]) for i in support_files],
    )
    query = np.concatenate(
        [np.arange(starts[i], starts[i] + lengths[i]) for i in query_files],
    )
    return support.astype(np.int64), query.astype(np.int64)


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)
    return safe or "session"


def _train_model_on_session_split(
    model: EegPoseVAE,
    session: ProfileSession,
    indices: np.ndarray,
    *,
    mode: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pose_decoder: nn.Module | None,
    device: str | torch.device,
) -> None:
    if epochs <= 0 or len(indices) == 0:
        return
    eeg = _session_eeg_contexts(model, session)[indices]
    target_raw = session.pose_latent[indices]
    target_model = standardize_pose_latents(target_raw, model.config)
    dataset = TensorDataset(
        torch.from_numpy(eeg),
        torch.from_numpy(target_model),
        torch.from_numpy(target_raw),
    )
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    set_trainable_scope(model, mode)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    position_weights = _landmark_weight_tensor(EEG_POSITION_LANDMARK_WEIGHTS, device=device)
    velocity_weights = _landmark_weight_tensor(EEG_VELOCITY_LANDMARK_WEIGHTS, device=device)
    model.train()
    for _ in range(epochs):
        for batch_eeg, batch_target_model, batch_target_raw in loader:
            batch_eeg = batch_eeg.to(device)
            batch_target_model = batch_target_model.to(device)
            batch_target_raw = batch_target_raw.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted, reconstruction, mean, log_variance = model(batch_eeg)
            prediction_loss = loss_fn(predicted, batch_target_model)
            reconstruction_target = model.adapt_eeg_features(batch_eeg).detach()
            reconstruction_loss = nn.functional.mse_loss(reconstruction, reconstruction_target)
            predicted_raw = unstandardize_pose_latents(predicted, model.config)
            decoded_loss = (
                decoded_pose_training_loss(
                    predicted_raw,
                    batch_target_raw,
                    pose_decoder,
                    position_weight=EEG_DECODED_POSITION_LOSS_WEIGHT,
                    velocity_weight=EEG_DECODED_VELOCITY_LOSS_WEIGHT,
                    stillness_weight=EEG_STILLNESS_LOSS_WEIGHT,
                    target_velocity_threshold=EEG_STILLNESS_TARGET_VELOCITY_THRESHOLD,
                    allowed_predicted_velocity=EEG_STILLNESS_ALLOWED_PREDICTED_VELOCITY,
                    position_landmark_weights=position_weights,
                    velocity_landmark_weights=velocity_weights,
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
                + EEG_MODEL_RECONSTRUCTION_WEIGHT * reconstruction_loss
                + EEG_MODEL_BETA * kl_loss
            )
            loss.backward()
            optimizer.step()
    model.eval()


def _evaluate_query_metrics(
    model: EegPoseVAE,
    session: ProfileSession,
    indices: np.ndarray,
    *,
    pose_decoder: nn.Module | None,
    device: str | torch.device,
) -> PostAdaptationMetrics:
    eeg = _session_eeg_contexts(model, session)[indices]
    target = session.pose_latent[indices]
    with torch.no_grad():
        predicted_model = model.predict_pose_latent(torch.from_numpy(eeg).to(device))
        predicted = unstandardize_pose_latents(
            predicted_model,
            model.config,
        ).cpu().numpy()
    error = predicted - target
    pose_mae = float(np.mean(np.abs(error)))
    decoded_error = pose_mae
    stationary_score = 0.0
    movement_score = 0.0
    if pose_decoder is not None:
        with torch.no_grad():
            predicted_features = pose_decoder.decode(
                torch.from_numpy(predicted).to(device)
            ).cpu().numpy()
            target_features = pose_decoder.decode(
                torch.from_numpy(target).to(device)
            ).cpu().numpy()
        decoded_error = float(np.mean(_position_error(predicted_features, target_features)))
        predicted_velocity = _mean_velocity(predicted_features)
        target_velocity = _mean_velocity(target_features)
        still_mask = target_velocity < EEG_STILLNESS_TARGET_VELOCITY_THRESHOLD
        stationary_score = _stationary_false_positive(predicted_velocity, still_mask)
        movement_score = max(0.0, _movement_response(predicted_velocity, target_velocity))

    readiness = score_readiness(
        ReadinessMetrics(
            trusted_sample_count=len(indices),
            mean_pose_confidence=float(np.mean(session.pose_confidence[indices])),
            mean_interpolation_confidence=float(
                np.mean(session.interpolation_confidence[indices])
            ),
            mean_pose_reconstruction_error=float(
                np.mean(session.pose_reconstruction_error[indices])
            ),
            decoded_pose_error=decoded_error,
            stationary_false_positive_score=stationary_score,
            movement_response_score=movement_score,
        )
    )
    return PostAdaptationMetrics(
        session=session.name,
        query_samples=len(indices),
        pose_mae=pose_mae,
        decoded_pose_error=decoded_error,
        stationary_false_positive_score=stationary_score,
        movement_response_score=movement_score,
        readiness_score=readiness.score,
        ready=readiness.ready,
    )


def _session_eeg_contexts(model: EegPoseVAE, session: ProfileSession) -> np.ndarray:
    return build_context_windows(
        transform_eeg_for_model(session.eeg, model.config),
        context_packet_count=model.config.context_packet_count,
    )


def _copy_profile_core_state(source: EegPoseVAE, target: EegPoseVAE) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name.startswith("band_adapter.") or name.startswith("decoder."):
            continue
        target_state[name].copy_(value)


def _landmark_weight_tensor(
    values: tuple[float, ...],
    *,
    device: str | torch.device,
) -> torch.Tensor:
    weights = torch.tensor(values, dtype=torch.float32, device=device)
    assert weights.shape == (8,)
    return weights.reshape(1, 8, 1)


def _position_error(predicted_features: np.ndarray, target_features: np.ndarray) -> np.ndarray:
    predicted = predicted_features[:, :24].reshape(-1, 8, 3)
    target = target_features[:, :24].reshape(-1, 8, 3)
    return np.linalg.norm(predicted - target, axis=2).mean(axis=1)


def _mean_velocity(features: np.ndarray) -> np.ndarray:
    if features.shape[1] < 48:
        return np.zeros(len(features), dtype=np.float32)
    velocity = features[:, 24:48].reshape(-1, 8, 3)
    return np.linalg.norm(velocity, axis=2).mean(axis=1)


def _stationary_false_positive(
    predicted_velocity: np.ndarray,
    still_mask: np.ndarray,
) -> float:
    if not np.any(still_mask):
        return 0.0
    excess = np.maximum(
        0.0,
        predicted_velocity[still_mask] - EEG_STILLNESS_ALLOWED_PREDICTED_VELOCITY,
    )
    return float(np.mean(excess))


def _movement_response(
    predicted_velocity: np.ndarray,
    target_velocity: np.ndarray,
) -> float:
    if np.std(predicted_velocity) <= 1e-8 or np.std(target_velocity) <= 1e-8:
        return 0.0
    return float(np.corrcoef(predicted_velocity, target_velocity)[0, 1])


def _write_profile_build_report(history_root: Path, report: ProfileBuildReport) -> None:
    _write_json(
        history_root / "profile_build_report.json",
        {
            "user_id": report.user_id,
            "profile_model": report.profile_model,
            "proposed_model": report.proposed_model,
            "committed": report.committed,
            "start_checkpoint": report.start_checkpoint,
            "history_dir": report.history_dir,
            "session_count": report.session_count,
            "old_metrics": report.old_metrics,
            "new_metrics": report.new_metrics,
            "old_session_metrics": (
                [asdict(metric) for metric in report.old_session_metrics]
                if report.old_session_metrics is not None
                else None
            ),
            "session_metrics": [asdict(metric) for metric in report.session_metrics],
        },
    )
    old_by_session = {
        metric.session: metric
        for metric in (report.old_session_metrics or ())
    }
    with (history_root / "profile_build_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "session",
            "query_samples",
            "old_decoded_pose_error",
            "new_decoded_pose_error",
            "decoded_pose_error_delta",
            "old_readiness_score",
            "new_readiness_score",
            "readiness_score_delta",
            "old_stationary_false_positive_score",
            "new_stationary_false_positive_score",
            "old_movement_response_score",
            "new_movement_response_score",
            "new_pose_mae",
            "new_ready",
        ]
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for metric in report.session_metrics:
            old = old_by_session.get(metric.session)
            old_decoded = old.decoded_pose_error if old is not None else ""
            old_readiness = old.readiness_score if old is not None else ""
            writer.writerow(
                {
                    "session": metric.session,
                    "query_samples": metric.query_samples,
                    "old_decoded_pose_error": old_decoded,
                    "new_decoded_pose_error": metric.decoded_pose_error,
                    "decoded_pose_error_delta": (
                        metric.decoded_pose_error - old.decoded_pose_error
                        if old is not None
                        else ""
                    ),
                    "old_readiness_score": old_readiness,
                    "new_readiness_score": metric.readiness_score,
                    "readiness_score_delta": (
                        metric.readiness_score - old.readiness_score
                        if old is not None
                        else ""
                    ),
                    "old_stationary_false_positive_score": (
                        old.stationary_false_positive_score if old is not None else ""
                    ),
                    "new_stationary_false_positive_score": (
                        metric.stationary_false_positive_score
                    ),
                    "old_movement_response_score": (
                        old.movement_response_score if old is not None else ""
                    ),
                    "new_movement_response_score": metric.movement_response_score,
                    "new_pose_mae": metric.pose_mae,
                    "new_ready": metric.ready,
                }
            )


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
