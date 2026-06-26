from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from config import (
    DEFAULT_DEVICE,
    EEG_ADAPTATION_MODE_PROFILE_HEAD,
    EEG_ADAPTATION_MODE_SESSION,
    EEG_CALIBRATION_MAX_POSE_RECONSTRUCTION_ERROR,
    EEG_CALIBRATION_MAX_DECODED_POSE_ERROR,
    EEG_CALIBRATION_MAX_STATIONARY_FALSE_POSITIVE_SCORE,
    EEG_CALIBRATION_MIN_INTERPOLATION_CONFIDENCE,
    EEG_CALIBRATION_MIN_MOVEMENT_RESPONSE_SCORE,
    EEG_CALIBRATION_MIN_POSE_CONFIDENCE,
    EEG_CALIBRATION_MIN_TRUSTED_SAMPLES,
    EEG_CALIBRATION_READY_SCORE,
    EEG_MODEL_BATCH_SIZE,
    EEG_MODEL_EPOCHS,
    EEG_MODEL_LR,
    EEG_PROFILES_ROOT,
    EEG_VALIDATE_BY_RUN,
    POSE_ENCODING_MODEL,
)

from .model import EegTrainingReport, format_training_report, train_model


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


def propose_profile_update(
    *,
    user_id: str,
    start_checkpoint: str | Path,
    profile_data: str | Path | list[str | Path],
    proposal_id: str | None = None,
    profiles_root: str | Path = EEG_PROFILES_ROOT,
    epochs: int = EEG_MODEL_EPOCHS,
    batch_size: int = EEG_MODEL_BATCH_SIZE,
    learning_rate: float = EEG_MODEL_LR,
    adaptation_mode: str = EEG_ADAPTATION_MODE_PROFILE_HEAD,
    pose_checkpoint: str | Path | None = POSE_ENCODING_MODEL,
    validate_by_run: bool = EEG_VALIDATE_BY_RUN,
    device: str = DEFAULT_DEVICE,
) -> EegTrainingReport:
    paths = profile_paths(user_id, profiles_root=profiles_root)
    proposal_id = proposal_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    proposal_root = paths.history_root / proposal_id
    proposed_model_path = proposal_root / "proposed_profile_model.pt"
    predictor = train_model(
        profile_data,
        proposed_model_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_path=start_checkpoint,
        adaptation_mode=adaptation_mode,
        pose_checkpoint=pose_checkpoint,
        validate_by_run=validate_by_run,
        device=device,
    )
    assert predictor.training_report is not None
    _write_json(
        proposal_root / "proposal_training_report.json",
        {"training_report": format_training_report(predictor.training_report)},
    )
    return predictor.training_report


def should_commit_profile_update(
    old_metrics: dict[str, float] | None,
    new_metrics: dict[str, float],
) -> bool:
    if old_metrics is None:
        return True
    for metric_name in (
        "validation_pose_mae",
        "stationary_false_positive_score",
    ):
        if metric_name in old_metrics and metric_name in new_metrics:
            if new_metrics[metric_name] > old_metrics[metric_name]:
                return False
    if "movement_response_score" in old_metrics and "movement_response_score" in new_metrics:
        if new_metrics["movement_response_score"] < old_metrics["movement_response_score"]:
            return False
    return True


def commit_profile_update(
    *,
    user_id: str,
    proposed_model: str | Path,
    old_metrics: dict[str, float] | None,
    new_metrics: dict[str, float],
    profiles_root: str | Path = EEG_PROFILES_ROOT,
) -> bool:
    if not should_commit_profile_update(old_metrics, new_metrics):
        return False

    paths = profile_paths(user_id, profiles_root=profiles_root)
    paths.root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(proposed_model, paths.profile_model)
    _write_json(
        paths.root / "profile_metrics.json",
        {
            "metrics": new_metrics,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return True


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
