from __future__ import annotations

import copy
import csv
import glob
import json
import shutil
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
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
    build_eligible_grouped_context_windows,
    decoded_pose_training_loss,
    format_training_report,
    preprocessing_signature_from_archive,
    preprocessing_signature_from_config,
    require_matching_preprocessing,
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
from .permanent_holdout import (
    HoldoutSlot,
    HoldoutUpdate,
    PermanentHoldoutManifest,
    canonical_preprocessing_signature,
    candidates_from_test_blocks,
    checksum_archive_rows,
    checksum_preprocessing_signature,
    load_manifest,
    replace_next_slot_after_decision,
    require_complete_manifest,
    seed_manifest_from_first_session,
)


@dataclass(frozen=True, slots=True)
class ProfilePaths:
    root: Path
    profile_model: Path
    history_root: Path
    sessions_root: Path
    holdout_root: Path
    holdout_manifest: Path


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
    validation_indices: np.ndarray
    test_indices: np.ndarray
    round_roles: np.ndarray
    target_eligible: np.ndarray
    history_eligible: np.ndarray
    block_ids: np.ndarray | None = None
    block_names: np.ndarray | None = None
    block_is_rest: np.ndarray | None = None
    block_accepted: np.ndarray | None = None
    block_summary: tuple[dict[str, object], ...] = ()
    posture: str = ""
    preprocessing_signature: dict[str, object] | None = None


class _ProfileBuildProgress:
    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.completed_steps = 0

    def start(self, description: str) -> None:
        width = 24
        filled = round(width * self.completed_steps / self.total_steps)
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"Profile build [{bar}] "
            f"{self.completed_steps}/{self.total_steps}: {description}",
            flush=True,
        )

    def finish_step(self) -> None:
        self.completed_steps += 1


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
    session_block_summaries: dict[str, dict[str, object]]
    validation_passed: bool = True
    holdout_passed: bool = True
    old_holdout_metrics: dict[str, float] | None = None
    new_holdout_metrics: dict[str, float] | None = None
    test_metrics: tuple[PostAdaptationMetrics, ...] = ()
    holdout_update: dict[str, object] | None = None


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
        holdout_root=root / "permanent_holdout",
        holdout_manifest=root / "permanent_holdout" / "manifest.json",
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


def profile_update_gate(
    old_validation_metrics: dict[str, float],
    new_validation_metrics: dict[str, float],
    old_holdout_metrics: dict[str, float] | None,
    new_holdout_metrics: dict[str, float] | None,
    *,
    holdout_expected: bool = False,
) -> tuple[bool, bool, bool]:
    validation_passed = should_commit_profile_update(
        old_validation_metrics,
        new_validation_metrics,
    )
    if old_holdout_metrics is None or new_holdout_metrics is None:
        holdout_passed = not holdout_expected
    else:
        holdout_passed = should_commit_profile_update(
            old_holdout_metrics,
            new_holdout_metrics,
        )
    return (
        validation_passed and holdout_passed,
        validation_passed,
        holdout_passed,
    )


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


def seed_profile_holdout(
    *,
    user_id: str,
    session_archive: str | Path,
    profiles_root: str | Path = EEG_PROFILES_ROOT,
) -> PermanentHoldoutManifest:
    paths = profile_paths(user_id, profiles_root=profiles_root)
    candidates = _holdout_candidates_for_archives([Path(session_archive)])
    if not candidates:
        raise ValueError(
            "Cannot seed the permanent holdout because the session has no "
            "eligible fourth-round blocks."
        )
    return seed_manifest_from_first_session(paths.holdout_manifest, candidates)


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

    # Validate the incoming sources before copying them into permanent history.
    session_groups = _resolve_profile_session_groups(new_session_data)
    if not session_groups or any(not source_paths for _, source_paths in session_groups):
        raise ValueError("Profile build requires at least one captured session.")
    staged_current_sessions = [
        load_profile_session(
            source_paths if len(source_paths) > 1 else source_paths[0],
            query_fraction=query_fraction,
        )
        for _, source_paths in session_groups
    ]
    existing_session_roots = _profile_session_roots(paths.history_root)
    existing_sessions = [
        load_profile_session(root, query_fraction=query_fraction)
        for root in existing_session_roots
    ]
    start_checkpoint = profile_start_checkpoint(
        user_id,
        base_checkpoint,
        profiles_root=profiles_root,
    )
    start_model = _load_raw_model(start_checkpoint, device)
    start_signature = preprocessing_signature_from_config(start_model.config)
    for session in [*existing_sessions, *staged_current_sessions]:
        assert session.preprocessing_signature is not None
        require_matching_preprocessing(
            start_signature,
            session.preprocessing_signature,
            source=f"profile session {session.name}",
        )
    del start_model

    has_explicit_test_round = any(
        len(session.test_indices) > 0
        for session in staged_current_sessions
    )
    manifest_snapshot = None
    if has_explicit_test_round:
        multi_source_groups = [
            name
            for name, source_paths in session_groups
            if len(source_paths) > 1
        ]
        if multi_source_groups:
            raise ValueError(
                "Permanent holdout provenance requires one archive per profile "
                f"session; multi-source groups: {multi_source_groups}."
            )
        if not paths.holdout_manifest.exists():
            raise RuntimeError(
                "A four-round profile build requires an existing complete "
                "permanent holdout; run bootstrap-base first."
            )
        manifest_snapshot = load_manifest(paths.holdout_manifest)
        require_complete_manifest(
            manifest_snapshot,
            source=paths.holdout_manifest,
        )
    elif paths.holdout_manifest.exists():
        # Legacy no-role sessions remain supported for direct research tests, but
        # an existing corrected holdout must still be complete and active.
        manifest_snapshot = load_manifest(paths.holdout_manifest)
        require_complete_manifest(
            manifest_snapshot,
            source=paths.holdout_manifest,
        )

    if manifest_snapshot is not None:
        canonical_start_signature = canonical_preprocessing_signature(
            start_signature
        )
        start_signature_checksum = checksum_preprocessing_signature(
            canonical_start_signature
        )
        if (
            manifest_snapshot.preprocessing_signature
            != canonical_start_signature
            or manifest_snapshot.preprocessing_checksum
            != start_signature_checksum
        ):
            raise ValueError(
                "Permanent holdout preprocessing does not match the starting "
                "profile checkpoint. Rebuild the holdout from data captured "
                "with the checkpoint preprocessing."
            )

    history_root.mkdir(parents=True, exist_ok=True)
    paths.root.mkdir(parents=True, exist_ok=True)
    new_session_roots = _write_profile_session_archives(
        new_session_data,
        history_root=paths.history_root,
        batch_id=timestamp,
    )
    current_sessions = [
        load_profile_session(root, query_fraction=query_fraction)
        for root in new_session_roots
    ]
    sessions = [*existing_sessions, *current_sessions]
    current_archive_paths = [
        root / "paired_profile_session.npz"
        for root in new_session_roots
    ]
    progress = _ProfileBuildProgress(
        total_steps=max(1, epochs * len(sessions) + 5)
    )

    progress.start("evaluating the starting profile on validation rounds")
    old_session_metrics = evaluate_profile_post_adaptation(
        start_checkpoint,
        current_sessions,
        inner_adaptation_mode=inner_adaptation_mode,
        inner_epochs=inner_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pose_checkpoint=pose_checkpoint,
        device=device,
        evaluation_role="validation",
        adaptation_seed=0,
    )
    old_metrics = summarize_post_adaptation_metrics(old_session_metrics)
    progress.finish_step()

    progress.start("loading the starting model and pose decoder")
    proposed_model = _load_raw_model(start_checkpoint, device)
    reset_band_adapter_identity(proposed_model)
    pose_decoder = _load_pose_decoder(
        pose_checkpoint,
        pose_latent_dim=sessions[0].pose_latent.shape[1],
        device=device,
    )
    best_profile_state = copy.deepcopy(proposed_model.state_dict())
    best_validation_error: float | None = None
    for epoch_index in range(epochs):
        for session_index, session in enumerate(sessions):
            progress.start(
                f"training epoch {epoch_index + 1}/{epochs}, "
                f"session {session_index + 1}/{len(sessions)} ({session.name})"
            )
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
            progress.finish_step()
        epoch_validation_metrics = _evaluate_profile_model_post_adaptation(
            proposed_model,
            current_sessions,
            inner_adaptation_mode=inner_adaptation_mode,
            inner_epochs=inner_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            pose_decoder=pose_decoder,
            device=device,
            evaluation_role="validation",
            adaptation_seed=0,
        )
        epoch_validation = summarize_post_adaptation_metrics(
            epoch_validation_metrics
        )
        validation_error = epoch_validation["decoded_pose_error"]
        if not np.isfinite(validation_error):
            raise ValueError(
                f"Outer epoch {epoch_index + 1} produced non-finite "
                "validation decoded-pose error."
            )
        if (
            best_validation_error is None
            or validation_error < best_validation_error
        ):
            best_validation_error = validation_error
            best_profile_state = copy.deepcopy(proposed_model.state_dict())

    proposed_model.load_state_dict(best_profile_state)
    reset_band_adapter_identity(proposed_model)

    proposed_model_path = history_root / "proposed_profile_model.pt"
    reset_band_adapter_identity(proposed_model)
    progress.start("saving and validating the proposed profile")
    save_model(proposed_model_path, proposed_model)
    session_metrics = evaluate_profile_post_adaptation(
        proposed_model_path,
        current_sessions,
        inner_adaptation_mode=inner_adaptation_mode,
        inner_epochs=inner_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pose_checkpoint=pose_checkpoint,
        device=device,
        evaluation_role="validation",
        adaptation_seed=0,
    )
    new_metrics = summarize_post_adaptation_metrics(session_metrics)
    progress.finish_step()

    old_holdout_metrics = None
    new_holdout_metrics = None
    holdout_passed = True
    if manifest_snapshot is not None:
        progress.start("evaluating the permanent holdout snapshot")
        old_holdout_session_metrics = evaluate_permanent_holdout(
            start_checkpoint,
            manifest_snapshot,
            inner_adaptation_mode=inner_adaptation_mode,
            inner_epochs=inner_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            pose_checkpoint=pose_checkpoint,
            device=device,
            adaptation_seed=0,
        )
        new_holdout_session_metrics = evaluate_permanent_holdout(
            proposed_model_path,
            manifest_snapshot,
            inner_adaptation_mode=inner_adaptation_mode,
            inner_epochs=inner_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            pose_checkpoint=pose_checkpoint,
            device=device,
            adaptation_seed=0,
        )
        if old_holdout_session_metrics and new_holdout_session_metrics:
            old_holdout_metrics = summarize_permanent_holdout_metrics(
                old_holdout_session_metrics,
                manifest_snapshot,
            )
            new_holdout_metrics = summarize_permanent_holdout_metrics(
                new_holdout_session_metrics,
                manifest_snapshot,
            )
        progress.finish_step()

    progress.start("committing the accepted profile and reporting the test round")
    committed, validation_passed, holdout_passed = profile_update_gate(
        old_metrics,
        new_metrics,
        old_holdout_metrics,
        new_holdout_metrics,
        holdout_expected=manifest_snapshot is not None,
    )
    test_metrics = evaluate_profile_post_adaptation(
        proposed_model_path,
        current_sessions,
        inner_adaptation_mode=inner_adaptation_mode,
        inner_epochs=inner_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pose_checkpoint=pose_checkpoint,
        device=device,
        evaluation_role="test",
        adaptation_seed=0,
    )

    candidates = (
        _holdout_candidates_for_archives(current_archive_paths)
        if has_explicit_test_round
        else ()
    )
    manifest_before = (
        paths.holdout_manifest.read_bytes()
        if manifest_snapshot is not None
        else None
    )
    metrics_path = paths.root / "profile_metrics.json"
    previous_metrics = metrics_path.read_bytes() if metrics_path.exists() else None
    pending_profile = paths.root / f".profile_model.{timestamp}.pending"
    pending_metrics = paths.root / f".profile_metrics.{timestamp}.pending"
    metrics_installed = False
    report_json = history_root / "profile_build_report.json"
    report_csv = history_root / "profile_build_summary.csv"
    holdout_update: HoldoutUpdate | None = None
    try:
        if committed:
            shutil.copy2(proposed_model_path, pending_profile)
            _write_json(
                pending_metrics,
                {
                    "validation_metrics": new_metrics,
                    "holdout_metrics": new_holdout_metrics,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )

        if manifest_snapshot is not None and candidates:
            holdout_update = replace_next_slot_after_decision(
                paths.holdout_manifest,
                candidates,
                decision_id=timestamp,
                committed=committed,
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
            old_session_metrics=tuple(old_session_metrics),
            session_metrics=tuple(session_metrics),
            session_block_summaries={
                session.name: _summarize_profile_session_blocks(session)
                for session in sessions
            },
            validation_passed=validation_passed,
            holdout_passed=holdout_passed,
            old_holdout_metrics=old_holdout_metrics,
            new_holdout_metrics=new_holdout_metrics,
            test_metrics=tuple(test_metrics),
            holdout_update=(
                asdict(holdout_update)
                if holdout_update is not None
                else None
            ),
        )
        _write_profile_build_report(history_root, report)

        # Make the primary profile the final state change. A failure before this
        # point leaves the installed profile untouched and rolls the holdout back.
        if committed:
            pending_metrics.replace(metrics_path)
            metrics_installed = True
            pending_profile.replace(paths.profile_model)
    except Exception:
        if manifest_before is not None:
            paths.holdout_manifest.write_bytes(manifest_before)
        if metrics_installed:
            if previous_metrics is None:
                metrics_path.unlink(missing_ok=True)
            else:
                metrics_path.write_bytes(previous_metrics)
        report_json.unlink(missing_ok=True)
        report_csv.unlink(missing_ok=True)
        raise
    finally:
        pending_profile.unlink(missing_ok=True)
        pending_metrics.unlink(missing_ok=True)

    progress.finish_step()
    progress.start("complete")
    return report


def load_profile_session(
    path: str | Path | list[str | Path],
    *,
    query_fraction: float = EEG_PROFILE_BUILD_QUERY_FRACTION,
) -> ProfileSession:
    source_paths, name = _profile_session_source_paths(path)
    parts = [_load_profile_arrays(source_path) for source_path in source_paths]
    preprocessing_signature = parts[0]["preprocessing_signature"]
    for source_path, part in zip(source_paths[1:], parts[1:]):
        require_matching_preprocessing(
            preprocessing_signature,
            part["preprocessing_signature"],
            source=str(source_path),
        )
    lengths = [len(part["eeg"]) for part in parts]
    eeg = np.concatenate([part["eeg"] for part in parts], axis=0)
    target_eligible = np.concatenate(
        [np.asarray(part["target_eligible"], dtype=bool) for part in parts]
    )
    history_eligible = np.concatenate(
        [np.asarray(part["history_eligible"], dtype=bool) for part in parts]
    )
    assert target_eligible.shape == history_eligible.shape == (len(eeg),)
    assert not np.any(target_eligible & ~history_eligible)
    block_ids = _merged_block_ids(parts)
    has_round_roles = all("round_role" in part for part in parts)
    if has_round_roles:
        round_roles = np.concatenate(
            [np.asarray(part["round_role"]).astype(str) for part in parts]
        )
        unknown_roles = sorted(
            set(round_roles.tolist()) - {"support", "query", "validation", "test"}
        )
        if unknown_roles:
            raise ValueError(f"Unknown profile round roles: {unknown_roles}.")
        support_indices = np.flatnonzero(
            (round_roles == "support") & target_eligible
        ).astype(np.int64)
        query_indices = np.flatnonzero(
            (round_roles == "query") & target_eligible
        ).astype(np.int64)
        validation_indices = np.flatnonzero(
            (round_roles == "validation") & target_eligible
        ).astype(np.int64)
        test_indices = np.flatnonzero(
            (round_roles == "test") & target_eligible
        ).astype(np.int64)
        if any(
            len(indices) == 0
            for indices in (
                support_indices,
                query_indices,
                validation_indices,
                test_indices,
            )
        ):
            raise ValueError(
                "A four-round profile session must contain target-eligible "
                "support, query, validation, and test frames."
            )
    elif block_ids is not None:
        support_indices, query_indices = _support_query_indices_for_blocks(
            block_ids,
            query_fraction=query_fraction,
        )
        validation_indices = query_indices
        test_indices = np.empty(0, dtype=np.int64)
        round_roles = np.full(len(eeg), "support", dtype="<U10")
        round_roles[query_indices] = "query"
    else:
        support_indices, query_indices = _support_query_indices_for_files(
            lengths,
            query_fraction=query_fraction,
        )
        validation_indices = query_indices
        test_indices = np.empty(0, dtype=np.int64)
        round_roles = np.full(len(eeg), "support", dtype="<U10")
        round_roles[query_indices] = "query"
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
        validation_indices=validation_indices,
        test_indices=test_indices,
        round_roles=round_roles,
        target_eligible=target_eligible,
        history_eligible=history_eligible,
        block_ids=block_ids,
        block_names=_concatenate_optional(parts, "block_name"),
        block_is_rest=_concatenate_optional(parts, "block_is_rest"),
        block_accepted=_concatenate_optional(parts, "block_accepted"),
        block_summary=tuple(
            summary
            for part in parts
            for summary in part.get("block_summary", ())
        ),
        posture=_first_nonempty(part.get("posture", "") for part in parts),
        preprocessing_signature=preprocessing_signature,
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
    progress: _ProfileBuildProgress | None = None,
    progress_phase: str = "evaluating profile",
    evaluation_role: str = "query",
    adaptation_seed: int | None = None,
) -> list[PostAdaptationMetrics]:
    if not sessions:
        return []
    model = _load_raw_model(checkpoint, device)
    pose_decoder = _load_pose_decoder(
        pose_checkpoint,
        pose_latent_dim=sessions[0].pose_latent.shape[1],
        device=device,
    )
    return _evaluate_profile_model_post_adaptation(
        model,
        sessions,
        inner_adaptation_mode=inner_adaptation_mode,
        inner_epochs=inner_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pose_decoder=pose_decoder,
        device=device,
        progress=progress,
        progress_phase=progress_phase,
        evaluation_role=evaluation_role,
        adaptation_seed=adaptation_seed,
    )


def _evaluate_profile_model_post_adaptation(
    model_template: EegPoseVAE,
    sessions: list[ProfileSession],
    *,
    inner_adaptation_mode: str,
    inner_epochs: int,
    batch_size: int,
    learning_rate: float,
    pose_decoder: nn.Module | None,
    device: str | torch.device,
    progress: _ProfileBuildProgress | None = None,
    progress_phase: str = "evaluating profile",
    evaluation_role: str = "query",
    adaptation_seed: int | None = None,
) -> list[PostAdaptationMetrics]:
    metrics = []
    for session_index, session in enumerate(sessions):
        if progress is not None:
            progress.start(
                f"{progress_phase}, session "
                f"{session_index + 1}/{len(sessions)} ({session.name})"
            )
        model = copy.deepcopy(model_template)
        reset_band_adapter_identity(model)
        with _forked_torch_seed(
            None
            if adaptation_seed is None
            else adaptation_seed + session_index
        ):
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
        evaluation_indices = {
            "query": session.query_indices,
            "validation": session.validation_indices,
            "test": session.test_indices,
        }.get(evaluation_role)
        if evaluation_indices is None:
            raise ValueError(
                f"Unsupported profile evaluation role {evaluation_role!r}."
            )
        if len(evaluation_indices) == 0:
            continue
        metrics.append(
            _evaluate_query_metrics(
                model,
                session,
                evaluation_indices,
                pose_decoder=pose_decoder,
                device=device,
            )
        )
        if progress is not None:
            progress.finish_step()
    return metrics


def evaluate_permanent_holdout(
    checkpoint: str | Path,
    manifest: PermanentHoldoutManifest,
    *,
    inner_adaptation_mode: str = EEG_ADAPTATION_MODE_ADAPTER_HEAD,
    inner_epochs: int = EEG_PROFILE_BUILD_INNER_EPOCHS,
    batch_size: int = EEG_MODEL_BATCH_SIZE,
    learning_rate: float = EEG_MODEL_LR,
    pose_checkpoint: str | Path | None = POSE_ENCODING_MODEL,
    device: str | torch.device = DEFAULT_DEVICE,
    adaptation_seed: int | None = None,
) -> list[PostAdaptationMetrics]:
    """Evaluate one equally weighted metric record per active holdout slot."""
    require_complete_manifest(manifest)
    active_slots = [
        slot
        for category in manifest.categories
        if (slot := manifest.slots[category]) is not None
    ]
    assert len(active_slots) == len(manifest.categories)

    metrics = []
    pose_decoder: nn.Module | None = None
    for slot_index, slot in enumerate(active_slots):
        session, indices = _load_exact_holdout_slot(slot, manifest)

        model = _load_raw_model(checkpoint, device)
        assert session.preprocessing_signature is not None
        require_matching_preprocessing(
            preprocessing_signature_from_config(model.config),
            session.preprocessing_signature,
            source=f"permanent holdout block {slot.category}",
        )
        reset_band_adapter_identity(model)
        if pose_decoder is None:
            pose_decoder = _load_pose_decoder(
                pose_checkpoint,
                pose_latent_dim=session.pose_latent.shape[1],
                device=device,
            )
        with _forked_torch_seed(
            None
            if adaptation_seed is None
            else adaptation_seed + slot_index
        ):
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
        metric = _evaluate_query_metrics(
            model,
            session,
            indices,
            pose_decoder=pose_decoder,
            device=device,
        )
        metrics.append(
            replace(
                metric,
                session=f"{slot.source_session_id}:{slot.category}",
            )
        )
    return metrics


def _load_exact_holdout_slot(
    slot: HoldoutSlot,
    manifest: PermanentHoldoutManifest,
) -> tuple[ProfileSession, np.ndarray]:
    source_path = Path(slot.source_path)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Permanent holdout source archive is missing: {source_path}."
        )
    if (
        source_path.name == "paired_profile_session.npz"
        and len(list(source_path.parent.glob("source_*.npz"))) > 1
    ):
        raise ValueError(
            "Permanent holdout source is an ambiguous multi-source merged "
            f"archive: {source_path}."
        )

    raw_indices = np.asarray(slot.sample_indices, dtype=np.int64)
    if len(raw_indices) != slot.sample_count:
        raise ValueError(
            f"Permanent holdout slot {slot.category} sample count changed."
        )
    with np.load(source_path) as archive:
        sample_total = len(archive["eeg"])
        if (
            len(np.unique(raw_indices)) != len(raw_indices)
            or np.any(raw_indices < 0)
            or np.any(raw_indices >= sample_total)
        ):
            raise ValueError(
                f"Permanent holdout slot {slot.category} has invalid sample indices."
            )

        archive_signature = preprocessing_signature_from_archive(archive)
        signature = canonical_preprocessing_signature(archive_signature)
        checksum = checksum_preprocessing_signature(signature)
        if (
            signature != manifest.preprocessing_signature
            or checksum != manifest.preprocessing_checksum
            or signature != slot.preprocessing_signature
            or checksum != slot.preprocessing_checksum
        ):
            raise ValueError(
                f"Permanent holdout slot {slot.category} preprocessing changed."
            )
        data_checksum = checksum_archive_rows(
            archive,
            raw_indices,
            preprocessing_checksum=checksum,
        )
        if data_checksum != slot.data_checksum:
            raise ValueError(
                f"Permanent holdout slot {slot.category} source checksum changed."
            )

        _verify_holdout_slot_metadata(archive, raw_indices, slot)
        target_eligible = _profile_archive_target_eligible(archive)
        if target_eligible.shape != (sample_total,):
            raise ValueError(
                f"Permanent holdout slot {slot.category} frame eligibility "
                "metadata has the wrong length."
            )
        evaluation_indices = raw_indices[target_eligible[raw_indices]]
        if len(evaluation_indices) == 0:
            raise ValueError(
                f"Permanent holdout slot {slot.category} has no trusted "
                "split-eligible samples."
            )

    # Passing a list deliberately loads this exact archive instead of following
    # sibling source_*.npz files.
    session = load_profile_session([source_path])
    if (
        len(session.eeg) != sample_total
        or np.any(evaluation_indices < 0)
        or np.any(evaluation_indices >= len(session.eeg))
        or not np.all(session.target_eligible[evaluation_indices])
    ):
        raise ValueError(
            f"Permanent holdout slot {slot.category} could not be loaded exactly."
        )
    return session, evaluation_indices


def _verify_holdout_slot_metadata(
    archive: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    slot: HoldoutSlot,
) -> None:
    required = (
        "profile_block_id",
        "profile_block_name",
        "profile_round_role",
    )
    missing = [key for key in required if key not in archive.files]
    if missing:
        raise ValueError(
            f"Permanent holdout source {slot.source_path} is missing {missing}."
        )
    block_ids = np.asarray(archive["profile_block_id"], dtype=np.int64)
    block_names = np.asarray(archive["profile_block_name"]).astype(str)
    roles = np.asarray(archive["profile_round_role"]).astype(str)
    if not np.all(block_ids[indices] == slot.source_block_id):
        raise ValueError(
            f"Permanent holdout slot {slot.category} block id changed."
        )
    if not np.all(block_names[indices] == slot.source_block_name):
        raise ValueError(
            f"Permanent holdout slot {slot.category} block name changed."
        )
    if not np.all(roles[indices] == "test"):
        raise ValueError(
            f"Permanent holdout slot {slot.category} is no longer a test block."
        )
    if "profile_block_repeat_index" in archive.files:
        repeats = np.asarray(
            archive["profile_block_repeat_index"],
            dtype=np.int64,
        )
        if not np.all(repeats[indices] == slot.source_repeat_index):
            raise ValueError(
                f"Permanent holdout slot {slot.category} repeat index changed."
            )
    if "profile_session_id" in archive.files:
        source_session_id = str(np.asarray(archive["profile_session_id"]).item())
        if source_session_id != slot.source_session_id:
            raise ValueError(
                f"Permanent holdout slot {slot.category} session id changed."
            )


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


def summarize_permanent_holdout_metrics(
    metrics: list[PostAdaptationMetrics],
    manifest: PermanentHoldoutManifest,
) -> dict[str, float]:
    require_complete_manifest(manifest)
    if len(metrics) != len(manifest.categories):
        raise ValueError(
            "Permanent holdout metric count does not match the manifest: "
            f"{len(metrics)} != {len(manifest.categories)}."
        )
    by_category = dict(zip(manifest.categories, metrics, strict=True))
    rest_metrics = [by_category["rest"]]
    movement_metrics = [
        by_category[category]
        for category in manifest.categories
        if category != "rest"
    ]
    return {
        "pose_mae": float(np.mean([metric.pose_mae for metric in metrics])),
        "decoded_pose_error": float(
            np.mean([metric.decoded_pose_error for metric in metrics])
        ),
        "stationary_false_positive_score": float(
            np.mean(
                [
                    metric.stationary_false_positive_score
                    for metric in rest_metrics
                ]
            )
        ),
        "movement_response_score": float(
            np.mean(
                [
                    metric.movement_response_score
                    for metric in movement_metrics
                ]
            )
        ),
        "readiness_score": float(
            np.mean([metric.readiness_score for metric in metrics])
        ),
    }


def _holdout_candidates_for_archives(
    archive_paths: list[Path],
):
    candidates = []
    for archive_path in archive_paths:
        with np.load(archive_path) as archive:
            if "profile_round_role" not in archive.files:
                continue
            roles = np.asarray(archive["profile_round_role"]).astype(str)
            target_eligible = _profile_archive_target_eligible(archive)
            if target_eligible.shape != roles.shape:
                raise ValueError(
                    f"Frame eligibility length mismatch in {archive_path}."
                )
            test_indices = np.flatnonzero(roles == "test").astype(np.int64)
            if "profile_block_id" not in archive.files:
                raise ValueError(
                    f"Four-round holdout archive has no block IDs: {archive_path}."
                )
            block_ids = np.asarray(archive["profile_block_id"], dtype=np.int64)
            if len(block_ids) != len(roles):
                raise ValueError(
                    f"Block-id length mismatch in {archive_path}."
                )
            eligible_block_ids = set(
                int(block_id)
                for block_id in block_ids[
                    (roles == "test") & target_eligible
                ]
            )
            preprocessing = preprocessing_signature_from_archive(archive)
        if len(test_indices) == 0:
            continue
        candidates.extend(
            candidates_from_test_blocks(
                archive_path,
                test_indices,
                preprocessing=preprocessing,
                eligible_block_ids=eligible_block_ids,
            )
        )
    return tuple(candidates)


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
        preprocessing_signature = preprocessing_signature_from_archive(archive)
        accepted_mask = (
            np.asarray(archive["profile_block_accepted"], dtype=bool)
            if "profile_block_accepted" in archive.files
            else None
        )
        block_summary = _load_block_summary_json(archive)
        posture = (
            str(np.asarray(archive["profile_posture"]).item())
            if "profile_posture" in archive.files
            else ""
        )
        arrays = {
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
        if "profile_block_id" in archive.files:
            arrays["block_id"] = np.asarray(archive["profile_block_id"], dtype=np.int64)
        has_round_roles = "profile_round_role" in archive.files
        if has_round_roles:
            arrays["round_role"] = np.asarray(archive["profile_round_role"]).astype(str)
            if "profile_block_name" in archive.files:
                arrays["block_name"] = np.asarray(archive["profile_block_name"]).astype(str)
            if "profile_block_is_rest" in archive.files:
                arrays["block_is_rest"] = np.asarray(
                    archive["profile_block_is_rest"],
                    dtype=bool,
                )
            if accepted_mask is not None:
                arrays["block_accepted"] = accepted_mask
            target_eligible = _profile_archive_target_eligible(archive)
            history_eligible = _profile_archive_history_eligible(archive)
            expected_shape = (len(arrays["eeg"]),)
            if target_eligible.shape != expected_shape:
                raise ValueError(
                    f"Target-eligibility metadata in {path} must have shape "
                    f"{expected_shape}, got {target_eligible.shape}."
                )
            if history_eligible.shape != expected_shape:
                raise ValueError(
                    f"History-eligibility metadata in {path} must have shape "
                    f"{expected_shape}, got {history_eligible.shape}."
                )
            arrays["target_eligible"] = target_eligible
            arrays["history_eligible"] = history_eligible
        elif accepted_mask is not None:
            arrays = {
                key: value[accepted_mask]
                for key, value in arrays.items()
            }
            arrays["target_eligible"] = np.ones(
                len(arrays["eeg"]),
                dtype=bool,
            )
            arrays["history_eligible"] = np.ones(
                len(arrays["eeg"]),
                dtype=bool,
            )
        else:
            arrays["target_eligible"] = np.ones(
                len(arrays["eeg"]),
                dtype=bool,
            )
            arrays["history_eligible"] = np.ones(
                len(arrays["eeg"]),
                dtype=bool,
            )
        arrays["block_summary"] = block_summary
        arrays["posture"] = posture
        arrays["preprocessing_signature"] = preprocessing_signature
        return arrays


def _profile_archive_target_eligible(
    archive: np.lib.npyio.NpzFile,
) -> np.ndarray:
    expected_shape = (len(archive["eeg"]),)
    trusted = (
        np.asarray(archive["profile_frame_trusted"], dtype=bool)
        if "profile_frame_trusted" in archive.files
        else trusted_calibration_mask(archive)
    )
    if trusted.shape != expected_shape:
        raise ValueError(
            "profile_frame_trusted must have shape "
            f"{expected_shape}, got {trusted.shape}."
        )
    return trusted & _profile_archive_history_eligible(archive)


def _profile_archive_history_eligible(
    archive: np.lib.npyio.NpzFile,
) -> np.ndarray:
    expected_shape = (len(archive["eeg"]),)
    if "profile_frame_split_eligible" in archive.files:
        eligible = np.asarray(
            archive["profile_frame_split_eligible"],
            dtype=bool,
        )
        if eligible.shape != expected_shape:
            raise ValueError(
                "profile_frame_split_eligible must have shape "
                f"{expected_shape}, got {eligible.shape}."
            )
        return eligible
    return np.ones(expected_shape, dtype=bool)


def _concatenate_optional(
    parts: list[dict[str, np.ndarray]],
    key: str,
) -> np.ndarray | None:
    if not all(key in part for part in parts):
        return None
    return np.concatenate([np.asarray(part[key]) for part in parts])


def _load_block_summary_json(
    archive: np.lib.npyio.NpzFile,
) -> tuple[dict[str, object], ...]:
    if "profile_block_summary_json" not in archive.files:
        return ()
    raw = np.asarray(archive["profile_block_summary_json"]).item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return tuple(json.loads(str(raw)))


def _merged_block_ids(parts: list[dict[str, np.ndarray]]) -> np.ndarray | None:
    if not all("block_id" in part for part in parts):
        return None
    merged = []
    offset = 0
    for part in parts:
        block_ids = np.asarray(part["block_id"], dtype=np.int64)
        if len(block_ids) == 0:
            continue
        merged.append(block_ids + offset)
        offset += int(np.max(block_ids)) + 1
    if not merged:
        return None
    return np.concatenate(merged).astype(np.int64, copy=False)


def _first_nonempty(values) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def _summarize_profile_session_blocks(
    session: ProfileSession,
) -> dict[str, object]:
    accepted = [
        summary
        for summary in session.block_summary
        if bool(summary.get("accepted", False))
    ]
    rejected = [
        summary
        for summary in session.block_summary
        if not bool(summary.get("accepted", False))
    ]
    movement_counts: dict[str, int] = {}
    for summary in accepted:
        name = str(summary.get("movement_name", "unknown"))
        movement_counts[name] = movement_counts.get(name, 0) + 1
    return {
        "posture": session.posture,
        "accepted_block_count": len(accepted),
        "rejected_block_count": len(rejected),
        "accepted_sample_count": int(len(session.eeg)),
        "movement_counts": movement_counts,
        "reject_reasons": _reject_reason_counts(rejected),
    }


def _reject_reason_counts(
    rejected: list[dict[str, object]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for summary in rejected:
        reason = str(summary.get("reject_reason", "unknown") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


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


def _support_query_indices_for_blocks(
    block_ids: np.ndarray,
    *,
    query_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    unique_blocks = _unique_in_order(block_ids)
    if len(unique_blocks) < 2:
        return _support_query_indices(len(block_ids), query_fraction=query_fraction)

    query_block_count = min(
        max(1, round(len(unique_blocks) * query_fraction)),
        len(unique_blocks) - 1,
    )
    query_blocks = set(unique_blocks[-query_block_count:])
    support_mask = np.asarray(
        [block_id not in query_blocks for block_id in block_ids],
        dtype=bool,
    )
    query_mask = ~support_mask
    return (
        np.flatnonzero(support_mask).astype(np.int64),
        np.flatnonzero(query_mask).astype(np.int64),
    )


def _unique_in_order(values: np.ndarray) -> list[int]:
    seen: set[int] = set()
    ordered = []
    for value in values:
        int_value = int(value)
        if int_value not in seen:
            ordered.append(int_value)
            seen.add(int_value)
    return ordered


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)
    return safe or "session"


@contextmanager
def _forked_torch_seed(seed: int | None):
    if seed is None:
        yield
        return
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        yield


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
    eeg = _session_eeg_contexts_for_indices(model, session, indices)
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
    eeg = _session_eeg_contexts_for_indices(model, session, indices)
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


def _session_eeg_contexts(
    model: EegPoseVAE,
    session: ProfileSession,
) -> tuple[np.ndarray, np.ndarray]:
    features = transform_eeg_for_model(session.eeg, model.config)
    contexts, target_indices = build_eligible_grouped_context_windows(
        features,
        session.round_roles,
        session.target_eligible,
        session.history_eligible,
        context_packet_count=model.config.context_packet_count,
    )
    expected_target_indices = np.flatnonzero(
        session.target_eligible
    ).astype(np.int64)
    assert np.array_equal(target_indices, expected_target_indices)
    return contexts, target_indices


def _session_eeg_contexts_for_indices(
    model: EegPoseVAE,
    session: ProfileSession,
    indices: np.ndarray,
) -> np.ndarray:
    contexts, target_indices = _session_eeg_contexts(model, session)
    context_position = np.full(len(session.eeg), -1, dtype=np.int64)
    context_position[target_indices] = np.arange(
        len(target_indices),
        dtype=np.int64,
    )
    positions = context_position[indices]
    assert np.all(positions >= 0)
    return contexts[positions]


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
            "session_block_summaries": report.session_block_summaries,
            "validation_passed": report.validation_passed,
            "holdout_passed": report.holdout_passed,
            "old_holdout_metrics": report.old_holdout_metrics,
            "new_holdout_metrics": report.new_holdout_metrics,
            "test_metrics": [asdict(metric) for metric in report.test_metrics],
            "holdout_update": report.holdout_update,
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
