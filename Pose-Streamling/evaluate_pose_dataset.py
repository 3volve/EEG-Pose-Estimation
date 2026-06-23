"""Evaluate pose feature archives and optional autoencoder checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from pose_autoencoder import load_checkpoint


LANDMARK_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)


@dataclass(frozen=True, slots=True)
class PoseArchive:
    path: Path
    features: NDArray[np.float32]
    timestamp_ms: NDArray[np.int64] | None
    confidence: NDArray[np.float32] | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CheckpointStats:
    archive: PoseArchive
    total_error: NDArray[np.float32]
    position_error: NDArray[np.float32]
    velocity_error: NDArray[np.float32] | None
    position_signal: NDArray[np.float32]
    velocity_signal: NDArray[np.float32] | None
    latent: NDArray[np.float32]
    latent_step: NDArray[np.float32]


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize pose .npz datasets and optionally evaluate a trained "
            "pose autoencoder checkpoint."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["data/*.npz"],
        help="Input .npz files or glob patterns",
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional pose autoencoder checkpoint to evaluate",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--gap-ms",
        type=_positive_float,
        default=75.0,
        help="Timestamp gap threshold for reporting dropped/late samples",
    )
    parser.add_argument(
        "--low-confidence",
        type=float,
        default=0.7,
        help="Confidence threshold for low-confidence sample counts",
    )
    parser.add_argument(
        "--outlier-percentile",
        type=float,
        default=99.5,
        help="Training-set percentile used for velocity outlier thresholds",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of worst archive-level findings to print",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archives = [load_archive(path) for path in resolve_paths(args.paths)]
    if not archives:
        raise RuntimeError("No pose feature archives matched the input paths")

    validate_archive_contracts(archives)
    thresholds = compute_training_thresholds(archives, args.outlier_percentile)
    print_dataset_report(archives, thresholds, args)

    if args.checkpoint:
        print()
        print_checkpoint_report(archives, args.checkpoint, args.device)


def resolve_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern))
        if matches:
            paths.extend(matches)
            continue
        path = Path(pattern)
        if path.exists():
            paths.append(path)
    return sorted(set(paths))


def load_archive(path: Path) -> PoseArchive:
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Pose dataset must be a .npz archive: {path}")
    try:
        with np.load(path) as dataset:
            if "features" not in dataset:
                raise ValueError("archive is missing required 'features' array")
            features = np.asarray(dataset["features"], dtype=np.float32)
            timestamp_ms = (
                np.asarray(dataset["timestamp_ms"], dtype=np.int64)
                if "timestamp_ms" in dataset
                else None
            )
            confidence = (
                np.asarray(dataset["confidence"], dtype=np.float32)
                if "confidence" in dataset
                else None
            )
            metadata = {
                key: scalar_or_array(dataset[key])
                for key in dataset.files
                if key not in {"features", "timestamp_ms", "confidence"}
            }
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load pose dataset {path}: {exc}") from exc

    if features.ndim != 2:
        raise ValueError(f"{path}: features must be 2D; got {features.shape}")
    if len(features) == 0:
        raise ValueError(f"{path}: features array is empty")
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: features contain non-finite values")
    if timestamp_ms is not None and timestamp_ms.shape != (len(features),):
        raise ValueError(
            f"{path}: timestamp_ms shape {timestamp_ms.shape} does not match "
            f"{len(features)} feature rows"
        )
    if confidence is not None and confidence.shape != (len(features),):
        raise ValueError(
            f"{path}: confidence shape {confidence.shape} does not match "
            f"{len(features)} feature rows"
        )
    return PoseArchive(path, features, timestamp_ms, confidence, metadata)


def scalar_or_array(value: NDArray[Any]) -> Any:
    if value.shape == ():
        return value.item()
    return value.tolist()


def validate_archive_contracts(archives: list[PoseArchive]) -> None:
    dims = {archive.features.shape[1] for archive in archives}
    if len(dims) != 1:
        raise ValueError(f"Feature dimensions are inconsistent: {sorted(dims)}")

    keys = (
        "feature_dim",
        "include_velocity",
        "use_world_landmarks",
        "mirror_frame",
        "target_fps",
        "model_path",
    )
    for key in keys:
        values = {
            archive.metadata[key]
            for archive in archives
            if key in archive.metadata
        }
        if len(values) > 1:
            raise ValueError(f"Archive metadata differs for {key}: {values}")


def compute_training_thresholds(
    archives: list[PoseArchive],
    percentile: float,
) -> dict[str, float]:
    training_archives = [
        archive for archive in archives if archive.path.name.startswith("train_")
    ]
    source = training_archives or archives
    velocity_values: list[NDArray[np.float32]] = []
    delta_velocity_values: list[NDArray[np.float32]] = []
    for archive in source:
        split = split_features(archive.features)
        if split["velocity"] is None:
            continue
        velocity = split["velocity"]
        velocity_values.append(np.linalg.norm(velocity, axis=2).reshape(-1))
        if len(velocity) > 1:
            delta_velocity_values.append(
                np.linalg.norm(np.diff(velocity, axis=0), axis=2).reshape(-1)
            )

    return {
        "velocity": percentile_or_nan(velocity_values, percentile),
        "delta_velocity": percentile_or_nan(delta_velocity_values, percentile),
    }


def split_features(features: NDArray[np.float32]) -> dict[str, NDArray[np.float32] | None]:
    if features.shape[1] == 48:
        return {
            "position": features[:, :24].reshape(len(features), 8, 3),
            "velocity": features[:, 24:].reshape(len(features), 8, 3),
        }
    if features.shape[1] == 24:
        return {
            "position": features.reshape(len(features), 8, 3),
            "velocity": None,
        }
    return {"position": None, "velocity": None}


def percentile_or_nan(values: list[NDArray[np.float32]], percentile: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.concatenate(values), percentile))


def print_dataset_report(
    archives: list[PoseArchive],
    thresholds: dict[str, float],
    args: argparse.Namespace,
) -> None:
    print("Dataset")
    print(f"  archives: {len(archives)}")
    print(f"  samples: {sum(len(archive.features) for archive in archives)}")
    print(f"  feature_dim: {archives[0].features.shape[1]}")
    print(
        "  outlier thresholds from training archives: "
        f"velocity>{thresholds['velocity']:.3f}, "
        f"delta_velocity>{thresholds['delta_velocity']:.3f}"
    )
    print()

    warnings: list[str] = []
    scored: list[tuple[float, PoseArchive, dict[str, Any]]] = []
    for archive in archives:
        stats = archive_stats(archive, thresholds, args)
        scored.append((stats["score"], archive, stats))
        warnings.extend(stats["warnings"])
        print_archive_stats(archive, stats)

    print()
    print("Worst archives")
    for _, archive, stats in sorted(scored, reverse=True)[: args.top]:
        print(
            f"  {archive.path.name}: score={stats['score']:.1f}, "
            f"low_conf={stats['low_confidence_count']}, "
            f"gaps={stats['gap_count']}, "
            f"high_delta_velocity={stats['high_delta_velocity_count']}"
        )

    print()
    if warnings:
        print("Warnings")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Warnings")
        print("  none")


def archive_stats(
    archive: PoseArchive,
    thresholds: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    features = archive.features
    timestamp_ms = archive.timestamp_ms
    confidence = archive.confidence
    split = split_features(features)
    position = split["position"]
    velocity = split["velocity"]
    dt = np.diff(timestamp_ms) / 1000.0 if timestamp_ms is not None else np.array([])
    duration_s = (
        float((timestamp_ms[-1] - timestamp_ms[0]) / 1000.0)
        if timestamp_ms is not None and len(timestamp_ms) > 1
        else 0.0
    )
    fps = (len(features) - 1) / duration_s if duration_s > 0 else float("nan")
    confidence_values = (
        confidence
        if confidence is not None
        else np.full(len(features), np.nan, dtype=np.float32)
    )
    low_confidence_count = int(
        np.sum(confidence_values < args.low_confidence)
    )
    gap_count = int(np.sum(dt * 1000.0 > args.gap_ms)) if len(dt) else 0

    high_velocity_count = 0
    high_delta_velocity_count = 0
    worst_velocity = (float("nan"), "", -1)
    worst_delta_velocity = (float("nan"), "", -1)
    if velocity is not None:
        velocity_mag = np.linalg.norm(velocity, axis=2)
        delta_velocity_mag = np.linalg.norm(np.diff(velocity, axis=0), axis=2)
        high_velocity_count = count_above(velocity_mag, thresholds["velocity"])
        high_delta_velocity_count = count_above(
            delta_velocity_mag,
            thresholds["delta_velocity"],
        )
        worst_velocity = worst_landmark_event(velocity_mag, timestamp_ms, 0)
        worst_delta_velocity = worst_landmark_event(
            delta_velocity_mag,
            timestamp_ms,
            1,
        )

    warnings = []
    if low_confidence_count:
        warnings.append(
            f"{archive.path.name}: {low_confidence_count} samples below "
            f"confidence {args.low_confidence}"
        )
    if gap_count:
        warnings.append(
            f"{archive.path.name}: {gap_count} timestamp gaps above "
            f"{args.gap_ms:g} ms"
        )
    if position is None:
        warnings.append(
            f"{archive.path.name}: unsupported feature shape {features.shape}"
        )

    score = (
        low_confidence_count
        + gap_count * 10
        + high_delta_velocity_count
        + high_velocity_count * 0.5
    )
    return {
        "duration_s": duration_s,
        "fps": fps,
        "confidence_median": nanpercentile(confidence_values, 50),
        "confidence_p05": nanpercentile(confidence_values, 5),
        "low_confidence_count": low_confidence_count,
        "gap_count": gap_count,
        "high_velocity_count": high_velocity_count,
        "high_delta_velocity_count": high_delta_velocity_count,
        "worst_velocity": worst_velocity,
        "worst_delta_velocity": worst_delta_velocity,
        "warnings": warnings,
        "score": score,
    }


def count_above(values: NDArray[np.float32], threshold: float) -> int:
    if not np.isfinite(threshold):
        return 0
    return int(np.sum(values > threshold))


def worst_landmark_event(
    values: NDArray[np.float32],
    timestamp_ms: NDArray[np.int64] | None,
    timestamp_offset: int,
) -> tuple[float, str, int]:
    if values.size == 0:
        return float("nan"), "", -1
    row, landmark = np.unravel_index(np.argmax(values), values.shape)
    timestamp = (
        int(timestamp_ms[min(row + timestamp_offset, len(timestamp_ms) - 1)])
        if timestamp_ms is not None
        else -1
    )
    return float(values[row, landmark]), LANDMARK_NAMES[landmark], timestamp


def nanpercentile(values: NDArray[np.float32], percentile: float) -> float:
    if len(values) == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanpercentile(values, percentile))


def print_archive_stats(archive: PoseArchive, stats: dict[str, Any]) -> None:
    worst_v, worst_v_landmark, worst_v_time = stats["worst_velocity"]
    worst_dv, worst_dv_landmark, worst_dv_time = stats["worst_delta_velocity"]
    print(
        f"{archive.path.name}: n={len(archive.features)}, "
        f"duration={stats['duration_s']:.1f}s, fps={stats['fps']:.1f}, "
        f"conf_med={stats['confidence_median']:.3f}, "
        f"conf_p05={stats['confidence_p05']:.3f}, "
        f"low_conf={stats['low_confidence_count']}, "
        f"gaps={stats['gap_count']}, "
        f"high_v={stats['high_velocity_count']}, "
        f"high_dv={stats['high_delta_velocity_count']}"
    )
    print(
        f"  worst velocity={worst_v:.3f} {worst_v_landmark} "
        f"at timestamp {worst_v_time}; "
        f"worst delta_velocity={worst_dv:.3f} {worst_dv_landmark} "
        f"at timestamp {worst_dv_time}"
    )


def print_checkpoint_report(
    archives: list[PoseArchive],
    checkpoint_path: str,
    device: str,
) -> None:
    model, config = load_checkpoint(checkpoint_path, map_location=device)
    expected_dim = model.input_dim
    mismatched = [
        archive.path.name
        for archive in archives
        if archive.features.shape[1] != expected_dim
    ]
    if mismatched:
        raise ValueError(
            f"Checkpoint expects {expected_dim} features, but these archives "
            f"differ: {mismatched}"
        )

    print("Checkpoint")
    print(f"  path: {checkpoint_path}")
    print(f"  input_dim: {model.input_dim}")
    print(f"  latent_dim: {model.latent_dim}")
    if "best_validation_loss" in config:
        print(f"  training best_validation_loss: {config['best_validation_loss']}")
    print()

    stats: list[CheckpointStats] = []
    for archive in archives:
        archive_stats = checkpoint_archive_stats(
            model,
            archive,
            torch.device(device),
        )
        stats.append(archive_stats)
        print_checkpoint_archive_stats(archive_stats)

    print()
    print_checkpoint_summary(stats)


def checkpoint_archive_stats(
    model: torch.nn.Module,
    archive: PoseArchive,
    device: torch.device,
) -> CheckpointStats:
    features = archive.features
    batch = torch.from_numpy(features).to(device)
    with torch.no_grad():
        reconstruction, latent = model(batch)
    reconstruction_np = reconstruction.cpu().numpy()
    latent_np = latent.cpu().numpy()
    abs_error = np.abs(reconstruction_np - features)
    total_error = np.mean(abs_error, axis=1).astype(
        np.float32,
        copy=False,
    )
    split = split_features(features)
    if split["position"] is None:
        raise ValueError(f"{archive.path}: unsupported feature shape {features.shape}")
    position_error = np.mean(abs_error[:, :24], axis=1).astype(
        np.float32,
        copy=False,
    )
    position_signal = np.mean(np.abs(features[:, :24]), axis=1).astype(
        np.float32,
        copy=False,
    )
    velocity_error = None
    velocity_signal = None
    if features.shape[1] == 48:
        velocity_error = np.mean(abs_error[:, 24:], axis=1).astype(
            np.float32,
            copy=False,
        )
        velocity_signal = np.mean(np.abs(features[:, 24:]), axis=1).astype(
            np.float32,
            copy=False,
        )
    if len(latent_np) > 1:
        latent_steps = np.linalg.norm(np.diff(latent_np, axis=0), axis=1)
    else:
        latent_steps = np.zeros(1, dtype=np.float32)
    return CheckpointStats(
        archive=archive,
        total_error=total_error,
        position_error=position_error,
        velocity_error=velocity_error,
        position_signal=position_signal,
        velocity_signal=velocity_signal,
        latent=latent_np.astype(np.float32, copy=False),
        latent_step=latent_steps.astype(np.float32, copy=False),
    )


def print_checkpoint_archive_stats(stats: CheckpointStats) -> None:
    velocity_part = ""
    if stats.velocity_error is not None and stats.velocity_signal is not None:
        velocity_part = (
            f", vel_mae_p95={np.percentile(stats.velocity_error, 95):.6f}"
            f", vel_rel_p95={relative_p95(stats.velocity_error, stats.velocity_signal):.3f}"
        )
    print(
        f"{stats.archive.path.name}: "
        f"recon_mae_med={np.percentile(stats.total_error, 50):.6f}, "
        f"p95={np.percentile(stats.total_error, 95):.6f}, "
        f"p99={np.percentile(stats.total_error, 99):.6f}, "
        f"pos_mae_p95={np.percentile(stats.position_error, 95):.6f}, "
        f"pos_rel_p95={relative_p95(stats.position_error, stats.position_signal):.3f}"
        f"{velocity_part}, "
        f"latent_step_p95={np.percentile(stats.latent_step, 95):.6f}, "
        f"latent_step_p99={np.percentile(stats.latent_step, 99):.6f}"
    )
    print_worst_source_breakdown(stats)


def print_worst_source_breakdown(stats: CheckpointStats) -> None:
    source_file = stats.archive.metadata.get("source_file")
    if source_file is None or len(source_file) != len(stats.total_error):
        return

    rows: list[tuple[float, str, int, float | None]] = []
    for source in sorted(set(source_file)):
        mask = np.asarray(source_file) == source
        if int(np.sum(mask)) < 10:
            continue
        velocity_p95 = (
            float(np.percentile(stats.velocity_error[mask], 95))
            if stats.velocity_error is not None
            else None
        )
        rows.append(
            (
                float(np.percentile(stats.total_error[mask], 95)),
                str(source),
                int(np.sum(mask)),
                velocity_p95,
            )
        )
    if not rows:
        return

    print("  worst sources by reconstruction p95:")
    for total_p95, source, count, velocity_p95 in sorted(rows, reverse=True)[:3]:
        velocity_text = (
            f", vel_mae_p95={velocity_p95:.6f}"
            if velocity_p95 is not None
            else ""
        )
        print(
            f"    {source}: n={count}, recon_p95={total_p95:.6f}"
            f"{velocity_text}"
        )


def print_checkpoint_summary(stats: list[CheckpointStats]) -> None:
    training = [item for item in stats if item.archive.path.name.startswith("train_")]
    holdout = [item for item in stats if not item.archive.path.name.startswith("train_")]
    if not training or not holdout:
        print("Trust summary")
        print("  Need both train_* and holdout archives for generalization scoring.")
        return

    train_total = concatenate_stat(training, "total_error")
    holdout_total = concatenate_stat(holdout, "total_error")
    train_position = concatenate_stat(training, "position_error")
    holdout_position = concatenate_stat(holdout, "position_error")
    train_velocity = concatenate_optional_stat(training, "velocity_error")
    holdout_velocity = concatenate_optional_stat(holdout, "velocity_error")
    holdout_position_signal = concatenate_stat(holdout, "position_signal")
    holdout_velocity_signal = concatenate_optional_stat(holdout, "velocity_signal")
    all_latent = np.concatenate([item.latent for item in stats], axis=0)
    all_latent_step = np.concatenate([item.latent_step for item in stats])

    total_ratio = p95_ratio(holdout_total, train_total)
    position_ratio = p95_ratio(holdout_position, train_position)
    velocity_ratio = (
        p95_ratio(holdout_velocity, train_velocity)
        if holdout_velocity is not None and train_velocity is not None
        else float("nan")
    )
    position_relative = relative_p95(holdout_position, holdout_position_signal)
    velocity_relative = (
        relative_p95(holdout_velocity, holdout_velocity_signal)
        if holdout_velocity is not None and holdout_velocity_signal is not None
        else float("nan")
    )
    latent_std = np.std(all_latent, axis=0)
    active_latent_dims = int(np.sum(latent_std > max(np.max(latent_std) * 0.05, 1e-6)))
    top_latent_share = latent_top_variance_share(all_latent)
    confidence = trust_score(
        total_ratio,
        position_ratio,
        velocity_ratio,
        position_relative,
        velocity_relative,
        active_latent_dims,
        all_latent.shape[1],
        top_latent_share,
    )

    print("Trust summary")
    print(
        f"  holdout/train p95 ratio: total={total_ratio:.3f}, "
        f"position={position_ratio:.3f}, velocity={velocity_ratio:.3f}"
    )
    print(
        f"  holdout relative p95 error: position={position_relative:.3f}, "
        f"velocity={velocity_relative:.3f}"
    )
    print(
        f"  latent health: active_dims={active_latent_dims}/{all_latent.shape[1]}, "
        f"top_dim_variance_share={top_latent_share:.3f}, "
        f"latent_step_p95={np.percentile(all_latent_step, 95):.6f}"
    )
    print(f"  pose-encoder trust score: {confidence:.1f}%")


def concatenate_stat(
    stats: list[CheckpointStats],
    attribute: str,
) -> NDArray[np.float32]:
    return np.concatenate([getattr(item, attribute) for item in stats])


def concatenate_optional_stat(
    stats: list[CheckpointStats],
    attribute: str,
) -> NDArray[np.float32] | None:
    values = [getattr(item, attribute) for item in stats]
    if any(value is None for value in values):
        return None
    return np.concatenate(values)


def p95_ratio(
    numerator: NDArray[np.float32],
    denominator: NDArray[np.float32],
) -> float:
    denominator_p95 = float(np.percentile(denominator, 95))
    if denominator_p95 <= 0:
        return float("inf")
    return float(np.percentile(numerator, 95) / denominator_p95)


def relative_p95(
    error: NDArray[np.float32],
    signal: NDArray[np.float32],
) -> float:
    signal_p95 = float(np.percentile(signal, 95))
    if signal_p95 <= 0:
        return float("inf")
    return float(np.percentile(error, 95) / signal_p95)


def latent_top_variance_share(latent: NDArray[np.float32]) -> float:
    variances = np.var(latent, axis=0)
    total = float(np.sum(variances))
    if total <= 0:
        return 1.0
    return float(np.max(variances) / total)


def trust_score(
    total_ratio: float,
    position_ratio: float,
    velocity_ratio: float,
    position_relative: float,
    velocity_relative: float,
    active_latent_dims: int,
    latent_dim: int,
    top_latent_share: float,
) -> float:
    # This is a pose-encoder trust score, not an EEG predictability score.
    score = 90.0
    score -= penalty_above(total_ratio, good=1.15, bad=1.8, maximum=14.0)
    score -= penalty_above(position_ratio, good=1.15, bad=1.8, maximum=10.0)
    score -= penalty_above(velocity_ratio, good=1.25, bad=2.0, maximum=16.0)
    score -= penalty_above(position_relative, good=0.08, bad=0.2, maximum=10.0)
    score -= penalty_above(velocity_relative, good=0.18, bad=0.45, maximum=18.0)

    active_fraction = active_latent_dims / max(latent_dim, 1)
    if active_fraction < 0.5:
        score -= 12.0
    elif active_fraction < 0.75:
        score -= 6.0
    score -= penalty_above(top_latent_share, good=0.45, bad=0.8, maximum=8.0)
    return float(np.clip(score, 0.0, 90.0))


def penalty_above(
    value: float,
    *,
    good: float,
    bad: float,
    maximum: float,
) -> float:
    if not np.isfinite(value):
        return maximum
    if value <= good:
        return 0.0
    if value >= bad:
        return maximum
    return maximum * ((value - good) / (bad - good))


if __name__ == "__main__":
    main()
