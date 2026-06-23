"""Prepare pose feature archives for autoencoder training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_odd_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0 or parsed % 2 == 0:
        raise argparse.ArgumentTypeError("must be a positive odd integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trim, filter, smooth, and combine pose .npz archives. Positions "
            "are smoothed first; velocity features are then recomputed from "
            "the smoothed positions and original timestamps."
        )
    )
    parser.add_argument("paths", nargs="+", help="Input .npz files or globs")
    parser.add_argument("--out", required=True, help="Output .npz archive")
    parser.add_argument(
        "--drop-start-seconds",
        type=_nonnegative_float,
        default=3.0,
        help="Drop this much time from the start of each input archive",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Drop samples below this aggregate landmark confidence",
    )
    parser.add_argument(
        "--gap-ms",
        type=_positive_float,
        default=75.0,
        help="Split smoothing segments across timestamp gaps above this size",
    )
    parser.add_argument(
        "--median-window",
        type=_positive_odd_int,
        default=5,
        help="Centered median window for removing isolated landmark jumps",
    )
    parser.add_argument(
        "--mean-window",
        type=_positive_odd_int,
        default=5,
        help="Centered moving-average window for baseline jitter reduction",
    )
    parser.add_argument(
        "--min-segment-samples",
        type=int,
        default=5,
        help="Drop filtered contiguous segments shorter than this",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("min-confidence must be between zero and one")
    if args.min_segment_samples <= 0:
        raise ValueError("min-segment-samples must be greater than zero")

    output_path = Path(args.out)
    if output_path.suffix.lower() != ".npz":
        raise ValueError(f"output dataset must be a .npz file: {output_path}")

    input_paths = resolve_paths(args.paths)
    if not input_paths:
        raise RuntimeError("No input pose archives matched the provided paths")

    prepared = prepare_archives(input_paths, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **prepared)
    print(
        f"Saved {prepared['features'].shape} prepared pose feature archive "
        f"to {output_path}"
    )
    print(
        f"Dropped startup={prepared['dropped_start_count'].item()}, "
        f"low_confidence={prepared['dropped_low_confidence_count'].item()}, "
        f"short_segment={prepared['dropped_short_segment_count'].item()}"
    )


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


def prepare_archives(
    paths: list[Path],
    args: argparse.Namespace,
) -> dict[str, NDArray[Any]]:
    output_features: list[NDArray[np.float32]] = []
    output_confidence: list[NDArray[np.float32]] = []
    output_source_file: list[NDArray[np.str_]] = []
    output_source_index: list[NDArray[np.int64]] = []
    output_source_timestamp_ms: list[NDArray[np.int64]] = []
    output_source_segment: list[NDArray[np.int64]] = []
    metadata_values: dict[str, Any] = {}
    synthetic_timestamps: list[int] = []
    dropped_start = 0
    dropped_low_confidence = 0
    dropped_short_segment = 0
    original_sample_count = 0
    next_segment_id = 0
    next_timestamp_ms = 0

    for path in paths:
        archive = load_archive(path)
        original_sample_count += len(archive["features"])
        update_metadata_contract(metadata_values, archive["metadata"], path)

        keep, start_drop, confidence_drop = sample_keep_mask(archive, args)
        dropped_start += start_drop
        dropped_low_confidence += confidence_drop

        for segment in contiguous_segments(archive["timestamp_ms"], keep, args.gap_ms):
            if len(segment) < args.min_segment_samples:
                dropped_short_segment += len(segment)
                continue

            prepared_positions = smooth_positions(
                archive["positions"][segment],
                args.median_window,
                args.mean_window,
            )
            prepared_features = rebuild_features(
                prepared_positions,
                archive["timestamp_ms"][segment],
                archive["include_velocity"],
            )
            output_features.append(prepared_features)
            output_confidence.append(archive["confidence"][segment])
            output_source_file.append(
                np.full(len(segment), path.name, dtype=f"<U{len(path.name)}")
            )
            output_source_index.append(segment.astype(np.int64, copy=False))
            output_source_timestamp_ms.append(archive["timestamp_ms"][segment])
            output_source_segment.append(
                np.full(len(segment), next_segment_id, dtype=np.int64)
            )
            synthetic_timestamps.extend(
                synthetic_segment_timestamps(
                    archive["timestamp_ms"][segment],
                    next_timestamp_ms,
                    archive["target_fps"],
                )
            )
            next_timestamp_ms = synthetic_timestamps[-1] + frame_interval_ms(
                archive["target_fps"]
            )
            next_segment_id += 1

    if not output_features:
        raise RuntimeError("No samples remained after filtering")

    features = np.concatenate(output_features).astype(np.float32, copy=False)
    confidence = np.concatenate(output_confidence).astype(np.float32, copy=False)
    source_timestamp_ms = np.concatenate(output_source_timestamp_ms).astype(
        np.int64,
        copy=False,
    )
    metadata = {
        key: np.asarray(value)
        for key, value in metadata_values.items()
    }
    return {
        "features": features,
        "timestamp_ms": np.asarray(synthetic_timestamps, dtype=np.int64),
        "source_timestamp_ms": source_timestamp_ms,
        "source_file": np.concatenate(output_source_file),
        "source_index": np.concatenate(output_source_index).astype(
            np.int64,
            copy=False,
        ),
        "source_segment": np.concatenate(output_source_segment).astype(
            np.int64,
            copy=False,
        ),
        "pose_detected": np.ones(len(features), dtype=np.bool_),
        "confidence": confidence,
        "feature_dim": np.asarray(features.shape[1], dtype=np.int64),
        "prepared": np.asarray(True),
        "input_paths": np.asarray([str(path) for path in paths]),
        "original_sample_count": np.asarray(original_sample_count, dtype=np.int64),
        "kept_sample_count": np.asarray(len(features), dtype=np.int64),
        "dropped_start_count": np.asarray(dropped_start, dtype=np.int64),
        "dropped_low_confidence_count": np.asarray(
            dropped_low_confidence,
            dtype=np.int64,
        ),
        "dropped_short_segment_count": np.asarray(
            dropped_short_segment,
            dtype=np.int64,
        ),
        "drop_start_seconds": np.asarray(args.drop_start_seconds, dtype=np.float32),
        "smoothing_median_window": np.asarray(args.median_window, dtype=np.int64),
        "smoothing_mean_window": np.asarray(args.mean_window, dtype=np.int64),
        "filter_min_confidence": np.asarray(args.min_confidence, dtype=np.float32),
        "filter_gap_ms": np.asarray(args.gap_ms, dtype=np.float32),
        "min_segment_samples": np.asarray(args.min_segment_samples, dtype=np.int64),
        **metadata,
    }


def load_archive(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Pose dataset must be a .npz archive: {path}")
    try:
        with np.load(path) as dataset:
            if "features" not in dataset:
                raise ValueError("archive is missing required 'features' array")
            features = np.asarray(dataset["features"], dtype=np.float32)
            timestamp_ms = required_array(dataset, "timestamp_ms", path, np.int64)
            confidence = required_array(dataset, "confidence", path, np.float32)
            metadata = {
                key: scalar_or_array(dataset[key])
                for key in dataset.files
                if key not in {"features", "timestamp_ms", "confidence"}
            }
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load pose dataset {path}: {exc}") from exc

    if features.ndim != 2:
        raise ValueError(f"{path}: features must be 2D; got {features.shape}")
    if features.shape[1] not in {24, 48}:
        raise ValueError(
            f"{path}: expected 24 or 48 pose features; got {features.shape[1]}"
        )
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: features contain non-finite values")
    if timestamp_ms.shape != (len(features),):
        raise ValueError(f"{path}: timestamp_ms does not match feature rows")
    if confidence.shape != (len(features),):
        raise ValueError(f"{path}: confidence does not match feature rows")

    include_velocity = features.shape[1] == 48
    if "include_velocity" in metadata:
        include_velocity = bool(metadata["include_velocity"])
    target_fps = float(metadata.get("target_fps", 30.0))
    if target_fps <= 0:
        raise ValueError(f"{path}: target_fps must be greater than zero")

    return {
        "features": features,
        "positions": features[:, :24].reshape(len(features), 8, 3),
        "timestamp_ms": timestamp_ms,
        "confidence": confidence,
        "include_velocity": include_velocity,
        "target_fps": target_fps,
        "metadata": metadata,
    }


def required_array(
    dataset: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
    dtype: type,
) -> NDArray[Any]:
    if key not in dataset:
        raise ValueError(f"{path}: archive is missing required '{key}' array")
    return np.asarray(dataset[key], dtype=dtype)


def scalar_or_array(value: NDArray[Any]) -> Any:
    if value.shape == ():
        return value.item()
    return value.tolist()


def update_metadata_contract(
    metadata_values: dict[str, Any],
    metadata: dict[str, Any],
    path: Path,
) -> None:
    for key in (
        "include_velocity",
        "use_world_landmarks",
        "mirror_frame",
        "target_fps",
        "model_path",
        "camera_index",
    ):
        if key not in metadata:
            continue
        value = metadata[key]
        if key in metadata_values and metadata_values[key] != value:
            raise ValueError(
                f"{path}: metadata '{key}' differs from earlier inputs "
                f"({value!r} != {metadata_values[key]!r})"
            )
        metadata_values[key] = value


def sample_keep_mask(
    archive: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[NDArray[np.bool_], int, int]:
    timestamps = archive["timestamp_ms"]
    confidence = archive["confidence"]
    keep_start = timestamps - timestamps[0] >= args.drop_start_seconds * 1000.0
    keep_confidence = confidence >= args.min_confidence
    keep = keep_start & keep_confidence
    return (
        keep,
        int(np.sum(~keep_start)),
        int(np.sum(keep_start & ~keep_confidence)),
    )


def contiguous_segments(
    timestamp_ms: NDArray[np.int64],
    keep: NDArray[np.bool_],
    gap_ms: float,
) -> list[NDArray[np.int64]]:
    kept_indices = np.flatnonzero(keep)
    if len(kept_indices) == 0:
        return []

    segments: list[list[int]] = [[int(kept_indices[0])]]
    for index in kept_indices[1:]:
        previous = segments[-1][-1]
        is_adjacent = index == previous + 1
        gap = timestamp_ms[index] - timestamp_ms[previous]
        if is_adjacent and gap <= gap_ms:
            segments[-1].append(int(index))
        else:
            segments.append([int(index)])
    return [np.asarray(segment, dtype=np.int64) for segment in segments]


def smooth_positions(
    positions: NDArray[np.float32],
    median_window: int,
    mean_window: int,
) -> NDArray[np.float32]:
    smoothed = rolling_median(positions, median_window)
    smoothed = rolling_mean(smoothed, mean_window)
    return smoothed.astype(np.float32, copy=False)


def rolling_median(
    values: NDArray[np.float32],
    window: int,
) -> NDArray[np.float32]:
    if window == 1 or len(values) == 1:
        return values.astype(np.float32, copy=True)
    padded = edge_pad_time(values, window)
    windows = np.stack(
        [padded[offset : offset + len(values)] for offset in range(window)],
        axis=0,
    )
    return np.median(windows, axis=0).astype(np.float32, copy=False)


def rolling_mean(
    values: NDArray[np.float32],
    window: int,
) -> NDArray[np.float32]:
    if window == 1 or len(values) == 1:
        return values.astype(np.float32, copy=True)
    padded = edge_pad_time(values, window)
    total = np.zeros_like(values, dtype=np.float32)
    for offset in range(window):
        total += padded[offset : offset + len(values)]
    return total / float(window)


def edge_pad_time(
    values: NDArray[np.float32],
    window: int,
) -> NDArray[np.float32]:
    radius = window // 2
    return np.pad(
        values,
        ((radius, radius), (0, 0), (0, 0)),
        mode="edge",
    )


def rebuild_features(
    positions: NDArray[np.float32],
    timestamp_ms: NDArray[np.int64],
    include_velocity: bool,
) -> NDArray[np.float32]:
    flat_positions = positions.reshape(len(positions), 24)
    if not include_velocity:
        return flat_positions.astype(np.float32, copy=False)

    velocity = np.zeros_like(flat_positions, dtype=np.float32)
    elapsed_s = np.diff(timestamp_ms).astype(np.float32) / 1000.0
    if not np.all(elapsed_s > 0):
        raise ValueError("timestamps must increase within each segment")
    velocity[1:] = np.diff(flat_positions, axis=0) / elapsed_s[:, None]
    return np.concatenate((flat_positions, velocity), axis=1).astype(
        np.float32,
        copy=False,
    )


def synthetic_segment_timestamps(
    source_timestamp_ms: NDArray[np.int64],
    start_timestamp_ms: int,
    target_fps: float,
) -> list[int]:
    timestamps = [start_timestamp_ms]
    if len(source_timestamp_ms) == 1:
        return timestamps

    fallback_step = frame_interval_ms(target_fps)
    for elapsed_ms in np.diff(source_timestamp_ms):
        step = int(elapsed_ms)
        timestamps.append(timestamps[-1] + max(1, step or fallback_step))
    return timestamps


def frame_interval_ms(target_fps: float) -> int:
    return max(1, round(1000.0 / target_fps))


if __name__ == "__main__":
    main()
