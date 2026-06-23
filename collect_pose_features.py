"""Collect normalized pose features from the webcam into a NumPy archive."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from pose_async import AsyncPoseEstimator
from pose_features import PoseFeatureExtractor, feature_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record webcam pose features for autoencoder training."
    )
    parser.add_argument(
        "--model",
        default="models/pose_landmarker_full.task",
        help="MediaPipe pose model path",
    )
    parser.add_argument("--out", default="pose_features.npz")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the webcam preview",
    )
    parser.add_argument(
        "--image-landmarks",
        action="store_true",
        help="Use image landmarks instead of preferring world landmarks",
    )
    parser.add_argument(
        "--no-velocity",
        action="store_true",
        help=(
            "Save 24 position features instead of 48 position/velocity "
            "features"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("duration must be greater than zero")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("min-confidence must be between zero and one")
    output_path = Path(args.out)
    if output_path.suffix.lower() != ".npz":
        raise ValueError(f"output dataset must be a .npz file: {output_path}")

    estimator = AsyncPoseEstimator(
        model_path=args.model,
        camera_index=args.camera_index,
        target_fps=args.fps,
        mirror_frame=args.mirror,
        draw_preview=not args.no_preview,
    )
    extractor = PoseFeatureExtractor(
        include_velocity=not args.no_velocity,
        use_world_landmarks=not args.image_landmarks,
    )
    vectors: list[np.ndarray] = []
    timestamp_ms: list[int] = []
    received_time_s: list[float] = []
    confidence: list[float] = []
    deadline = time.monotonic() + args.duration

    try:
        estimator.start()
        print(
            f"Recording for {args.duration:g} seconds. Move through the poses "
            "you want represented; press q in the preview to stop early."
        )
        for result in estimator.results():
            if time.monotonic() >= deadline:
                break
            frame = extractor.extract(result)
            if frame.pose_detected and frame.confidence >= args.min_confidence:
                vectors.append(frame.vector)
                timestamp_ms.append(frame.timestamp_ms)
                received_time_s.append(frame.received_time_s)
                confidence.append(frame.confidence)
                print(
                    f"\rvalid samples: {len(vectors)}",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        estimator.stop()
        print()

    if estimator.error is not None:
        raise estimator.error
    if not vectors:
        raise RuntimeError(
            "No valid pose samples were recorded; no dataset was written"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = np.stack(vectors).astype(np.float32, copy=False)
    expected_dim = feature_dim(not args.no_velocity)
    assert features.shape[1] == expected_dim
    np.savez_compressed(
        output_path,
        features=features,
        timestamp_ms=np.asarray(timestamp_ms, dtype=np.int64),
        received_time_s=np.asarray(received_time_s, dtype=np.float64),
        pose_detected=np.ones(len(vectors), dtype=np.bool_),
        confidence=np.asarray(confidence, dtype=np.float32),
        feature_dim=np.asarray(expected_dim, dtype=np.int64),
        include_velocity=np.asarray(not args.no_velocity),
        use_world_landmarks=np.asarray(not args.image_landmarks),
        mirror_frame=np.asarray(args.mirror),
        target_fps=np.asarray(args.fps, dtype=np.float32),
        min_confidence=np.asarray(args.min_confidence, dtype=np.float32),
        camera_index=np.asarray(args.camera_index, dtype=np.int64),
        model_path=np.asarray(args.model),
    )
    print(f"Saved {features.shape} pose feature archive to {output_path}")


if __name__ == "__main__":
    main()
