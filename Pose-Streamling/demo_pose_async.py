"""Command-line demo for AsyncPoseEstimator."""

from __future__ import annotations

import argparse
import time

from pose_streaming import AsyncPoseEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream webcam poses with MediaPipe Pose Landmarker Lite."
    )
    parser.add_argument("--model_path", help="Path to pose_landmarker_lite.task", default="./models/pose_landmarker_full.task")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the webcam preview with pose landmarks; press q to stop.",
        default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    estimator = AsyncPoseEstimator(
        model_path=args.model_path,
        camera_index=args.camera_index,
        target_fps=args.fps,
        mirror_frame=args.mirror,
        draw_preview=args.preview,
    )

    try:
        estimator.start()
        while estimator.is_running:
            result = estimator.get_latest()
            if result is not None:
                print(
                    f"\rpose timestamp={result.timestamp_ms} ms, "
                    f"detected={result.pose_detected}",
                    end="",
                    flush=True,
                )
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        estimator.stop()
        print()

    if estimator.error is not None:
        raise estimator.error


if __name__ == "__main__":
    main()
