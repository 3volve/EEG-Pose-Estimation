from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import eeg_encoding
from config import (
    ROOT_DIR,
    BASE_OUTPUT_DIR,
    COLLECT_DURATION_S,
    DEFAULT_DEVICE,
    EEG_CONTEXT_PACKET_COUNT,
    EEG_DECODED_POSITION_LOSS_WEIGHT,
    EEG_DECODED_VELOCITY_LOSS_WEIGHT,
    EEG_MODEL_BATCH_SIZE,
    EEG_MODEL_EPOCHS,
    EEG_MODEL_LR,
    EEG_MODEL_VAL_SPLIT,
    EEG_ENCODING_MODEL,
    EEG_DATA_GLOB,
    EEG_STANDARDIZE_POSE_LATENTS,
    EEG_STILLNESS_LOSS_WEIGHT,
    EEG_VALIDATE_BY_RUN,
    LIVE_PREDICT_DURATION_S,
    PAIRING_MAX_POSE_GAP_S,
    PAIRING_POLL_DELAY_S,
    POSE_CAMERA_INDEX,
    POSE_LIVE_MEAN_WINDOW,
    POSE_LIVE_MEDIAN_WINDOW,
    POSE_LIVE_SMOOTHING,
    POSE_MODEL,
    POSE_ENCODING_MODEL,
    POSE_TARGET_FPS,
)
from pose_encoding import PoseLatentStream
from streaming.eeg import SignalStreamer
from streaming.pose import AsyncPoseEstimator
from streaming import collect_and_save_paired_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EEG-to-pose-latent streaming and training commands."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect-paired")
    collect.add_argument("--out", required=True)
    collect.add_argument("--duration", type=float, default=COLLECT_DURATION_S)
    collect.add_argument("--pose-model", default=POSE_MODEL)
    collect.add_argument("--pose-checkpoint", default=POSE_ENCODING_MODEL)
    collect.add_argument("--camera-index", type=int, default=POSE_CAMERA_INDEX)
    collect.add_argument("--pose-fps", type=float, default=POSE_TARGET_FPS)
    collect.add_argument("--max-pose-gap", type=float, default=PAIRING_MAX_POSE_GAP_S)
    collect.add_argument("--mirror", action="store_true", default=False)
    collect.add_argument("--preview", action="store_true")
    collect.add_argument("--device", default=DEFAULT_DEVICE)

    train = subparsers.add_parser("train-eeg")
    train.add_argument("--data", default=EEG_DATA_GLOB)
    train.add_argument("--out", required=True)
    train.add_argument("--epochs", type=int, default=EEG_MODEL_EPOCHS)
    train.add_argument("--batch-size", type=int, default=EEG_MODEL_BATCH_SIZE)
    train.add_argument("--lr", type=float, default=EEG_MODEL_LR)
    train.add_argument("--val-split", type=float, default=EEG_MODEL_VAL_SPLIT)
    train.add_argument("--context-packets", type=int, default=EEG_CONTEXT_PACKET_COUNT)
    standardize = train.add_mutually_exclusive_group()
    standardize.add_argument(
        "--standardize-pose-latents",
        dest="standardize_pose_latents",
        action="store_true",
    )
    standardize.add_argument(
        "--no-standardize-pose-latents",
        dest="standardize_pose_latents",
        action="store_false",
    )
    train.set_defaults(standardize_pose_latents=EEG_STANDARDIZE_POSE_LATENTS)
    validation = train.add_mutually_exclusive_group()
    validation.add_argument(
        "--validate-by-run",
        dest="validate_by_run",
        action="store_true",
    )
    validation.add_argument(
        "--random-validation",
        dest="validate_by_run",
        action="store_false",
    )
    train.set_defaults(validate_by_run=EEG_VALIDATE_BY_RUN)
    train.add_argument("--checkpoint")
    train.add_argument("--pose-checkpoint", default=POSE_ENCODING_MODEL)
    train.add_argument(
        "--decoded-position-weight",
        type=float,
        default=EEG_DECODED_POSITION_LOSS_WEIGHT,
    )
    train.add_argument(
        "--decoded-velocity-weight",
        type=float,
        default=EEG_DECODED_VELOCITY_LOSS_WEIGHT,
    )
    train.add_argument(
        "--stillness-weight",
        type=float,
        default=EEG_STILLNESS_LOSS_WEIGHT,
    )
    train.add_argument("--device", default=DEFAULT_DEVICE)

    live = subparsers.add_parser("predict-live")
    live.add_argument("--model", default=EEG_ENCODING_MODEL)
    live.add_argument("--device", default=DEFAULT_DEVICE)
    live.add_argument("--duration", type=float, default=LIVE_PREDICT_DURATION_S)

    subparsers.add_parser("recalibrate")
    return parser.parse_args()


def collect_paired(args: argparse.Namespace) -> None:
    args.out = str(ROOT_DIR / "eeg_encoding" / "data" / args.out)
    
    eeg_stream = SignalStreamer()
    pose_estimator = AsyncPoseEstimator(
        model_path=args.pose_model,
        camera_index=args.camera_index,
        target_fps=args.pose_fps,
        mirror_frame=args.mirror,
        draw_preview=args.preview,
    )
    pose_stream = PoseLatentStream.from_checkpoint(
        pose_estimator,
        args.pose_checkpoint,
        device=args.device,
    )
    eeg_thread = threading.Thread(
        target=eeg_stream.start_streaming,
        name="eeg-stream",
        daemon=True,
    )

    try:
        pose_estimator.start()
        eeg_thread.start()
        frames = collect_and_save_paired_frames(
            eeg_stream,
            pose_stream,
            duration_s=args.duration,
            out_path=args.out,
            max_pose_gap_s=args.max_pose_gap,
            metadata={
                "pose_model": args.pose_model,
                "pose_checkpoint": args.pose_checkpoint,
                "pose_fps": args.pose_fps,
                "pose_live_smoothing": POSE_LIVE_SMOOTHING,
                "pose_live_median_window": POSE_LIVE_MEDIAN_WINDOW,
                "pose_live_mean_window": POSE_LIVE_MEAN_WINDOW,
            },
        )
    finally:
        eeg_stream.stop_streaming()
        pose_estimator.stop()
        eeg_thread.join(timeout=2.0)

    print(f"Saved {len(frames)} paired EEG/pose frames to {Path(args.out)}")


def train_eeg(args: argparse.Namespace) -> None:
    predictor = eeg_encoding.train_model(
        args.data,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        context_packet_count=args.context_packets,
        standardize_pose_latents_for_training=args.standardize_pose_latents,
        validate_by_run=args.validate_by_run,
        checkpoint_path=args.checkpoint,
        pose_checkpoint=args.pose_checkpoint,
        decoded_position_weight=args.decoded_position_weight,
        decoded_velocity_weight=args.decoded_velocity_weight,
        stillness_weight=args.stillness_weight,
        device=args.device,
    )
    if predictor.training_report is not None:
        print(eeg_encoding.format_training_report(predictor.training_report))
    print(f"Saved EEG pose-latent model to {Path(args.out)}")


def predict_live(args: argparse.Namespace) -> None:
    eeg_stream = SignalStreamer()
    predictor = eeg_encoding.load_model(args.model, device=args.device)
    eeg_thread = threading.Thread(
        target=eeg_stream.start_streaming,
        name="eeg-stream",
        daemon=True,
    )
    deadline = time.monotonic() + args.duration if args.duration > 0 else None

    try:
        eeg_thread.start()
        while deadline is None or time.monotonic() < deadline:
            packet = eeg_stream.pop_packet()
            if packet is None:
                time.sleep(PAIRING_POLL_DELAY_S)
                continue
            prediction = predictor.predict(packet)
            print(
                f"packet={prediction.packet_id} "
                f"time={prediction.target_time_s:.3f} "
                f"latent={prediction.predicted_latent}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        eeg_stream.stop_streaming()
        eeg_thread.join(timeout=2.0)


def main() -> None:
    args = parse_args()
    if args.command == "collect-paired":
        collect_paired(args)
    elif args.command == "train-eeg":
        train_eeg(args)
    elif args.command == "predict-live":
        predict_live(args)
    elif args.command == "recalibrate":
        raise NotImplementedError("recalibrate will be added after the base model path")


if __name__ == "__main__":
    main()
