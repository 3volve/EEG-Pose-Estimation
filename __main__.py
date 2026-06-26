from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import eeg_encoding
import torch
from config import (
    ROOT_DIR,
    BASE_OUTPUT_DIR,
    COLLECT_DURATION_S,
    DEFAULT_DEVICE,
    EEG_CONTEXT_PACKET_COUNT,
    EEG_ADAPTATION_MODE_PROFILE_ENCODER,
    EEG_ADAPTATION_MODE_PROFILE_FULL,
    EEG_ADAPTATION_MODE_PROFILE_HEAD,
    EEG_ADAPTATION_MODE_SESSION,
    EEG_ADAPTATION_MODE_SESSION_DEEP,
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
from pose_encoding import PoseLatentStream, load_checkpoint
from streaming.eeg import SignalStreamer
from streaming.pose import AsyncPoseEstimator
from streaming import (
    DEFAULT_CALIBRATION_BLOCKS,
    CalibrationOverlayState,
    collect_and_save_paired_frames,
)


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
    train.add_argument(
        "--adaptation-mode",
        choices=[
            EEG_ADAPTATION_MODE_SESSION,
            EEG_ADAPTATION_MODE_SESSION_DEEP,
            EEG_ADAPTATION_MODE_PROFILE_HEAD,
            EEG_ADAPTATION_MODE_PROFILE_ENCODER,
            EEG_ADAPTATION_MODE_PROFILE_FULL,
        ],
    )
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

    preview = subparsers.add_parser("calibration-preview")
    preview.add_argument("--eeg-model", default=EEG_ENCODING_MODEL)
    preview.add_argument("--pose-model", default=POSE_MODEL)
    preview.add_argument("--pose-checkpoint", default=POSE_ENCODING_MODEL)
    preview.add_argument("--camera-index", type=int, default=POSE_CAMERA_INDEX)
    preview.add_argument("--pose-fps", type=float, default=POSE_TARGET_FPS)
    preview.add_argument("--duration", type=float, default=LIVE_PREDICT_DURATION_S)
    preview.add_argument("--mirror-preview", action="store_true", default=True)
    preview.add_argument("--no-mirror-preview", dest="mirror_preview", action="store_false")
    preview.add_argument("--device", default=DEFAULT_DEVICE)

    recalibrate = subparsers.add_parser("recalibrate")
    recalibrate.add_argument("--user-id", required=True)
    recalibrate.add_argument("--data", default=EEG_DATA_GLOB)
    recalibrate.add_argument("--base-checkpoint", default=EEG_ENCODING_MODEL)
    recalibrate.add_argument("--session-id")
    recalibrate.add_argument("--epochs", type=int, default=EEG_MODEL_EPOCHS)
    recalibrate.add_argument("--batch-size", type=int, default=EEG_MODEL_BATCH_SIZE)
    recalibrate.add_argument("--lr", type=float, default=EEG_MODEL_LR)
    recalibrate.add_argument(
        "--adaptation-mode",
        choices=[
            EEG_ADAPTATION_MODE_SESSION,
            EEG_ADAPTATION_MODE_SESSION_DEEP,
            EEG_ADAPTATION_MODE_PROFILE_HEAD,
            EEG_ADAPTATION_MODE_PROFILE_ENCODER,
            EEG_ADAPTATION_MODE_PROFILE_FULL,
        ],
        default=EEG_ADAPTATION_MODE_SESSION,
    )
    recalibrate.add_argument("--pose-checkpoint", default=POSE_ENCODING_MODEL)
    recalibrate.add_argument("--device", default=DEFAULT_DEVICE)
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
        adaptation_mode=args.adaptation_mode,
        pose_checkpoint=args.pose_checkpoint,
        decoded_position_weight=args.decoded_position_weight,
        decoded_velocity_weight=args.decoded_velocity_weight,
        stillness_weight=args.stillness_weight,
        device=args.device,
    )
    if predictor.training_report is not None:
        print(eeg_encoding.format_training_report(predictor.training_report))
    print(f"Saved EEG pose-latent model to {Path(args.out)}")


def recalibrate(args: argparse.Namespace) -> None:
    report = eeg_encoding.train_session_model(
        user_id=args.user_id,
        base_checkpoint=args.base_checkpoint,
        calibration_data=args.data,
        session_id=args.session_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        adaptation_mode=args.adaptation_mode,
        pose_checkpoint=args.pose_checkpoint,
        device=args.device,
    )
    print(eeg_encoding.format_training_report(report))


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


def calibration_preview(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    overlay_state = CalibrationOverlayState()
    eeg_stream = SignalStreamer()
    eeg_predictor = eeg_encoding.load_model(args.eeg_model, device=device)
    pose_decoder, _ = load_checkpoint(args.pose_checkpoint, map_location=device)
    pose_estimator = AsyncPoseEstimator(
        model_path=args.pose_model,
        camera_index=args.camera_index,
        target_fps=args.pose_fps,
        mirror_frame=args.mirror_preview,
        draw_preview=True,
        preview_renderer=overlay_state.render,
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
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    block_index = 0
    block_started_at = time.monotonic()

    try:
        pose_estimator.start()
        eeg_thread.start()
        while deadline is None or time.monotonic() < deadline:
            now = time.monotonic()
            block = DEFAULT_CALIBRATION_BLOCKS[
                block_index % len(DEFAULT_CALIBRATION_BLOCKS)
            ]
            if now - block_started_at >= block.duration_s:
                block_index += 1
                block_started_at = now
                block = DEFAULT_CALIBRATION_BLOCKS[
                    block_index % len(DEFAULT_CALIBRATION_BLOCKS)
                ]
            overlay_state.update_dummy(block, now - block_started_at)

            pose_frame = pose_stream.get_latest()
            if pose_frame is not None and pose_frame.pose_detected:
                overlay_state.update_truth(pose_frame.feature_vector)

            packet = eeg_stream.pop_packet()
            if packet is not None:
                prediction = eeg_predictor.predict(packet)
                latent = torch.from_numpy(prediction.predicted_latent).to(device)
                with torch.no_grad():
                    decoded = pose_decoder.decode(latent).cpu().numpy()
                overlay_state.update_eeg(decoded.astype("float32", copy=False))

            time.sleep(PAIRING_POLL_DELAY_S)
    except KeyboardInterrupt:
        pass
    finally:
        eeg_stream.stop_streaming()
        pose_estimator.stop()
        eeg_thread.join(timeout=2.0)


def main() -> None:
    args = parse_args()
    if args.command == "collect-paired":
        collect_paired(args)
    elif args.command == "train-eeg":
        train_eeg(args)
    elif args.command == "predict-live":
        predict_live(args)
    elif args.command == "calibration-preview":
        calibration_preview(args)
    elif args.command == "recalibrate":
        recalibrate(args)


if __name__ == "__main__":
    main()
