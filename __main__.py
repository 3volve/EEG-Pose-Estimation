from __future__ import annotations

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import eeg_encoding
import torch
from config import (
    ROOT_DIR,
    BASE_OUTPUT_DIR,
    COLLECT_DURATION_S,
    DEFAULT_DEVICE,
    EEG_CONTEXT_PACKET_COUNT,
    EEG_ADAPTATION_MODE_ADAPTER_HEAD,
    EEG_ADAPTATION_MODE_ADAPTER_ONLY,
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
    EEG_ONLINE_CALIBRATION_BATCH_SIZE,
    EEG_ONLINE_CALIBRATION_MAX_SAMPLES,
    EEG_ONLINE_CALIBRATION_MIN_BATCH_SIZE,
    EEG_ONLINE_CALIBRATION_STEPS_PER_UPDATE,
    EEG_ONLINE_CALIBRATION_UPDATE_EVERY,
    EEG_PROFILE_BUILD_INNER_EPOCHS,
    EEG_PROFILE_BUILD_QUERY_EPOCHS,
    EEG_PROFILE_CREATION_DURATION_S,
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
    CalibrationDisplayStatus,
    CalibrationOverlayState,
    PoseLatentBuffer,
    collect_and_save_paired_frames,
    pair_packet,
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
    preview_mirror = preview.add_mutually_exclusive_group()
    preview_mirror.add_argument(
        "--mirror-preview",
        dest="mirror_preview",
        action="store_true",
    )
    preview_mirror.add_argument(
        "--no-mirror-preview",
        dest="mirror_preview",
        action="store_false",
    )
    preview.set_defaults(mirror_preview=True)
    preview.add_argument("--device", default=DEFAULT_DEVICE)

    live_calibration = subparsers.add_parser("calibrate-live")
    add_live_profile_args(
        live_calibration,
        default_duration=LIVE_PREDICT_DURATION_S,
        default_adaptation_mode=EEG_ADAPTATION_MODE_SESSION,
    )

    profile_build = subparsers.add_parser("profile-build")
    profile_build.add_argument("--user-id", required=True)
    profile_build.add_argument("--data", nargs="*")
    profile_build.add_argument("--base-checkpoint", default=EEG_ENCODING_MODEL)
    profile_build.add_argument("--epochs", type=int, default=EEG_MODEL_EPOCHS)
    profile_build.add_argument("--inner-epochs", type=int, default=EEG_PROFILE_BUILD_INNER_EPOCHS)
    profile_build.add_argument("--query-epochs", type=int, default=EEG_PROFILE_BUILD_QUERY_EPOCHS)
    profile_build.add_argument("--batch-size", type=int, default=EEG_MODEL_BATCH_SIZE)
    profile_build.add_argument("--lr", type=float, default=EEG_MODEL_LR)
    profile_build.add_argument(
        "--inner-adaptation-mode",
        choices=[EEG_ADAPTATION_MODE_ADAPTER_HEAD, EEG_ADAPTATION_MODE_SESSION],
        default=EEG_ADAPTATION_MODE_ADAPTER_HEAD,
    )
    profile_build.add_argument("--pose-model", default=POSE_MODEL)
    profile_build.add_argument("--pose-checkpoint", default=POSE_ENCODING_MODEL)
    profile_build.add_argument("--camera-index", type=int, default=POSE_CAMERA_INDEX)
    profile_build.add_argument("--pose-fps", type=float, default=POSE_TARGET_FPS)
    profile_build.add_argument("--duration", type=float, default=EEG_PROFILE_CREATION_DURATION_S)
    profile_build.add_argument("--max-pose-gap", type=float, default=PAIRING_MAX_POSE_GAP_S)
    profile_build.add_argument("--mirror-preview", dest="mirror_preview", action="store_true")
    profile_build.add_argument("--no-mirror-preview", dest="mirror_preview", action="store_false")
    profile_build.set_defaults(mirror_preview=True)
    profile_build.add_argument("--device", default=DEFAULT_DEVICE)

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


def add_live_profile_args(
    parser: argparse.ArgumentParser,
    *,
    default_duration: float,
    default_adaptation_mode: str,
) -> None:
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--base-checkpoint", default=EEG_ENCODING_MODEL)
    parser.add_argument("--session-id")
    parser.add_argument("--pose-model", default=POSE_MODEL)
    parser.add_argument("--pose-checkpoint", default=POSE_ENCODING_MODEL)
    parser.add_argument("--camera-index", type=int, default=POSE_CAMERA_INDEX)
    parser.add_argument("--pose-fps", type=float, default=POSE_TARGET_FPS)
    parser.add_argument("--duration", type=float, default=default_duration)
    parser.add_argument("--max-pose-gap", type=float, default=PAIRING_MAX_POSE_GAP_S)
    parser.add_argument("--lr", type=float, default=EEG_MODEL_LR)
    parser.add_argument("--batch-size", type=int, default=EEG_ONLINE_CALIBRATION_BATCH_SIZE)
    parser.add_argument("--min-batch-size", type=int, default=EEG_ONLINE_CALIBRATION_MIN_BATCH_SIZE)
    parser.add_argument("--update-every", type=int, default=EEG_ONLINE_CALIBRATION_UPDATE_EVERY)
    parser.add_argument("--steps-per-update", type=int, default=EEG_ONLINE_CALIBRATION_STEPS_PER_UPDATE)
    parser.add_argument("--max-samples", type=int, default=EEG_ONLINE_CALIBRATION_MAX_SAMPLES)
    parser.add_argument(
        "--adaptation-mode",
        choices=[
            EEG_ADAPTATION_MODE_ADAPTER_HEAD,
            EEG_ADAPTATION_MODE_ADAPTER_ONLY,
            EEG_ADAPTATION_MODE_SESSION,
        ],
        default=default_adaptation_mode,
    )
    mirror = parser.add_mutually_exclusive_group()
    mirror.add_argument("--mirror-preview", dest="mirror_preview", action="store_true")
    mirror.add_argument("--no-mirror-preview", dest="mirror_preview", action="store_false")
    parser.set_defaults(mirror_preview=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)


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
    start_checkpoint = eeg_encoding.profile_start_checkpoint(
        args.user_id,
        args.base_checkpoint,
    )
    report = eeg_encoding.train_session_model(
        user_id=args.user_id,
        base_checkpoint=start_checkpoint,
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


def profile_build(args: argparse.Namespace) -> None:
    paths = eeg_encoding.profile_paths(args.user_id)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_data: str | Path | list[str | Path]
    if args.data:
        session_data = args.data
    else:
        session_data = paths.sessions_root / session_id / "paired_profile_session.npz"
        collect_profile_build_session(args, session_data)

    report = eeg_encoding.build_profile_model(
        user_id=args.user_id,
        new_session_data=session_data,
        base_checkpoint=args.base_checkpoint,
        epochs=args.epochs,
        inner_epochs=args.inner_epochs,
        query_epochs=args.query_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        inner_adaptation_mode=args.inner_adaptation_mode,
        pose_checkpoint=args.pose_checkpoint,
        device=args.device,
    )
    status = "committed" if report.committed else "rejected"
    print(f"Profile build {status}: {report.profile_model}")
    print(f"Report: {Path(report.history_dir) / 'profile_build_report.json'}")


def collect_profile_build_session(
    args: argparse.Namespace,
    out_path: str | Path,
) -> None:
    eeg_stream = SignalStreamer()
    pose_estimator = AsyncPoseEstimator(
        model_path=args.pose_model,
        camera_index=args.camera_index,
        target_fps=args.pose_fps,
        mirror_frame=args.mirror_preview,
        draw_preview=True,
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
            out_path=str(out_path),
            max_pose_gap_s=args.max_pose_gap,
            metadata={
                "sample_kind": "profile_build_session",
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
    print(f"Saved {len(frames)} profile paired frames to {Path(out_path)}")


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
        draw_builtin_pose_overlay=False,
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


def calibrate_live(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    session_id = args.session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = eeg_encoding.profile_paths(args.user_id).sessions_root / session_id
    overlay_state = CalibrationOverlayState()
    eeg_stream = SignalStreamer()
    start_checkpoint = eeg_encoding.profile_start_checkpoint(
        args.user_id,
        args.base_checkpoint,
    )
    eeg_predictor = eeg_encoding.load_model(start_checkpoint, device=device)
    pose_decoder, _ = load_checkpoint(args.pose_checkpoint, map_location=device)
    calibrator = eeg_encoding.OnlineEegCalibrator(
        eeg_predictor.model,
        pose_decoder,
        device=device,
        adaptation_mode=args.adaptation_mode,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        min_batch_size=args.min_batch_size,
        update_every=args.update_every,
        steps_per_update=args.steps_per_update,
        max_samples=args.max_samples,
    )
    pose_estimator = AsyncPoseEstimator(
        model_path=args.pose_model,
        camera_index=args.camera_index,
        target_fps=args.pose_fps,
        mirror_frame=args.mirror_preview,
        draw_preview=True,
        draw_builtin_pose_overlay=False,
        preview_renderer=overlay_state.render,
    )
    pose_stream = PoseLatentStream.from_checkpoint(
        pose_estimator,
        args.pose_checkpoint,
        device=args.device,
    )
    pose_buffer = PoseLatentBuffer(max_gap_s=args.max_pose_gap)
    pending_packets = []
    eeg_thread = threading.Thread(
        target=eeg_stream.start_streaming,
        name="eeg-stream",
        daemon=True,
    )
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    block_index = 0
    block_started_at = time.monotonic()
    last_status_at = 0.0

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
            pose_buffer.add(pose_frame)
            if pose_frame is not None and pose_frame.pose_detected:
                overlay_state.update_truth(pose_frame.feature_vector)

            packet = eeg_stream.pop_packet()
            if packet is not None:
                pending_packets.append(packet)
                prediction = eeg_predictor.predict(packet)
                latent = torch.from_numpy(prediction.predicted_latent).to(device)
                with torch.no_grad():
                    decoded = pose_decoder.decode(latent).cpu().numpy()
                overlay_state.update_eeg(decoded.astype("float32", copy=False))

            still_pending = []
            latest_pose_time_s = pose_buffer.latest_time_s
            for pending_packet in pending_packets:
                paired = pair_packet(pending_packet, pose_buffer)
                if paired is not None:
                    calibrator.observe(paired)
                elif (
                    latest_pose_time_s is None
                    or latest_pose_time_s <= pending_packet.end_time_s
                ):
                    still_pending.append(pending_packet)
            pending_packets = still_pending
            status = calibrator.status()
            overlay_state.update_status(
                CalibrationDisplayStatus(
                    readiness_score=status.readiness.score,
                    ready=status.readiness.ready,
                    trusted_samples=status.trusted_samples,
                    skipped_samples=status.skipped_samples,
                    update_count=status.update_count,
                    latest_loss=status.latest_loss,
                )
            )

            if now - last_status_at >= 2.0:
                print(
                    "online calibration: "
                    f"trusted={status.trusted_samples} "
                    f"skipped={status.skipped_samples} "
                    f"updates={status.update_count} "
                    f"ready={status.readiness.ready} "
                    f"score={status.readiness.score:.3f} "
                    f"loss={status.latest_loss}"
                )
                last_status_at = now

            time.sleep(PAIRING_POLL_DELAY_S)
    except KeyboardInterrupt:
        pass
    finally:
        calibrator.save(session_dir)
        eeg_stream.stop_streaming()
        pose_estimator.stop()
        eeg_thread.join(timeout=2.0)
        status = calibrator.status()
        print(f"Saved live calibration session to {session_dir}")
        print(
            "final online calibration: "
            f"trusted={status.trusted_samples} "
            f"skipped={status.skipped_samples} "
            f"updates={status.update_count} "
            f"ready={status.readiness.ready} "
            f"score={status.readiness.score:.3f} "
            f"loss={status.latest_loss}"
        )


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
    elif args.command == "calibrate-live":
        calibrate_live(args)
    elif args.command == "profile-build":
        profile_build(args)
    elif args.command == "recalibrate":
        recalibrate(args)


if __name__ == "__main__":
    main()
