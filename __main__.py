from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import eeg_encoding
import numpy as np
import torch
from config import (
    ROOT_DIR,
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
    EEG_CALIBRATION_MAX_POSE_RECONSTRUCTION_ERROR,
    EEG_CALIBRATION_MIN_INTERPOLATION_CONFIDENCE,
    EEG_CALIBRATION_MIN_POSE_CONFIDENCE,
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
    EEG_PACKET_SIZE,
    EEG_PACKET_STRIDE,
    EEG_PROFILE_BUILD_INNER_EPOCHS,
    EEG_PROFILE_BUILD_QUERY_EPOCHS,
    EEG_PROFILE_CREATION_DURATION_S,
    EEG_SAMPLE_RATE,
    EEG_ACQUISITION_TIMEOUT_S,
    EEG_ENCODING_MODEL,
    EEG_DATA_GLOB,
    EEG_STANDARDIZE_POSE_LATENTS,
    EEG_STILLNESS_LOSS_WEIGHT,
    EEG_VALIDATE_BY_RUN,
    LIVE_PREDICT_DURATION_S,
    PAIRING_MAX_POSE_GAP_S,
    PAIRING_POLL_DELAY_S,
    POSE_CAMERA_INDEX,
    POSE_INCLUDE_VELOCITY,
    POSE_LIVE_MEAN_WINDOW,
    POSE_LIVE_MEDIAN_WINDOW,
    POSE_LIVE_SMOOTHING,
    POSE_MODEL,
    POSE_ENCODING_MODEL,
    POSE_TARGET_FPS,
    POSE_USE_WORLD_LANDMARKS,
)
from pose_encoding import PoseLatentStream, load_checkpoint
from streaming.eeg import SignalStreamer
from streaming.pose import AsyncPoseEstimator
from streaming import (
    DEFAULT_CALIBRATION_BLOCKS,
    CalibrationDisplayStatus,
    CalibrationMovementBlock,
    CalibrationOverlayState,
    NeutralRestGate,
    PROFILE_MOVEMENT_TITLE_DELAY_S,
    PROFILE_MOVEMENT_TITLE_FADE_S,
    PROFILE_MOVEMENT_TITLES,
    PROFILE_REST_HOLD_DURATION_S,
    PROFILE_REST_VELOCITY_WINDOW_S,
    PROFILE_REST_VIOLATION_GRACE_S,
    PROFILE_REST_VELOCITY_THRESHOLD,
    PROFILE_REST_WRIST_TOLERANCE,
    PoseLatentBuffer,
    ProfileBlockResult,
    collect_and_save_paired_frames,
    profile_build_sequence,
    pair_packet,
    save_paired_frames,
    validate_profile_block,
)
from streaming.calibration import PROFILE_BUILD_ROLES
from streaming.debug_capture import (
    RawPoseEegDebugCapture,
    raw_pose_eeg_debug_path,
)
from eeg_encoding.permanent_holdout import (
    PERMANENT_HOLDOUT_CATEGORIES,
    candidates_from_test_blocks,
    create_manifest_from_first_session,
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
    _add_raw_pose_eeg_debug_argument(collect)

    verify_stream = subparsers.add_parser("verify-eeg-stream")
    verify_stream.add_argument("--duration", type=float, default=5.0)

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
    profile_build.add_argument("--posture", choices=["standing", "sitting"], default="standing")
    profile_build.add_argument("--max-pose-gap", type=float, default=PAIRING_MAX_POSE_GAP_S)
    profile_build.add_argument("--mirror-preview", dest="mirror_preview", action="store_true")
    profile_build.add_argument("--no-mirror-preview", dest="mirror_preview", action="store_false")
    profile_build.set_defaults(mirror_preview=True)
    profile_build.add_argument("--device", default=DEFAULT_DEVICE)
    _add_raw_pose_eeg_debug_argument(profile_build)

    bootstrap = subparsers.add_parser("bootstrap-base")
    bootstrap.add_argument("--user-id", required=True)
    bootstrap.add_argument("--out", required=True)
    bootstrap.add_argument("--epochs", type=int, default=EEG_MODEL_EPOCHS)
    bootstrap.add_argument("--inner-epochs", type=int, default=EEG_PROFILE_BUILD_INNER_EPOCHS)
    bootstrap.add_argument("--batch-size", type=int, default=EEG_MODEL_BATCH_SIZE)
    bootstrap.add_argument("--lr", type=float, default=EEG_MODEL_LR)
    bootstrap.add_argument("--pose-model", default=POSE_MODEL)
    bootstrap.add_argument("--pose-checkpoint", default=POSE_ENCODING_MODEL)
    bootstrap.add_argument("--camera-index", type=int, default=POSE_CAMERA_INDEX)
    bootstrap.add_argument("--pose-fps", type=float, default=POSE_TARGET_FPS)
    bootstrap.add_argument("--duration", type=float, default=EEG_PROFILE_CREATION_DURATION_S)
    bootstrap.add_argument("--posture", choices=["standing", "sitting"], default="standing")
    bootstrap.add_argument("--max-pose-gap", type=float, default=PAIRING_MAX_POSE_GAP_S)
    bootstrap.add_argument("--mirror-preview", dest="mirror_preview", action="store_true")
    bootstrap.add_argument("--no-mirror-preview", dest="mirror_preview", action="store_false")
    bootstrap.set_defaults(mirror_preview=True)
    bootstrap.add_argument("--device", default=DEFAULT_DEVICE)
    _add_raw_pose_eeg_debug_argument(bootstrap)

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


def _add_raw_pose_eeg_debug_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--save-raw-pose-eeg-debug",
        action="store_true",
        help=(
            "Write an additional native-rate EEG/MediaPipe sidecar for offline "
            "pose-representation experiments."
        ),
    )


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
    debug_capture = _raw_debug_capture(args)
    eeg_stream = SignalStreamer(
        corrected_sample_observer=(
            debug_capture.record_corrected_eeg if debug_capture is not None else None
        ),
        raw_lsl_observer=(
            debug_capture.record_raw_lsl if debug_capture is not None else None
        ),
        clock_observer=(
            debug_capture.record_eeg_clock if debug_capture is not None else None
        ),
    )
    pose_estimator = AsyncPoseEstimator(
        model_path=args.pose_model,
        camera_index=args.camera_index,
        target_fps=args.pose_fps,
        mirror_frame=args.mirror,
        draw_preview=args.preview,
        result_observer=(
            debug_capture.record_native_pose if debug_capture is not None else None
        ),
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
    metadata = {
        **eeg_stream.preprocessing_signature,
        **eeg_stream.timing_signature,
        "pose_model": args.pose_model,
        "pose_checkpoint": args.pose_checkpoint,
        "pose_camera_index": args.camera_index,
        "pose_fps": args.pose_fps,
        "pose_mirror_frame": args.mirror,
        "pose_include_velocity": POSE_INCLUDE_VELOCITY,
        "pose_use_world_landmarks": POSE_USE_WORLD_LANDMARKS,
        "pose_live_smoothing": POSE_LIVE_SMOOTHING,
        "pose_live_median_window": POSE_LIVE_MEDIAN_WINDOW,
        "pose_live_mean_window": POSE_LIVE_MEAN_WINDOW,
        "pairing_max_pose_gap_s": args.max_pose_gap,
        "pairing_pose_time_basis": "capture_timestamp_ms",
    }

    try:
        pose_estimator.start()
        eeg_thread.start()
        frames = collect_and_save_paired_frames(
            eeg_stream,
            pose_stream,
            duration_s=args.duration,
            out_path=args.out,
            max_pose_gap_s=args.max_pose_gap,
            metadata=metadata,
            debug_capture=debug_capture,
        )
    finally:
        eeg_stream.stop_streaming()
        pose_estimator.stop()
        eeg_thread.join(timeout=2.0)

    if debug_capture is not None:
        debug_path = debug_capture.save(
            raw_pose_eeg_debug_path(args.out),
            paired_frames=frames,
            metadata={**metadata, "training_archive": str(Path(args.out))},
        )
        print(f"Saved raw pose/EEG debug sidecar to {debug_path}")

    print(f"Saved {len(frames)} paired EEG/pose frames to {Path(args.out)}")


def verify_eeg_stream(args: argparse.Namespace) -> None:
    summary = _collect_eeg_stream_verification(
        SignalStreamer(),
        duration_s=args.duration,
    )
    print(json.dumps(summary, indent=2))


def _collect_eeg_stream_verification(
    eeg_stream,
    *,
    duration_s: float,
    poll_delay_s: float = PAIRING_POLL_DELAY_S,
) -> dict[str, object]:
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("EEG stream verification duration must be positive and finite.")

    packets = []
    stream_errors: list[Exception] = []

    def run_stream() -> None:
        try:
            eeg_stream.start_streaming()
        except Exception as error:
            stream_errors.append(error)

    eeg_thread = threading.Thread(
        target=run_stream,
        name="eeg-stream-verification",
        daemon=True,
    )
    deadline = time.monotonic() + duration_s
    try:
        eeg_thread.start()
        while time.monotonic() < deadline and not stream_errors:
            while (packet := eeg_stream.pop_packet()) is not None:
                packets.append(packet)
            time.sleep(poll_delay_s)
    finally:
        eeg_stream.stop_streaming()
        eeg_thread.join(timeout=2.0)

    while (packet := eeg_stream.pop_packet()) is not None:
        packets.append(packet)

    if stream_errors:
        error = stream_errors[0]
        raise RuntimeError(f"EEG stream verification failed: {error}") from error
    if not packets:
        raise RuntimeError(
            f"No EEG packet arrived during the {duration_s:g}-second verification. "
            "Check that the OpenBCI EEG LSL stream is running."
        )

    packet_shapes = {
        tuple(np.asarray(packet.samples).shape)
        for packet in packets
    }
    if len(packet_shapes) != 1:
        raise RuntimeError(
            f"EEG packets had inconsistent sample shapes: {sorted(packet_shapes)}."
        )
    packet_shape = next(iter(packet_shapes))
    if len(packet_shape) != 2:
        raise RuntimeError(
            f"EEG packets must be channels-by-samples arrays; got {packet_shape}."
        )

    samples = np.concatenate(
        [np.asarray(packet.samples, dtype=np.float64) for packet in packets],
        axis=1,
    )
    if not np.all(np.isfinite(samples)):
        raise RuntimeError("EEG stream verification received non-finite samples.")
    channel_stds = np.std(samples, axis=1)
    if not np.all(np.isfinite(channel_stds)):
        raise RuntimeError(
            "EEG stream verification produced non-finite channel standard deviations."
        )

    return {
        "preprocessing_signature": dict(eeg_stream.preprocessing_signature),
        "packet_count": len(packets),
        "packet_shape": list(packet_shape),
        "channel_standard_deviations": [
            float(value)
            for value in channel_stds
        ],
    }


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

    print(
        "Data collection is complete; starting profile training and evaluation.",
        flush=True,
    )
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


def bootstrap_base(args: argparse.Namespace) -> None:
    paths = eeg_encoding.profile_paths(args.user_id)
    if paths.holdout_manifest.exists():
        raise RuntimeError(
            "The corrected permanent holdout already exists for this user; "
            "bootstrap-base is only for the first corrected base checkpoint."
        )
    requested_out = Path(args.out)
    resolved_out = (
        requested_out
        if requested_out.is_absolute()
        else ROOT_DIR / "eeg_encoding" / "models" / requested_out
    )
    if resolved_out.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing base checkpoint: {resolved_out}"
        )
    temporary_out = resolved_out.with_name(
        f".{resolved_out.name}.bootstrap-{uuid4().hex}.tmp"
    )

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = paths.sessions_root / f"bootstrap_{session_id}"
    session_data = session_dir / "paired_profile_session.npz"
    bootstrap_report_path = session_dir / "bootstrap_report.json"
    bootstrap_committed = False
    try:
        collect_profile_build_session(args, session_data)
        session = _validate_bootstrap_session(session_data)

        predictor = eeg_encoding.train_model(
            session_data,
            temporary_out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            pose_checkpoint=args.pose_checkpoint,
            device=args.device,
        )
        test_metrics = eeg_encoding.evaluate_profile_post_adaptation(
            temporary_out,
            [session],
            inner_epochs=args.inner_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            pose_checkpoint=args.pose_checkpoint,
            device=args.device,
            evaluation_role="test",
        )
        eeg_encoding.seed_profile_holdout(
            user_id=args.user_id,
            session_archive=session_data,
        )

        training_report = predictor.training_report
        assert training_report is not None
        session_dir.mkdir(parents=True, exist_ok=True)
        bootstrap_report_path.write_text(
            json.dumps(
                {
                    "user_id": args.user_id,
                    "base_checkpoint": str(resolved_out),
                    "session_archive": str(session_data),
                    "raw_pose_eeg_debug_archive": (
                        str(raw_pose_eeg_debug_path(session_data))
                        if getattr(args, "save_raw_pose_eeg_debug", False)
                        else None
                    ),
                    "training_report": eeg_encoding.format_training_report(
                        training_report
                    ),
                    "test_metrics": [asdict(metric) for metric in test_metrics],
                    "holdout_manifest": str(paths.holdout_manifest),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if resolved_out.exists():
            raise FileExistsError(
                f"Refusing to overwrite an existing base checkpoint: {resolved_out}"
            )
        temporary_out.replace(resolved_out)
        bootstrap_committed = True
    finally:
        temporary_out.unlink(missing_ok=True)
        if not bootstrap_committed:
            # Both files are unique to this attempted bootstrap. Keep the captured
            # session for diagnosis/reuse, but leave no half-installed base lineage.
            paths.holdout_manifest.unlink(missing_ok=True)
            bootstrap_report_path.unlink(missing_ok=True)

    print(eeg_encoding.format_training_report(training_report))
    print(f"Saved corrected base checkpoint to {resolved_out}")
    print(f"Seeded permanent holdout at {paths.holdout_manifest}")
    print(f"Bootstrap report: {bootstrap_report_path}")


def _validate_bootstrap_session(session_data: str | Path):
    """Validate all data needed before spending time on one-off base training."""
    session_path = Path(session_data)
    session = eeg_encoding.load_profile_session(session_path)
    role_indices = {
        "support": session.support_indices,
        "query": session.query_indices,
        "validation": session.validation_indices,
        "test": session.test_indices,
    }
    empty_roles = [
        role for role, indices in role_indices.items() if len(indices) == 0
    ]
    if empty_roles:
        raise RuntimeError(
            "Bootstrap capture has no trusted, split-eligible frames for roles "
            f"{empty_roles}; rerun the capture before training a base model."
        )
    if session.preprocessing_signature is None:
        raise RuntimeError("Bootstrap capture has no EEG preprocessing signature.")
    if session.block_ids is None:
        raise RuntimeError("Bootstrap capture has no per-frame guide block IDs.")

    with np.load(session_path) as archive:
        if "profile_round_role" not in archive.files:
            raise RuntimeError("Bootstrap capture has no recorded four-round roles.")
        raw_roles = np.asarray(archive["profile_round_role"]).astype(str)
    raw_test_indices = np.flatnonzero(raw_roles == "test").astype(np.int64)
    candidates = candidates_from_test_blocks(
        session_path,
        raw_test_indices,
        preprocessing=session.preprocessing_signature,
    )
    try:
        prospective_manifest = create_manifest_from_first_session(candidates)
    except ValueError as error:
        raise RuntimeError(
            "Bootstrap capture cannot seed the complete permanent holdout: "
            f"{error}"
        ) from error
    missing_categories = [
        category
        for category in PERMANENT_HOLDOUT_CATEGORIES
        if prospective_manifest.slots[category] is None
    ]
    if missing_categories:
        raise RuntimeError(
            "Bootstrap capture cannot seed the complete permanent holdout; "
            f"missing eligible test blocks for {missing_categories}."
        )

    evaluable_test_block_ids = set(
        int(value) for value in session.block_ids[session.test_indices]
    )
    unevaluable_categories = [
        category
        for category, slot in prospective_manifest.slots.items()
        if slot is not None
        and slot.source_block_id not in evaluable_test_block_ids
    ]
    if unevaluable_categories:
        raise RuntimeError(
            "Bootstrap holdout blocks have no trusted, split-eligible test frames "
            f"for {unevaluable_categories}."
        )
    return session


def collect_profile_build_session(
    args: argparse.Namespace,
    out_path: str | Path,
) -> None:
    session_id = Path(out_path).parent.name
    sequence_seed = int(np.random.SeedSequence().generate_state(1)[0])
    sequence = profile_build_sequence(seed=sequence_seed)
    _require_full_profile_build_duration(args.duration, sequence=sequence)
    debug_capture = _raw_debug_capture(args)
    eeg_stream = SignalStreamer(
        corrected_sample_observer=(
            debug_capture.record_corrected_eeg if debug_capture is not None else None
        ),
        raw_lsl_observer=(
            debug_capture.record_raw_lsl if debug_capture is not None else None
        ),
        clock_observer=(
            debug_capture.record_eeg_clock if debug_capture is not None else None
        ),
    )
    overlay_state = CalibrationOverlayState()
    pose_estimator = AsyncPoseEstimator(
        model_path=args.pose_model,
        camera_index=args.camera_index,
        target_fps=args.pose_fps,
        mirror_frame=args.mirror_preview,
        draw_preview=True,
        draw_builtin_pose_overlay=False,
        preview_renderer=overlay_state.render,
        result_observer=(
            debug_capture.record_native_pose if debug_capture is not None else None
        ),
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
    metadata = {
        **eeg_stream.preprocessing_signature,
        **eeg_stream.timing_signature,
        "sample_kind": "profile_build_session",
        "pose_model": args.pose_model,
        "pose_checkpoint": args.pose_checkpoint,
        "pose_camera_index": args.camera_index,
        "pose_fps": args.pose_fps,
        "pose_mirror_frame": args.mirror_preview,
        "pose_include_velocity": POSE_INCLUDE_VELOCITY,
        "pose_use_world_landmarks": POSE_USE_WORLD_LANDMARKS,
        "pose_live_smoothing": POSE_LIVE_SMOOTHING,
        "pose_live_median_window": POSE_LIVE_MEDIAN_WINDOW,
        "pose_live_mean_window": POSE_LIVE_MEAN_WINDOW,
        "pairing_max_pose_gap_s": args.max_pose_gap,
        "pairing_pose_time_basis": "capture_timestamp_ms",
        "profile_sequence_seed": sequence_seed,
        "profile_rest_hold_duration_s": PROFILE_REST_HOLD_DURATION_S,
        "profile_rest_velocity_threshold": PROFILE_REST_VELOCITY_THRESHOLD,
        "profile_rest_velocity_window_s": PROFILE_REST_VELOCITY_WINDOW_S,
        "profile_rest_violation_grace_s": PROFILE_REST_VIOLATION_GRACE_S,
        "profile_rest_wrist_tolerance": PROFILE_REST_WRIST_TOLERANCE,
        "profile_movement_title_delay_s": PROFILE_MOVEMENT_TITLE_DELAY_S,
        "profile_movement_title_fade_s": PROFILE_MOVEMENT_TITLE_FADE_S,
        "profile_movement_titles_json": json.dumps(PROFILE_MOVEMENT_TITLES),
        "profile_chunk_movement_order_json": json.dumps(
            {
                role: [block.name for block in sequence if block.role == role]
                for role in PROFILE_BUILD_ROLES
            }
        ),
    }
    try:
        pose_estimator.start()
        eeg_thread.start()
        frames, block_results = collect_guided_profile_build_frames(
            eeg_stream,
            pose_stream,
            duration_s=args.duration,
            max_pose_gap_s=args.max_pose_gap,
            overlay_state=overlay_state,
            debug_capture=debug_capture,
            sequence=sequence,
        )
        save_guided_profile_build_frames(
            out_path,
            frames,
            block_results,
            session_id=session_id,
            posture=args.posture,
            metadata=metadata,
        )
    finally:
        eeg_stream.stop_streaming()
        pose_estimator.stop()
        eeg_thread.join(timeout=2.0)
    if debug_capture is not None:
        debug_path = debug_capture.save(
            raw_pose_eeg_debug_path(out_path),
            paired_frames=frames,
            metadata={**metadata, "training_archive": str(Path(out_path))},
            block_results=block_results,
        )
        print(f"Saved raw pose/EEG debug sidecar to {debug_path}")
    accepted_blocks = sum(result.accepted for result in block_results)
    print(
        f"Saved {len(frames)} movement-block profile paired frames; "
        f"{accepted_blocks}/{len(block_results)} guide blocks passed diagnostics. "
        f"Archive: {Path(out_path)}"
    )


def collect_guided_profile_build_frames(
    eeg_stream,
    pose_stream,
    *,
    duration_s: float,
    max_pose_gap_s: float,
    overlay_state: CalibrationOverlayState,
    debug_capture: RawPoseEegDebugCapture | None = None,
    sequence=None,
    rest_gate: NeutralRestGate | None = None,
) -> tuple[list, list[ProfileBlockResult]]:
    pose_buffer = PoseLatentBuffer(max_gap_s=max_pose_gap_s)
    sequence = tuple(profile_build_sequence() if sequence is None else sequence)
    if not sequence:
        return [], []
    rest_gate = NeutralRestGate() if rest_gate is None else rest_gate
    rest_block = CalibrationMovementBlock("neutral_gate", "rest", 0.0, rest=True)
    block_states = [
        {
            "block": block,
            "repeat_index": _repeat_index(sequence, index),
            "start_time_s": 0.0,
            "end_time_s": 0.0,
            "frames": [],
            "feature_vectors": [],
            "pose_confidences": [],
            "last_pose_timestamp_ms": None,
        }
        for index, block in enumerate(sequence)
    ]
    pending_packets = []
    collection_started_at = time.monotonic()
    collection_deadline = (
        collection_started_at + duration_s
        if duration_s > 0
        else float("inf")
    )
    block_index = 0
    block_started_at: float | None = None
    title_started_at: float | None = None
    movement_active = False
    capture_complete = False
    last_gate_pose_timestamp_ms = None
    overlay_state.update_dummy(rest_block, 0.0)
    overlay_state.update_neutral_rest(rest_gate.status)
    overlay_state.update_movement_title(None)

    while block_index < len(sequence) and time.monotonic() < collection_deadline:
        now = time.monotonic()
        if movement_active:
            assert block_started_at is not None
            block = sequence[block_index]
            if now - block_started_at >= block.duration_s:
                block_states[block_index]["end_time_s"] = now
                movement_active = False
                block_started_at = None
                overlay_state.update_dummy(rest_block, 0.0)
                overlay_state.update_movement_title(None)
                block_index += 1
                if block_index >= len(sequence):
                    capture_complete = True
                    break
                rest_gate.begin_return_to_rest()
                overlay_state.update_neutral_rest(rest_gate.status)
                last_gate_pose_timestamp_ms = None
        elif title_started_at is not None:
            if now - title_started_at >= PROFILE_MOVEMENT_TITLE_DELAY_S:
                block = sequence[block_index]
                title_started_at = None
                block_started_at = now
                block_states[block_index]["start_time_s"] = now
                movement_active = True
                overlay_state.update_dummy(block, 0.0)
                overlay_state.update_movement_title(block)

        if movement_active:
            assert block_started_at is not None
            title_alpha = 1.0 - (
                (now - block_started_at) / PROFILE_MOVEMENT_TITLE_FADE_S
            )
            overlay_state.update_movement_title(
                sequence[block_index] if title_alpha > 0.0 else None,
                alpha=title_alpha,
            )

        pose_frame = pose_stream.get_latest()
        pose_buffer.add(pose_frame)
        if debug_capture is not None:
            debug_capture.record_processed_pose(pose_frame)
        if pose_frame is not None and pose_frame.pose_detected:
            overlay_state.update_truth(pose_frame.feature_vector)
            gate_pose_is_usable = (
                pose_frame.confidence >= EEG_CALIBRATION_MIN_POSE_CONFIDENCE
                and pose_frame.timestamp_ms != last_gate_pose_timestamp_ms
            )
            if (
                not movement_active
                and title_started_at is None
                and gate_pose_is_usable
            ):
                last_gate_pose_timestamp_ms = pose_frame.timestamp_ms
                gate_ready = rest_gate.update(pose_frame.feature_vector, now)
                overlay_state.update_neutral_rest(rest_gate.status)
                if gate_ready:
                    block = sequence[block_index]
                    title_started_at = now
                    overlay_state.update_neutral_rest(None)
                    overlay_state.update_dummy(block, 0.0)
                    overlay_state.update_movement_title(block)
            if movement_active:
                assert block_started_at is not None
                block = sequence[block_index]
                overlay_state.update_dummy(block, now - block_started_at)
                state = block_states[block_index]
                if state["last_pose_timestamp_ms"] != pose_frame.timestamp_ms:
                    state["feature_vectors"].append(pose_frame.feature_vector.copy())
                    state["pose_confidences"].append(float(pose_frame.confidence))
                    state["last_pose_timestamp_ms"] = pose_frame.timestamp_ms
        elif movement_active:
            assert block_started_at is not None
            overlay_state.update_dummy(
                sequence[block_index],
                now - block_started_at,
            )

        while True:
            packet = eeg_stream.pop_packet()
            if packet is None:
                break
            if debug_capture is not None:
                debug_capture.record_eeg_packet(packet)
            packet_block_index = _profile_packet_block_index(packet, block_states)
            if packet_block_index is not None:
                pending_packets.append((packet, packet_block_index))

        pending_packets = _pair_pending_profile_packets(
            pending_packets,
            pose_buffer,
            block_states,
        )
        time.sleep(PAIRING_POLL_DELAY_S)

    finished_at = time.monotonic()
    if movement_active:
        block_states[block_index]["end_time_s"] = finished_at
    if block_index >= len(sequence):
        capture_complete = True

    # A packet fully contained by the final movement may not be emitted until
    # its overlapping EEG window is complete. Continue pairing those packets,
    # while rejecting every packet that overlaps a neutral gate.
    packet_span_s = (EEG_PACKET_SIZE - 1) / EEG_SAMPLE_RATE
    packet_step_s = EEG_PACKET_STRIDE / EEG_SAMPLE_RATE
    drain_deadline = (
        time.monotonic()
        + packet_span_s
        + packet_step_s
        + EEG_ACQUISITION_TIMEOUT_S
        + max_pose_gap_s
    )
    while time.monotonic() < drain_deadline:
        pose_frame = pose_stream.get_latest()
        pose_buffer.add(pose_frame)
        if debug_capture is not None:
            debug_capture.record_processed_pose(pose_frame)
        while True:
            packet = eeg_stream.pop_packet()
            if packet is None:
                break
            if debug_capture is not None:
                debug_capture.record_eeg_packet(packet)
            packet_block_index = _profile_packet_block_index(packet, block_states)
            if packet_block_index is not None:
                pending_packets.append((packet, packet_block_index))
        pending_packets = _pair_pending_profile_packets(
            pending_packets,
            pose_buffer,
            block_states,
        )
        time.sleep(PAIRING_POLL_DELAY_S)

    if not capture_complete:
        completed_blocks = sum(
            state["end_time_s"] > state["start_time_s"] > 0.0
            for state in block_states
        )
        raise RuntimeError(
            "Profile capture ended before all randomized movement blocks were "
            f"completed: {completed_blocks}/{len(sequence)} finished. Increase "
            "--duration or use 0 for no deadline."
        )

    block_results = [
        validate_profile_block(
            block_id=index,
            block=state["block"],
            repeat_index=state["repeat_index"],
            start_time_s=state["start_time_s"],
            end_time_s=state["end_time_s"],
            feature_vectors=state["feature_vectors"],
            pose_confidences=state["pose_confidences"],
            paired_sample_count=len(state["frames"]),
        )
        for index, state in enumerate(block_states)
    ]
    frames = [
        frame
        for state in block_states
        for frame in state["frames"]
    ]
    return frames, block_results


def _raw_debug_capture(
    args: argparse.Namespace,
) -> RawPoseEegDebugCapture | None:
    return (
        RawPoseEegDebugCapture()
        if getattr(args, "save_raw_pose_eeg_debug", False)
        else None
    )


def _profile_packet_block_index(packet, block_states) -> int | None:
    for index, state in enumerate(block_states):
        start_time_s = state["start_time_s"]
        end_time_s = state["end_time_s"]
        if start_time_s <= 0.0:
            break
        if packet.start_time_s < start_time_s:
            continue
        if end_time_s == 0.0 or packet.end_time_s <= end_time_s:
            return index
    return None


def _pair_pending_profile_packets(
    pending_packets,
    pose_buffer: PoseLatentBuffer,
    block_states,
):
    still_pending = []
    latest_pose_time_s = pose_buffer.latest_time_s
    for packet, block_index in pending_packets:
        paired = pair_packet(packet, pose_buffer)
        if paired is not None:
            block_states[block_index]["frames"].append(paired)
        elif latest_pose_time_s is None or latest_pose_time_s <= packet.end_time_s:
            still_pending.append((packet, block_index))
    return still_pending


def _repeat_index(sequence, index: int) -> int:
    role = sequence[index].role
    assert role is not None
    return PROFILE_BUILD_ROLES.index(role)


def _require_full_profile_build_duration(duration_s: float, *, sequence=None) -> None:
    sequence = profile_build_sequence() if sequence is None else sequence
    movement_duration_s = sum(
        block.duration_s
        for block in sequence
    )
    minimum_gate_duration_s = len(sequence) * PROFILE_REST_HOLD_DURATION_S
    minimum_title_duration_s = len(sequence) * PROFILE_MOVEMENT_TITLE_DELAY_S
    minimum_duration_s = (
        movement_duration_s
        + minimum_gate_duration_s
        + minimum_title_duration_s
    )
    if 0 < duration_s < minimum_duration_s:
        raise ValueError(
            "Profile capture duration is shorter than the minimum movement-and-rest "
            "time for the "
            f"full four-chunk guide: got {duration_s:g} seconds, need at least "
            f"{minimum_duration_s:g} seconds. Actual neutral-return waits may take "
            "longer (or use 0 for no deadline)."
        )


def save_guided_profile_build_frames(
    out_path: str | Path,
    frames: list,
    block_results: list[ProfileBlockResult],
    *,
    session_id: str,
    posture: str,
    metadata: dict[str, object],
) -> None:
    if not frames:
        raise RuntimeError("No paired guided profile-build frames were collected.")
    per_frame_block_ids = []
    per_frame_block_names = []
    per_frame_repeat_indices = []
    per_frame_accepted = []
    per_frame_roles = []
    per_frame_is_rest = []
    frame_index = 0
    block_result_by_id = {result.block_id: result for result in block_results}
    for result in block_results:
        assert result.role is not None
        frame_count = result.paired_sample_count
        per_frame_block_ids.extend([result.block_id] * frame_count)
        per_frame_block_names.extend([result.movement_name] * frame_count)
        per_frame_repeat_indices.extend([result.repeat_index] * frame_count)
        per_frame_accepted.extend([result.accepted] * frame_count)
        per_frame_roles.extend([result.role] * frame_count)
        per_frame_is_rest.extend([result.is_rest] * frame_count)
        frame_index += frame_count
    assert frame_index == len(frames)
    frame_trusted = np.asarray(
        [
            frame.pose_confidence >= EEG_CALIBRATION_MIN_POSE_CONFIDENCE
            and frame.interpolation_confidence
            >= EEG_CALIBRATION_MIN_INTERPOLATION_CONFIDENCE
            and frame.pose_reconstruction_error
            <= EEG_CALIBRATION_MAX_POSE_RECONSTRUCTION_ERROR
            for frame in frames
        ],
        dtype=np.bool_,
    )
    split_eligible = np.ones(len(frames), dtype=np.bool_)
    role_array = np.asarray(per_frame_roles)
    packet_span_s = (EEG_PACKET_SIZE - 1) / EEG_SAMPLE_RATE
    for index in range(1, len(frames)):
        if role_array[index] == role_array[index - 1]:
            continue
        previous_role_end_s = frames[index - 1].target_time_s
        guard_index = index
        while (
            guard_index < len(frames)
            and role_array[guard_index] == role_array[index]
            and frames[guard_index].target_time_s - packet_span_s
            <= previous_role_end_s
        ):
            split_eligible[guard_index] = False
            guard_index += 1
    save_paired_frames(
        out_path,
        frames,
        metadata={
            **metadata,
            "profile_session_id": session_id,
            "profile_posture": posture,
            "profile_block_id": np.asarray(per_frame_block_ids, dtype=np.int64),
            "profile_block_name": np.asarray(per_frame_block_names),
            "profile_block_repeat_index": np.asarray(per_frame_repeat_indices, dtype=np.int64),
            "profile_block_accepted": np.asarray(per_frame_accepted, dtype=np.bool_),
            "profile_round_role": np.asarray(per_frame_roles),
            "profile_block_is_rest": np.asarray(per_frame_is_rest, dtype=np.bool_),
            "profile_frame_trusted": frame_trusted,
            "profile_frame_split_eligible": split_eligible,
            "profile_block_summary_json": json.dumps(
                [
                    {
                        **asdict(result),
                        "accepted": bool(result.accepted),
                    }
                    for result in block_results
                ]
            ),
            "profile_accepted_block_count": sum(
                result.accepted for result in block_result_by_id.values()
            ),
        },
    )


def predict_live(args: argparse.Namespace) -> None:
    eeg_stream = SignalStreamer()
    predictor = eeg_encoding.load_model(args.model, device=args.device)
    _require_live_preprocessing(eeg_stream, predictor, checkpoint=args.model)
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
    _require_live_preprocessing(
        eeg_stream,
        eeg_predictor,
        checkpoint=args.eeg_model,
    )
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
    _require_live_preprocessing(
        eeg_stream,
        eeg_predictor,
        checkpoint=start_checkpoint,
    )
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


def _require_live_preprocessing(
    eeg_stream: SignalStreamer,
    predictor,
    *,
    checkpoint: str | Path,
) -> None:
    eeg_encoding.require_matching_preprocessing(
        eeg_encoding.preprocessing_signature_from_config(predictor.model.config),
        eeg_stream.preprocessing_signature,
        source=str(checkpoint),
    )


def main() -> None:
    args = parse_args()
    if args.command == "collect-paired":
        collect_paired(args)
    elif args.command == "verify-eeg-stream":
        verify_eeg_stream(args)
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
    elif args.command == "bootstrap-base":
        bootstrap_base(args)
    elif args.command == "recalibrate":
        recalibrate(args)


if __name__ == "__main__":
    main()
