"""Opt-in native EEG and pose capture for representation experiments."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from threading import Lock
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from pose_encoding import PoseLatentFrame
from streaming.eeg import EegPacket
from streaming.pose import PoseLandmark, PoseResult
from streaming.pose.pose_features import UPPER_BODY_LANDMARKS

from .records import PairedTrainingFrame


RAW_POSE_EEG_DEBUG_SCHEMA = "raw-pose-eeg-debug-v1"
MEDIAPIPE_POSE_LANDMARK_COUNT = 33
POSE_LANDMARK_FIELDS = ("x", "y", "z", "visibility", "presence")
POSE_FEATURE_AXES = ("x", "y", "z")


class RawPoseEegDebugCapture:
    """Accumulate native streams without changing the training archive schema."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._eeg_chunks: list[
            tuple[
                NDArray[np.float32],
                NDArray[np.float64],
                NDArray[np.float64],
                float,
            ]
        ] = []
        self._corrected_eeg_chunks: list = []
        self._raw_lsl_chunks: list = []
        self._clock_measurements: list[dict[str, float]] = []
        self._eeg_packets: list[EegPacket] = []
        self._native_pose_frames: list[
            tuple[PoseResult, NDArray[np.float32], NDArray[np.float32]]
        ] = []
        self._processed_pose_frames: list[PoseLatentFrame] = []
        self._processed_pose_timestamps: set[int] = set()

    def record_native_eeg(
        self,
        samples: NDArray[np.float32],
        monotonic_times_s: NDArray[np.float64],
        source_times_s: NDArray[np.float64],
        received_time_s: float,
    ) -> None:
        with self._lock:
            self._eeg_chunks.append(
                (
                    samples.astype(np.float32, copy=True),
                    monotonic_times_s.astype(np.float64, copy=True),
                    source_times_s.astype(np.float64, copy=True),
                    float(received_time_s),
                )
            )

    def record_corrected_eeg(self, samples, monotonic_times_s, corrected_lsl_times_s, received_time_s) -> None:
        with self._lock:
            self._corrected_eeg_chunks.append((
                samples.copy(), monotonic_times_s.copy(), corrected_lsl_times_s.copy(),
                float(received_time_s),
            ))

    def record_raw_lsl(self, samples, source_times_s, received_time_s) -> None:
        with self._lock:
            self._raw_lsl_chunks.append((
                samples.copy(), np.full(len(samples), np.nan), source_times_s.copy(),
                float(received_time_s),
            ))

    def record_eeg_clock(self, measurement: dict[str, float]) -> None:
        with self._lock:
            self._clock_measurements.append(dict(measurement))

    def record_eeg_packet(self, packet: EegPacket) -> None:
        with self._lock:
            self._eeg_packets.append(
                EegPacket(
                    packet_id=packet.packet_id,
                    samples=packet.samples.astype(np.float32, copy=True),
                    start_time_s=packet.start_time_s,
                    end_time_s=packet.end_time_s,
                    sample_rate=packet.sample_rate,
                )
            )

    def record_native_pose(self, result: PoseResult) -> None:
        image_landmarks = _landmark_array(result.landmarks)
        world_landmarks = _landmark_array(result.world_landmarks)
        with self._lock:
            self._native_pose_frames.append(
                (result, image_landmarks, world_landmarks)
            )

    def record_processed_pose(self, frame: PoseLatentFrame | None) -> None:
        if frame is None:
            return
        with self._lock:
            if frame.timestamp_ms in self._processed_pose_timestamps:
                return
            self._processed_pose_timestamps.add(frame.timestamp_ms)
            self._processed_pose_frames.append(
                PoseLatentFrame(
                    timestamp_ms=frame.timestamp_ms,
                    received_time_s=frame.received_time_s,
                    feature_vector=frame.feature_vector.astype(
                        np.float32,
                        copy=True,
                    ),
                    latent=frame.latent.astype(np.float32, copy=True),
                    reconstruction=frame.reconstruction.astype(
                        np.float32,
                        copy=True,
                    ),
                    reconstruction_error=frame.reconstruction_error,
                    pose_detected=frame.pose_detected,
                    confidence=frame.confidence,
                )
            )

    def save(
        self,
        path: str | Path,
        *,
        paired_frames: list[PairedTrainingFrame],
        metadata: dict[str, object],
        block_results: Iterable[object] = (),
    ) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            eeg_chunks = list(self._eeg_chunks)
            corrected_chunks = list(self._corrected_eeg_chunks)
            raw_lsl_chunks = list(self._raw_lsl_chunks)
            clock_measurements = list(self._clock_measurements)
            eeg_packets = list(self._eeg_packets)
            native_pose_frames = list(self._native_pose_frames)
            processed_pose_frames = list(self._processed_pose_frames)

        native_pose_frames.sort(key=lambda frame: frame[0].timestamp_ms)
        processed_pose_frames.sort(key=lambda frame: frame.timestamp_ms)
        eeg_packets.sort(key=lambda packet: packet.packet_id)
        blocks = list(block_results)
        native_pose_index_by_timestamp = {
            frame[0].timestamp_ms: index
            for index, frame in enumerate(native_pose_frames)
        }
        processed_native_indices = np.asarray(
            [
                native_pose_index_by_timestamp.get(frame.timestamp_ms, -1)
                for frame in processed_pose_frames
            ],
            dtype=np.int64,
        )
        native_arrays = _native_eeg_arrays(corrected_chunks or eeg_chunks)
        if corrected_chunks:
            native_arrays["eeg_native_corrected_lsl_time_s"] = native_arrays["eeg_native_source_time_s"]
            # Original source timestamps belong to the independent raw inlet below.
            native_arrays["eeg_native_source_time_s"] = np.full(
                len(native_arrays["eeg_native_samples"]), np.nan,
            )
        raw_arrays = {
            key.replace("eeg_native_", "eeg_raw_lsl_"): value
            for key, value in _native_eeg_arrays(raw_lsl_chunks).items()
            if key != "eeg_native_monotonic_time_s"
        }
        arrays = {
            "schema_version": np.asarray(RAW_POSE_EEG_DEBUG_SCHEMA),
            "metadata_json": np.asarray(json.dumps({**metadata, "pairing_pose_time_basis": "capture_timestamp_ms"}, default=str)),
            "pose_landmark_fields": np.asarray(POSE_LANDMARK_FIELDS),
            "pose_feature_landmark_indices": np.asarray(
                UPPER_BODY_LANDMARKS,
                dtype=np.int64,
            ),
            "pose_feature_axes": np.asarray(POSE_FEATURE_AXES),
            **native_arrays,
            **raw_arrays,
            "eeg_clock_measurements_json": np.asarray(json.dumps(clock_measurements)),
            **_eeg_packet_arrays(eeg_packets),
            **_native_pose_arrays(native_pose_frames),
            **_processed_pose_arrays(
                processed_pose_frames,
                native_indices=processed_native_indices,
            ),
            **_paired_arrays(
                paired_frames,
                eeg_packets=eeg_packets,
                processed_pose_frames=processed_pose_frames,
                processed_native_indices=processed_native_indices,
            ),
            **_block_arrays(blocks),
        }
        np.savez_compressed(output_path, **arrays)
        return output_path


def raw_pose_eeg_debug_path(training_archive_path: str | Path) -> Path:
    path = Path(training_archive_path)
    return path.with_suffix(".raw_pose_eeg_debug.npz")


def _landmark_array(landmarks: list[PoseLandmark]) -> NDArray[np.float32]:
    output = np.full(
        (MEDIAPIPE_POSE_LANDMARK_COUNT, 5),
        np.nan,
        dtype=np.float32,
    )
    if not landmarks:
        return output
    if len(landmarks) != MEDIAPIPE_POSE_LANDMARK_COUNT:
        raise ValueError(
            "MediaPipe pose result must contain 33 landmarks; got "
            f"{len(landmarks)}."
        )
    output[:] = [
        [
            _optional_float(landmark.x),
            _optional_float(landmark.y),
            _optional_float(landmark.z),
            _optional_float(landmark.visibility),
            _optional_float(landmark.presence),
        ]
        for landmark in landmarks
    ]
    return output


def _optional_float(value: float | None) -> float:
    return float(value) if value is not None else float("nan")


def _native_eeg_arrays(chunks) -> dict[str, NDArray]:
    if not chunks:
        return {
            "eeg_native_samples": np.empty((0, 0), dtype=np.float32),
            "eeg_native_monotonic_time_s": np.empty(0, dtype=np.float64),
            "eeg_native_source_time_s": np.empty(0, dtype=np.float64),
            "eeg_native_chunk_start_index": np.empty(0, dtype=np.int64),
            "eeg_native_chunk_length": np.empty(0, dtype=np.int64),
            "eeg_native_chunk_received_time_s": np.empty(0, dtype=np.float64),
        }
    column_count = chunks[0][0].shape[1]
    assert all(chunk[0].shape[1] == column_count for chunk in chunks)
    lengths = np.asarray([len(chunk[0]) for chunk in chunks], dtype=np.int64)
    starts = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1]))
    )
    return {
        "eeg_native_samples": np.concatenate(
            [chunk[0] for chunk in chunks],
            axis=0,
        ),
        "eeg_native_monotonic_time_s": np.concatenate(
            [chunk[1] for chunk in chunks]
        ),
        "eeg_native_source_time_s": np.concatenate(
            [chunk[2] for chunk in chunks]
        ),
        "eeg_native_chunk_start_index": starts,
        "eeg_native_chunk_length": lengths,
        "eeg_native_chunk_received_time_s": np.asarray(
            [chunk[3] for chunk in chunks],
            dtype=np.float64,
        ),
    }


def _eeg_packet_arrays(packets: list[EegPacket]) -> dict[str, NDArray]:
    if not packets:
        return {
            "eeg_packet_id": np.empty(0, dtype=np.int64),
            "eeg_packet_samples": np.empty((0, 0, 0), dtype=np.float32),
            "eeg_packet_start_time_s": np.empty(0, dtype=np.float64),
            "eeg_packet_end_time_s": np.empty(0, dtype=np.float64),
            "eeg_packet_sample_rate_hz": np.empty(0, dtype=np.int64),
        }
    return {
        "eeg_packet_id": np.asarray(
            [packet.packet_id for packet in packets],
            dtype=np.int64,
        ),
        "eeg_packet_samples": np.stack(
            [packet.samples for packet in packets]
        ).astype(np.float32, copy=False),
        "eeg_packet_start_time_s": np.asarray(
            [packet.start_time_s for packet in packets],
            dtype=np.float64,
        ),
        "eeg_packet_end_time_s": np.asarray(
            [packet.end_time_s for packet in packets],
            dtype=np.float64,
        ),
        "eeg_packet_sample_rate_hz": np.asarray(
            [packet.sample_rate for packet in packets],
            dtype=np.int64,
        ),
    }


def _native_pose_arrays(frames) -> dict[str, NDArray]:
    if not frames:
        empty_landmarks = np.empty(
            (0, MEDIAPIPE_POSE_LANDMARK_COUNT, 5),
            dtype=np.float32,
        )
        return {
            "pose_native_timestamp_ms": np.empty(0, dtype=np.int64),
            "pose_native_received_time_s": np.empty(0, dtype=np.float64),
            "pose_native_image_width": np.empty(0, dtype=np.int32),
            "pose_native_image_height": np.empty(0, dtype=np.int32),
            "pose_native_detected": np.empty(0, dtype=np.bool_),
            "pose_native_image_landmarks": empty_landmarks,
            "pose_native_world_landmarks": empty_landmarks.copy(),
        }
    return {
        "pose_native_timestamp_ms": np.asarray(
            [frame[0].timestamp_ms for frame in frames],
            dtype=np.int64,
        ),
        "pose_native_received_time_s": np.asarray(
            [frame[0].received_time_s for frame in frames],
            dtype=np.float64,
        ),
        "pose_native_image_width": np.asarray(
            [frame[0].image_width for frame in frames],
            dtype=np.int32,
        ),
        "pose_native_image_height": np.asarray(
            [frame[0].image_height for frame in frames],
            dtype=np.int32,
        ),
        "pose_native_detected": np.asarray(
            [frame[0].pose_detected for frame in frames],
            dtype=np.bool_,
        ),
        "pose_native_image_landmarks": np.stack([frame[1] for frame in frames]),
        "pose_native_world_landmarks": np.stack([frame[2] for frame in frames]),
    }


def _processed_pose_arrays(
    frames: list[PoseLatentFrame],
    *,
    native_indices: NDArray[np.int64],
) -> dict[str, NDArray]:
    assert native_indices.shape == (len(frames),)
    if not frames:
        return {
            "pose_processed_timestamp_ms": np.empty(0, dtype=np.int64),
            "pose_processed_received_time_s": np.empty(0, dtype=np.float64),
            "pose_processed_native_index": np.empty(0, dtype=np.int64),
            "pose_processed_feature_vector": np.empty((0, 0), dtype=np.float32),
            "pose_processed_latent": np.empty((0, 0), dtype=np.float32),
            "pose_processed_reconstruction": np.empty((0, 0), dtype=np.float32),
            "pose_processed_reconstruction_error": np.empty(0, dtype=np.float32),
            "pose_processed_detected": np.empty(0, dtype=np.bool_),
            "pose_processed_confidence": np.empty(0, dtype=np.float32),
        }
    return {
        "pose_processed_timestamp_ms": np.asarray(
            [frame.timestamp_ms for frame in frames],
            dtype=np.int64,
        ),
        "pose_processed_received_time_s": np.asarray(
            [frame.received_time_s for frame in frames],
            dtype=np.float64,
        ),
        "pose_processed_native_index": native_indices,
        "pose_processed_feature_vector": np.stack(
            [frame.feature_vector for frame in frames]
        ),
        "pose_processed_latent": np.stack([frame.latent for frame in frames]),
        "pose_processed_reconstruction": np.stack(
            [frame.reconstruction for frame in frames]
        ),
        "pose_processed_reconstruction_error": np.asarray(
            [frame.reconstruction_error for frame in frames],
            dtype=np.float32,
        ),
        "pose_processed_detected": np.asarray(
            [frame.pose_detected for frame in frames],
            dtype=np.bool_,
        ),
        "pose_processed_confidence": np.asarray(
            [frame.confidence for frame in frames],
            dtype=np.float32,
        ),
    }


def _paired_arrays(
    frames: list[PairedTrainingFrame],
    *,
    eeg_packets: list[EegPacket],
    processed_pose_frames: list[PoseLatentFrame],
    processed_native_indices: NDArray[np.int64],
) -> dict[str, NDArray]:
    if not frames:
        return {
            "paired_packet_id": np.empty(0, dtype=np.int64),
            "paired_eeg_packet_index": np.empty(0, dtype=np.int64),
            "paired_target_time_s": np.empty(0, dtype=np.float64),
            "paired_pose_before_processed_index": np.empty(0, dtype=np.int64),
            "paired_pose_after_processed_index": np.empty(0, dtype=np.int64),
            "paired_pose_before_native_index": np.empty(0, dtype=np.int64),
            "paired_pose_after_native_index": np.empty(0, dtype=np.int64),
            "paired_interpolation_alpha": np.empty(0, dtype=np.float64),
            "paired_pose_feature_vector": np.empty((0, 0), dtype=np.float32),
            "paired_pose_latent": np.empty((0, 0), dtype=np.float32),
            "paired_pose_reconstruction": np.empty((0, 0), dtype=np.float32),
            "paired_pose_confidence": np.empty(0, dtype=np.float32),
            "paired_pose_reconstruction_error": np.empty(0, dtype=np.float32),
            "paired_interpolation_confidence": np.empty(0, dtype=np.float32),
        }

    packet_index_by_id = {
        packet.packet_id: index for index, packet in enumerate(eeg_packets)
    }
    assert all(frame.packet_id in packet_index_by_id for frame in frames)
    assert processed_native_indices.shape == (len(processed_pose_frames),)
    pose_times = np.asarray(
        [frame.timestamp_ms / 1000.0 for frame in processed_pose_frames],
        dtype=np.float64,
    )
    targets = np.asarray(
        [frame.target_time_s for frame in frames],
        dtype=np.float64,
    )
    before_indices = np.searchsorted(pose_times, targets, side="right") - 1
    after_indices = before_indices + 1
    assert np.all(before_indices >= 0)
    assert np.all(after_indices < len(processed_pose_frames))
    before_times = pose_times[before_indices]
    after_times = pose_times[after_indices]
    gaps = after_times - before_times
    assert np.all(gaps > 0)
    alpha = (targets - before_times) / gaps

    pose_features = np.stack(
        [frame.feature_vector for frame in processed_pose_frames]
    )
    pose_latents = np.stack([frame.latent for frame in processed_pose_frames])
    pose_reconstructions = np.stack(
        [frame.reconstruction for frame in processed_pose_frames]
    )
    interpolated_features = _interpolate_rows(
        pose_features,
        before_indices,
        after_indices,
        alpha,
    )
    interpolated_latents = _interpolate_rows(
        pose_latents,
        before_indices,
        after_indices,
        alpha,
    )
    interpolated_reconstructions = _interpolate_rows(
        pose_reconstructions,
        before_indices,
        after_indices,
        alpha,
    )
    paired_latents = np.stack([frame.pose_latent for frame in frames])
    assert np.allclose(interpolated_latents, paired_latents, atol=1e-5)

    return {
        "paired_packet_id": np.asarray(
            [frame.packet_id for frame in frames],
            dtype=np.int64,
        ),
        "paired_eeg_packet_index": np.asarray(
            [packet_index_by_id[frame.packet_id] for frame in frames],
            dtype=np.int64,
        ),
        "paired_target_time_s": targets,
        "paired_pose_before_processed_index": before_indices.astype(np.int64),
        "paired_pose_after_processed_index": after_indices.astype(np.int64),
        "paired_pose_before_native_index": processed_native_indices[before_indices],
        "paired_pose_after_native_index": processed_native_indices[after_indices],
        "paired_interpolation_alpha": alpha,
        "paired_pose_feature_vector": interpolated_features,
        "paired_pose_latent": paired_latents,
        "paired_pose_reconstruction": interpolated_reconstructions,
        "paired_pose_confidence": np.asarray(
            [frame.pose_confidence for frame in frames],
            dtype=np.float32,
        ),
        "paired_pose_reconstruction_error": np.asarray(
            [frame.pose_reconstruction_error for frame in frames],
            dtype=np.float32,
        ),
        "paired_interpolation_confidence": np.asarray(
            [frame.interpolation_confidence for frame in frames],
            dtype=np.float32,
        ),
    }


def _interpolate_rows(
    values: NDArray[np.float32],
    before_indices: NDArray[np.int64],
    after_indices: NDArray[np.int64],
    alpha: NDArray[np.float64],
) -> NDArray[np.float32]:
    weights = alpha.astype(np.float32)[:, None]
    return (
        (1.0 - weights) * values[before_indices]
        + weights * values[after_indices]
    ).astype(np.float32, copy=False)


def _block_arrays(blocks: list[object]) -> dict[str, NDArray]:
    if not blocks:
        return {
            "guide_block_id": np.empty(0, dtype=np.int64),
            "guide_block_name": np.empty(0, dtype=np.str_),
            "guide_block_repeat_index": np.empty(0, dtype=np.int64),
            "guide_block_role": np.empty(0, dtype=np.str_),
            "guide_block_is_rest": np.empty(0, dtype=np.bool_),
            "guide_block_start_time_s": np.empty(0, dtype=np.float64),
            "guide_block_end_time_s": np.empty(0, dtype=np.float64),
            "guide_block_accepted": np.empty(0, dtype=np.bool_),
            "guide_block_summary_json": np.asarray("[]"),
        }
    return {
        "guide_block_id": np.asarray(
            [block.block_id for block in blocks],
            dtype=np.int64,
        ),
        "guide_block_name": np.asarray(
            [block.movement_name for block in blocks]
        ),
        "guide_block_repeat_index": np.asarray(
            [block.repeat_index for block in blocks],
            dtype=np.int64,
        ),
        "guide_block_role": np.asarray(
            [block.role or "" for block in blocks]
        ),
        "guide_block_is_rest": np.asarray(
            [block.is_rest for block in blocks],
            dtype=np.bool_,
        ),
        "guide_block_start_time_s": np.asarray(
            [block.start_time_s for block in blocks],
            dtype=np.float64,
        ),
        "guide_block_end_time_s": np.asarray(
            [block.end_time_s for block in blocks],
            dtype=np.float64,
        ),
        "guide_block_accepted": np.asarray(
            [block.accepted for block in blocks],
            dtype=np.bool_,
        ),
        "guide_block_summary_json": np.asarray(
            json.dumps([asdict(block) for block in blocks])
        ),
    }
