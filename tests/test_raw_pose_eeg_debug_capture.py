from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from pose_encoding import PoseLatentFrame
from streaming import collect_paired_frames
from streaming.debug_capture import (
    RAW_POSE_EEG_DEBUG_SCHEMA,
    RawPoseEegDebugCapture,
)
from streaming.eeg import EegPacket, SignalStreamer
from streaming.pose import PoseLandmark, PoseResult
from streaming.records import PairedTrainingFrame


def _output_path(name: str) -> Path:
    return Path("test_outputs") / f"{name}_{uuid4().hex}.npz"


def _landmarks(offset: float) -> list[PoseLandmark]:
    return [
        PoseLandmark(
            x=offset + index,
            y=offset + index + 0.1,
            z=offset + index + 0.2,
            visibility=0.8,
            presence=0.9,
        )
        for index in range(33)
    ]


def _pose_result(timestamp_ms: int, received_time_s: float) -> PoseResult:
    return PoseResult(
        timestamp_ms=timestamp_ms,
        received_time_s=received_time_s,
        image_width=640,
        image_height=480,
        landmarks=_landmarks(0.0),
        world_landmarks=_landmarks(100.0),
        pose_detected=True,
    )


def _processed_pose(
    timestamp_ms: int,
    received_time_s: float,
    value: float,
) -> PoseLatentFrame:
    return PoseLatentFrame(
        timestamp_ms=timestamp_ms,
        received_time_s=received_time_s,
        feature_vector=np.full(48, value, dtype=np.float32),
        latent=np.full(24, value, dtype=np.float32),
        reconstruction=np.full(48, 2.0 * value, dtype=np.float32),
        reconstruction_error=0.1 + 0.1 * value,
        pose_detected=True,
        confidence=0.9 - 0.1 * value,
    )


def test_debug_sidecar_preserves_native_streams_and_exact_pairing() -> None:
    capture = RawPoseEegDebugCapture()
    native_eeg = np.arange(24, dtype=np.float32).reshape(4, 6)
    native_times = np.asarray([0.8, 0.9, 1.0, 1.1], dtype=np.float64)
    source_times = native_times + 10.0
    capture.record_native_eeg(
        native_eeg,
        native_times,
        source_times,
        received_time_s=1.1,
    )
    native_eeg[:] = -1.0

    packet = EegPacket(
        packet_id=7,
        samples=np.arange(800, dtype=np.float32).reshape(4, 200),
        start_time_s=1.0,
        end_time_s=2.0,
        sample_rate=250,
    )
    capture.record_eeg_packet(packet)
    capture.record_native_pose(_pose_result(1000, 1.3))
    capture.record_native_pose(_pose_result(3000, 3.7))
    capture.record_processed_pose(_processed_pose(1000, 1.3, 0.0))
    capture.record_processed_pose(_processed_pose(3000, 3.7, 2.0))
    paired = PairedTrainingFrame(
        packet_id=7,
        eeg=packet.samples,
        target_time_s=2.0,
        pose_latent=np.ones(24, dtype=np.float32),
        pose_confidence=0.8,
        pose_reconstruction_error=0.2,
        interpolation_confidence=0.7,
    )

    path = capture.save(
        _output_path("session_raw_pose_eeg_debug"),
        paired_frames=[paired],
        metadata={"training_archive": "session.npz"},
    )

    with np.load(path, allow_pickle=False) as archive:
        assert archive["schema_version"].item() == RAW_POSE_EEG_DEBUG_SCHEMA
        assert json.loads(archive["metadata_json"].item()) == {
            "training_archive": "session.npz",
            "pairing_pose_time_basis": "capture_timestamp_ms",
        }
        assert archive["eeg_native_samples"].shape == (4, 6)
        np.testing.assert_array_equal(
            archive["eeg_native_samples"],
            np.arange(24, dtype=np.float32).reshape(4, 6),
        )
        np.testing.assert_array_equal(
            archive["eeg_native_monotonic_time_s"],
            native_times,
        )
        assert archive["eeg_packet_samples"].shape == (1, 4, 200)
        assert archive["pose_native_image_landmarks"].shape == (2, 33, 5)
        assert archive["pose_native_world_landmarks"].shape == (2, 33, 5)
        assert archive["pose_landmark_fields"].tolist() == [
            "x",
            "y",
            "z",
            "visibility",
            "presence",
        ]
        assert archive["pose_feature_landmark_indices"].tolist() == [
            11,
            12,
            13,
            14,
            15,
            16,
            23,
            24,
        ]
        assert archive["paired_eeg_packet_index"].tolist() == [0]
        assert archive["paired_pose_before_processed_index"].tolist() == [0]
        assert archive["paired_pose_after_processed_index"].tolist() == [1]
        assert archive["pose_processed_native_index"].tolist() == [0, 1]
        assert archive["paired_pose_before_native_index"].tolist() == [0]
        assert archive["paired_pose_after_native_index"].tolist() == [1]
        np.testing.assert_allclose(
            archive["paired_interpolation_alpha"],
            [0.5],
        )
        np.testing.assert_allclose(
            archive["paired_pose_feature_vector"],
            np.ones((1, 48), dtype=np.float32),
        )
        np.testing.assert_allclose(
            archive["paired_pose_latent"],
            np.ones((1, 24), dtype=np.float32),
        )
        np.testing.assert_allclose(
            archive["paired_pose_reconstruction"],
            np.full((1, 48), 2.0, dtype=np.float32),
        )


def test_debug_sidecar_records_pose_detection_gaps() -> None:
    capture = RawPoseEegDebugCapture()
    capture.record_native_pose(
        PoseResult(
            timestamp_ms=10,
            received_time_s=1.0,
            image_width=640,
            image_height=480,
            landmarks=[],
            world_landmarks=[],
            pose_detected=False,
        )
    )

    path = capture.save(
        _output_path("gap_raw_pose_eeg_debug"),
        paired_frames=[],
        metadata={},
    )

    with np.load(path, allow_pickle=False) as archive:
        assert archive["pose_native_detected"].tolist() == [False]
        assert np.isnan(archive["pose_native_image_landmarks"]).all()
        assert np.isnan(archive["pose_native_world_landmarks"]).all()
        assert archive["paired_pose_feature_vector"].shape == (0, 0)


def test_paired_collection_feeds_processed_streams_to_debug_capture() -> None:
    packet = EegPacket(
        packet_id=3,
        samples=np.zeros((4, 200), dtype=np.float32),
        start_time_s=1.0,
        end_time_s=2.0,
        sample_rate=250,
    )

    class FakeEegStream:
        def __init__(self) -> None:
            self.packet = packet

        def pop_packet(self) -> EegPacket | None:
            packet_to_return = self.packet
            self.packet = None
            return packet_to_return

    class FakePoseStream:
        def __init__(self) -> None:
            self.frames = [
                _processed_pose(1000, 1.3, 0.0),
                _processed_pose(3000, 3.7, 2.0),
            ]

        def get_latest(self) -> PoseLatentFrame:
            if self.frames:
                return self.frames.pop(0)
            return _processed_pose(3000, 3.7, 2.0)

    capture = RawPoseEegDebugCapture()
    paired = collect_paired_frames(
        FakeEegStream(),
        FakePoseStream(),
        duration_s=0.02,
        max_pose_gap_s=3.0,
        poll_delay_s=0.001,
        debug_capture=capture,
    )
    assert len(paired) == 1

    path = capture.save(
        _output_path("collected_raw_pose_eeg_debug"),
        paired_frames=paired,
        metadata={},
    )
    with np.load(path, allow_pickle=False) as archive:
        assert archive["eeg_packet_id"].tolist() == [3]
        assert archive["pose_processed_timestamp_ms"].tolist() == [1000, 3000]
        np.testing.assert_allclose(
            archive["paired_pose_feature_vector"],
            np.ones((1, 48), dtype=np.float32),
        )


def test_native_eeg_observer_runs_before_channel_selection() -> None:
    observed = []
    streamer = SignalStreamer(
        sample_rate=10,
        packet_size=4,
        packet_stride=4,
        channel_indices=(1, 3),
        bandstop_hz=None,
        raw_sample_observer=lambda samples, times, source_times, received: observed.append(
            (samples.copy(), times.copy(), source_times.copy(), received)
        ),
    )
    raw = np.arange(20, dtype=np.float32).reshape(4, 5)
    source_times = np.arange(4, dtype=np.float64) + 10.0

    streamer._append_samples(raw, source_timestamps=source_times)
    packet = streamer.pop_packet()

    assert packet is not None
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0][0], raw)
    np.testing.assert_array_equal(observed[0][2], source_times)
    np.testing.assert_array_equal(packet.samples, raw[:, (1, 3)].T)
