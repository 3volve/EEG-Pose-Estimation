from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from eeg_encoding import (
    EegPoseModelConfig,
    build_context_windows,
    format_training_report,
    load_model,
    train_model,
    transform_eeg_for_model,
    transformed_eeg_feature_count,
)
from pose_encoding import PoseLatentFrame, PoseLatentStream
from streaming.eeg import EegPacket, SignalStreamer
from streaming import (
    PoseLatentBuffer,
    collect_paired_frames,
    load_paired_arrays,
    pair_packet,
    save_paired_frames,
)
from streaming.records import PairedTrainingFrame


def output_path(name: str) -> Path:
    path = Path("test_outputs") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def make_pose_frame(
    timestamp_ms: int,
    received_time_s: float,
    latent_value: float,
    *,
    confidence: float = 0.8,
    reconstruction_error: float = 0.1,
) -> PoseLatentFrame:
    return PoseLatentFrame(
        timestamp_ms=timestamp_ms,
        received_time_s=received_time_s,
        feature_vector=np.zeros(48, dtype=np.float32),
        latent=np.full(4, latent_value, dtype=np.float32),
        reconstruction=np.zeros(48, dtype=np.float32),
        reconstruction_error=reconstruction_error,
        pose_detected=True,
        confidence=confidence,
    )


def test_signal_streamer_emits_packet_ids_and_end_times() -> None:
    streamer = SignalStreamer(
        sample_rate=10,
        packet_size=4,
        packet_stride=2,
        n_channels=2,
    )
    raw = np.arange(12, dtype=np.float32).reshape(6, 2)

    streamer._append_samples(raw)
    first = streamer.pop_packet()
    second = streamer.pop_packet()

    assert first is not None
    assert second is not None
    assert first.packet_id == 0
    assert second.packet_id == 1
    assert first.samples.shape == (2, 4)
    assert second.samples.shape == (2, 4)
    assert first.end_time_s > first.start_time_s
    assert second.end_time_s > first.end_time_s


def test_pose_buffer_interpolates_bracketing_latents() -> None:
    buffer = PoseLatentBuffer(max_gap_s=1.0)
    buffer.add(make_pose_frame(1, 10.0, 0.0, confidence=0.9))
    buffer.add(make_pose_frame(2, 10.5, 2.0, confidence=0.7))

    interpolated = buffer.latent_at(10.25)

    assert interpolated is not None
    np.testing.assert_allclose(interpolated.latent, np.ones(4, dtype=np.float32))
    assert interpolated.pose_confidence == 0.8
    assert 0.0 < interpolated.interpolation_confidence < 1.0


def test_pose_buffer_rejects_missing_or_stale_brackets() -> None:
    buffer = PoseLatentBuffer(max_gap_s=0.1)
    buffer.add(make_pose_frame(1, 10.0, 0.0))
    buffer.add(make_pose_frame(2, 10.5, 2.0))

    assert buffer.latent_at(9.9) is None
    assert buffer.latent_at(10.25) is None
    assert buffer.latent_at(10.6) is None


def test_pair_packet_targets_eeg_end_time() -> None:
    buffer = PoseLatentBuffer(max_gap_s=3.0)
    buffer.add(make_pose_frame(1, 4.0, 0.0))
    buffer.add(make_pose_frame(2, 6.0, 2.0))
    packet = EegPacket(
        packet_id=3,
        samples=np.zeros((2, 4), dtype=np.float32),
        start_time_s=1.0,
        end_time_s=5.0,
        sample_rate=10,
    )

    paired = pair_packet(packet, buffer)

    assert paired is not None
    assert paired.packet_id == 3
    assert paired.target_time_s == packet.end_time_s
    np.testing.assert_allclose(paired.pose_latent, np.ones(4, dtype=np.float32))


def test_collection_keeps_packet_pending_until_future_pose_arrives() -> None:
    class FakeEegStream:
        def __init__(self) -> None:
            self.packets = [
                EegPacket(
                    packet_id=1,
                    samples=np.zeros((2, 4), dtype=np.float32),
                    start_time_s=9.0,
                    end_time_s=10.25,
                    sample_rate=10,
                )
            ]

        def pop_packet(self) -> EegPacket | None:
            return self.packets.pop(0) if self.packets else None

    class FakePoseStream:
        def __init__(self) -> None:
            self.frames = [
                make_pose_frame(1, 10.0, 0.0),
                make_pose_frame(2, 10.5, 2.0),
            ]

        def get_latest(self) -> PoseLatentFrame | None:
            if self.frames:
                return self.frames.pop(0)
            return make_pose_frame(2, 10.5, 2.0)

    paired = collect_paired_frames(
        FakeEegStream(),
        FakePoseStream(),
        duration_s=0.02,
        max_pose_gap_s=1.0,
        poll_delay_s=0.001,
    )

    assert len(paired) == 1
    np.testing.assert_allclose(paired[0].pose_latent, np.ones(4, dtype=np.float32))


def test_paired_dataset_round_trip() -> None:
    path = output_path("paired_round_trip.npz")
    frame = PairedTrainingFrame(
        packet_id=1,
        eeg=np.ones((2, 4), dtype=np.float32),
        target_time_s=5.0,
        pose_latent=np.ones(4, dtype=np.float32),
        pose_confidence=0.8,
        pose_reconstruction_error=0.1,
        interpolation_confidence=0.7,
    )

    save_paired_frames(path, [frame], metadata={"source": "test"})
    arrays = load_paired_arrays(path)

    assert arrays["eeg"].shape == (1, 2, 4)
    assert arrays["pose_latent"].shape == (1, 4)
    assert arrays["source"].item() == "test"


def test_eeg_encoding_model_trains_saves_and_predicts() -> None:
    data_path = output_path("eeg_encoding_paired.npz")
    model_path = output_path("eeg_encoding.pt")
    rng = np.random.default_rng(4)
    eeg = rng.normal(size=(8, 2, 200)).astype(np.float32)
    pose_latent = rng.normal(size=(8, 3)).astype(np.float32)
    np.savez_compressed(data_path, eeg=eeg, pose_latent=pose_latent)

    train_model(
        data_path,
        model_path,
        epochs=2,
        batch_size=4,
        hidden_dim=16,
        model_latent_dim=5,
        pose_checkpoint=None,
    )
    predictor = load_model(model_path)
    packet = EegPacket(
        packet_id=9,
        samples=eeg[0],
        start_time_s=1.0,
        end_time_s=1.5,
        sample_rate=10,
    )

    prediction = predictor.predict(packet)

    assert prediction.packet_id == 9
    assert prediction.target_time_s == 1.5
    assert prediction.predicted_latent.shape == (3,)
    assert torch.isfinite(torch.from_numpy(prediction.predicted_latent)).all()


def test_eeg_encoding_training_report_and_checkpoint_initialization() -> None:
    data_path = output_path("eeg_encoding_report_paired.npz")
    first_model_path = output_path("eeg_encoding_report_first.pt")
    second_model_path = output_path("eeg_encoding_report_second.pt")
    rng = np.random.default_rng(9)
    eeg = rng.normal(size=(10, 2, 200)).astype(np.float32)
    pose_latent = rng.normal(size=(10, 3)).astype(np.float32)
    np.savez_compressed(data_path, eeg=eeg, pose_latent=pose_latent)

    first = train_model(
        data_path,
        first_model_path,
        epochs=1,
        batch_size=5,
        hidden_dim=16,
        model_latent_dim=5,
        pose_checkpoint=None,
    )
    second = train_model(
        data_path,
        second_model_path,
        epochs=1,
        batch_size=5,
        checkpoint_path=first_model_path,
        pose_checkpoint=None,
    )

    assert first.training_report is not None
    assert second.training_report is not None
    assert second.training_report.checkpoint_path is not None
    assert second.training_report.train.n_samples > 0
    assert second.training_report.validation is not None
    assert "validation" in format_training_report(second.training_report)


def test_eeg_encoding_wavelet_transform_uses_fixed_packet_shape() -> None:
    rng = np.random.default_rng(12)
    raw_eeg = rng.normal(size=(3, 2, 200)).astype(np.float32)
    feature_count = transformed_eeg_feature_count(
        n_samples=200,
        use_wavelet=True,
        wavelet="db4",
        wavelet_level=4,
        wavelet_mode="periodization",
    )
    config = EegPoseModelConfig(
        n_channels=2,
        n_samples=200,
        eeg_feature_count=feature_count,
        pose_latent_dim=3,
        wavelet="db4",
        wavelet_level=4,
        wavelet_mode="periodization",
        standardize_input=True,
    )

    transformed = transform_eeg_for_model(raw_eeg, config)

    assert transformed.shape == (3, 2, 201)
    assert np.isfinite(transformed).all()


def test_eeg_context_windows_are_causal_and_padded() -> None:
    eeg_features = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)

    windows = build_context_windows(eeg_features, context_packet_count=3)

    assert windows.shape == (4, 3, 2, 3)
    np.testing.assert_allclose(windows[0, 0], eeg_features[0])
    np.testing.assert_allclose(windows[0, 1], eeg_features[0])
    np.testing.assert_allclose(windows[0, 2], eeg_features[0])
    np.testing.assert_allclose(windows[2], eeg_features[:3])
    np.testing.assert_allclose(windows[3], eeg_features[1:4])


def test_pose_encoding_exports_public_wrapper() -> None:
    assert PoseLatentStream is not None
