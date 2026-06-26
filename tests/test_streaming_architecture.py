from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from eeg_encoding import (
    EegPoseModelConfig,
    EegPoseVAE,
    ReadinessMetrics,
    build_context_windows,
    format_training_report,
    load_model,
    score_readiness,
    set_trainable_scope,
    should_commit_profile_update,
    train_model,
    train_session_model,
    trainable_parameter_names,
    transform_eeg_for_model,
    transformed_eeg_band_lengths,
    transformed_eeg_feature_count,
    trusted_calibration_mask,
)
from pose_encoding import PoseLatentFrame, PoseLatentStream
from streaming.eeg import EegPacket, SignalStreamer
from streaming import (
    CalibrationOverlayState,
    RegionScores,
    PoseLatentBuffer,
    dummy_positions_for_block,
    collect_paired_frames,
    load_paired_arrays,
    pair_packet,
    positions_from_feature_vector,
    region_scores,
    save_paired_frames,
    select_next_movement_block,
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
    band_lengths = transformed_eeg_band_lengths(
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
    assert band_lengths == (13, 13, 25, 50, 100)
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


def test_eeg_band_adapter_initializes_as_identity() -> None:
    config = EegPoseModelConfig(
        n_channels=2,
        n_samples=200,
        eeg_feature_count=201,
        pose_latent_dim=3,
        context_packet_count=2,
        hidden_dim=16,
        model_latent_dim=5,
        wavelet_band_lengths=(13, 13, 25, 50, 100),
    )
    model = EegPoseVAE(config)
    features = torch.randn(4, 2, 2, 201)

    adapted = model.adapt_eeg_features(features)

    torch.testing.assert_close(adapted, features)


def test_eeg_band_adapter_changes_predictions_after_update() -> None:
    config = EegPoseModelConfig(
        n_channels=2,
        n_samples=200,
        eeg_feature_count=201,
        pose_latent_dim=3,
        context_packet_count=2,
        hidden_dim=16,
        model_latent_dim=5,
        wavelet_band_lengths=(13, 13, 25, 50, 100),
    )
    model = EegPoseVAE(config)
    features = torch.randn(4, 2, 2, 201)

    with torch.no_grad():
        before = model.predict_pose_latent(features)
        model.band_adapter.scale[0, 0] = 1.5
        model.band_adapter.bias[1, 2] = 0.25
        after = model.predict_pose_latent(features)

    assert not torch.allclose(before, after)


def test_daily_recalibration_freezes_first_encoder_layer() -> None:
    config = EegPoseModelConfig(
        n_channels=2,
        n_samples=200,
        eeg_feature_count=201,
        pose_latent_dim=3,
        context_packet_count=2,
        hidden_dim=16,
        model_latent_dim=5,
        wavelet_band_lengths=(13, 13, 25, 50, 100),
    )
    model = EegPoseVAE(config)

    set_trainable_scope(model, "session")
    trainable = trainable_parameter_names(model)

    assert "band_adapter.scale" in trainable
    assert "latent_mean.weight" in trainable
    assert "latent_log_variance.weight" in trainable
    assert "pose_head.weight" in trainable
    assert "encoder.0.weight" not in trainable
    assert "encoder.2.weight" not in trainable

    set_trainable_scope(model, "session_deep")
    trainable = trainable_parameter_names(model)

    assert "encoder.0.weight" not in trainable
    assert "encoder.2.weight" in trainable


def test_session_update_does_not_mutate_base_model() -> None:
    config = EegPoseModelConfig(
        n_channels=2,
        n_samples=200,
        eeg_feature_count=201,
        pose_latent_dim=3,
        context_packet_count=2,
        hidden_dim=16,
        model_latent_dim=5,
        wavelet_band_lengths=(13, 13, 25, 50, 100),
    )
    base_model = EegPoseVAE(config)
    session_model = copy.deepcopy(base_model)
    base_state = {
        name: value.detach().clone()
        for name, value in base_model.state_dict().items()
    }
    set_trainable_scope(session_model, "session")
    optimizer = torch.optim.Adam(
        [p for p in session_model.parameters() if p.requires_grad],
        lr=1e-2,
    )
    features = torch.randn(4, 2, 2, 201)
    target = torch.randn(4, 3)

    predicted = session_model.predict_pose_latent(features)
    loss = torch.nn.functional.smooth_l1_loss(predicted, target)
    loss.backward()
    optimizer.step()

    for name, value in base_model.state_dict().items():
        torch.testing.assert_close(value, base_state[name])


def test_fake_daily_recalibration_saves_session_model() -> None:
    data_path = output_path("session_calibration_paired.npz")
    base_model_path = output_path("session_base.pt")
    profiles_root = output_path("profiles")
    rng = np.random.default_rng(21)
    eeg = rng.normal(size=(8, 2, 200)).astype(np.float32)
    pose_latent = rng.normal(size=(8, 3)).astype(np.float32)
    np.savez_compressed(
        data_path,
        eeg=eeg,
        pose_latent=pose_latent,
        pose_confidence=np.full(8, 0.9, dtype=np.float32),
        interpolation_confidence=np.full(8, 0.9, dtype=np.float32),
        pose_reconstruction_error=np.full(8, 0.05, dtype=np.float32),
    )
    train_model(
        data_path,
        base_model_path,
        epochs=1,
        batch_size=4,
        hidden_dim=16,
        model_latent_dim=5,
        pose_checkpoint=None,
    )

    report = train_session_model(
        user_id="test_user",
        base_checkpoint=base_model_path,
        calibration_data=data_path,
        session_id="test_session",
        profiles_root=profiles_root,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        pose_checkpoint=None,
    )

    assert report.adaptation_mode == "session"
    assert (
        profiles_root
        / "test_user"
        / "sessions"
        / "test_session"
        / "session_model.pt"
    ).exists()


def test_profile_commit_rejects_regressed_metrics() -> None:
    old_metrics = {
        "validation_pose_mae": 0.3,
        "stationary_false_positive_score": 0.2,
        "movement_response_score": 0.7,
    }

    assert not should_commit_profile_update(
        old_metrics,
        {
            "validation_pose_mae": 0.35,
            "stationary_false_positive_score": 0.2,
            "movement_response_score": 0.7,
        },
    )
    assert should_commit_profile_update(
        old_metrics,
        {
            "validation_pose_mae": 0.25,
            "stationary_false_positive_score": 0.15,
            "movement_response_score": 0.75,
        },
    )


def test_low_confidence_pose_labels_pause_calibration_learning() -> None:
    archive = {
        "pose_confidence": np.asarray([0.9, 0.4, 0.8], dtype=np.float32),
        "interpolation_confidence": np.asarray([0.8, 0.8, 0.2], dtype=np.float32),
        "pose_reconstruction_error": np.asarray([0.05, 0.05, 0.05], dtype=np.float32),
    }

    mask = trusted_calibration_mask(archive)

    np.testing.assert_array_equal(mask, np.asarray([True, False, False]))


def test_readiness_requires_eeg_model_metrics() -> None:
    result = score_readiness(
        ReadinessMetrics(
            trusted_sample_count=100,
            mean_pose_confidence=1.0,
            mean_interpolation_confidence=1.0,
            mean_pose_reconstruction_error=0.0,
        )
    )

    assert result.score == 0.0
    assert result.ready is False


def test_readiness_label_quality_is_only_a_gate_not_score_boost() -> None:
    clean_labels = ReadinessMetrics(
        trusted_sample_count=100,
        mean_pose_confidence=1.0,
        mean_interpolation_confidence=1.0,
        mean_pose_reconstruction_error=0.0,
        decoded_pose_error=0.02,
        stationary_false_positive_score=0.0,
        movement_response_score=0.8,
    )
    poor_label_summary = ReadinessMetrics(
        trusted_sample_count=100,
        mean_pose_confidence=0.0,
        mean_interpolation_confidence=0.0,
        mean_pose_reconstruction_error=10.0,
        decoded_pose_error=0.02,
        stationary_false_positive_score=0.0,
        movement_response_score=0.8,
    )

    clean_result = score_readiness(clean_labels)
    poor_label_result = score_readiness(poor_label_summary)

    assert clean_result.score == poor_label_result.score
    assert clean_result.ready == poor_label_result.ready


def test_readiness_requires_enough_trusted_samples() -> None:
    result = score_readiness(
        ReadinessMetrics(
            trusted_sample_count=1,
            mean_pose_confidence=1.0,
            mean_interpolation_confidence=1.0,
            mean_pose_reconstruction_error=0.0,
            decoded_pose_error=0.0,
            stationary_false_positive_score=0.0,
            movement_response_score=1.0,
        )
    )

    assert result.score >= 0.8
    assert result.ready is False


def test_calibration_positions_and_region_scores_target_weak_area() -> None:
    truth = np.zeros((8, 3), dtype=np.float32)
    predicted = truth.copy()
    predicted[[0, 2, 4], 0] = 1.0

    scores = region_scores(truth, predicted)
    block = select_next_movement_block(scores)

    assert scores.left_arm > scores.right_arm
    assert block.region == "left_arm"


def test_dummy_calibration_motion_changes_target_pose() -> None:
    start = dummy_positions_for_block("left_arm_raise", elapsed_s=0.0, duration_s=4.0)
    mid = dummy_positions_for_block("left_arm_raise", elapsed_s=2.0, duration_s=4.0)

    assert mid[4, 1] < start[4, 1]


def test_positions_from_feature_vector_uses_position_features_only() -> None:
    vector = np.arange(48, dtype=np.float32)

    positions = positions_from_feature_vector(vector)

    assert positions.shape == (8, 3)
    np.testing.assert_allclose(positions.reshape(-1), np.arange(24, dtype=np.float32))


def test_calibration_overlay_renders_truth_dummy_and_eeg_layers() -> None:
    class FakeCv2:
        LINE_AA = 16

        def __init__(self) -> None:
            self.lines = []
            self.circles = []
            self.weights = []

        def line(self, *args, **kwargs) -> None:
            self.lines.append((args, kwargs))

        def circle(self, *args, **kwargs) -> None:
            self.circles.append((args, kwargs))

        def addWeighted(self, overlay, alpha, frame, beta, gamma, dst) -> None:
            self.weights.append(alpha)
            dst[:] = overlay

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    overlay = CalibrationOverlayState()
    truth = np.zeros(48, dtype=np.float32)
    eeg = np.zeros(48, dtype=np.float32)
    eeg[0] = 1.0
    overlay.update_truth(truth)
    overlay.update_eeg(eeg)
    overlay.update_dummy(select_next_movement_block(RegionScores(1, 0, 0, 0, 0, 0)), 0.5)
    cv2 = FakeCv2()

    overlay.render(frame, cv2, pose_result=None, mirror_x=False)

    assert cv2.lines
    assert cv2.circles
    assert cv2.weights


def test_pose_encoding_exports_public_wrapper() -> None:
    assert PoseLatentStream is not None
