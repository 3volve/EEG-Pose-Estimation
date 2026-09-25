from __future__ import annotations

import copy
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

from eeg_encoding import (
    EegPoseModelConfig,
    EegPoseVAE,
    OnlineEegCalibrator,
    ReadinessMetrics,
    build_profile_model,
    build_context_windows,
    format_training_report,
    install_profile_model,
    load_model,
    profile_start_checkpoint,
    reset_band_adapter_identity,
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
from eeg_encoding.model import save_model
from eeg_encoding.personalization import load_profile_session
from pose_encoding import PoseLatentFrame, PoseLatentStream
from streaming.eeg import EegPacket, SignalStreamer
from streaming.pose import PoseLandmark, PoseResult
from streaming import (
    CalibrationMovementBlock,
    CalibrationOverlayState,
    CalibrationDisplayStatus,
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
    validate_profile_block,
)
from streaming.records import PairedTrainingFrame
from streaming.calibration import _aligned_head_guide_positions


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
    buffer.add(make_pose_frame(10000, 10.3, 0.0, confidence=0.9))
    buffer.add(make_pose_frame(10500, 10.9, 2.0, confidence=0.7))

    assert buffer.latest_time_s == 10.5
    interpolated = buffer.latent_at(10.25)

    assert interpolated is not None
    np.testing.assert_allclose(interpolated.latent, np.ones(4, dtype=np.float32))
    assert interpolated.pose_confidence == 0.8
    assert 0.0 < interpolated.interpolation_confidence < 1.0


def test_pose_buffer_rejects_missing_or_stale_brackets() -> None:
    buffer = PoseLatentBuffer(max_gap_s=0.1)
    buffer.add(make_pose_frame(10000, 10.3, 0.0))
    buffer.add(make_pose_frame(10500, 10.9, 2.0))

    assert buffer.latent_at(9.9) is None
    assert buffer.latent_at(10.25) is None
    assert buffer.latent_at(10.6) is None


def test_pair_packet_targets_eeg_end_time() -> None:
    buffer = PoseLatentBuffer(max_gap_s=3.0)
    buffer.add(make_pose_frame(4000, 4.3, 0.0))
    buffer.add(make_pose_frame(6000, 6.9, 2.0))
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
                make_pose_frame(10000, 10.3, 0.0),
                make_pose_frame(10500, 10.9, 2.0),
            ]

        def get_latest(self) -> PoseLatentFrame | None:
            if self.frames:
                return self.frames.pop(0)
            return make_pose_frame(10500, 10.9, 2.0)

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


def test_adapter_ablation_scopes_only_unfreeze_requested_layers() -> None:
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

    set_trainable_scope(model, "adapter_only")
    trainable = trainable_parameter_names(model)

    assert trainable == ("band_adapter.scale", "band_adapter.bias")

    set_trainable_scope(model, "adapter_head")
    trainable = trainable_parameter_names(model)

    assert "band_adapter.scale" in trainable
    assert "band_adapter.bias" in trainable
    assert "pose_head.weight" in trainable
    assert "pose_head.bias" in trainable
    assert "latent_mean.weight" not in trainable
    assert "latent_log_variance.weight" not in trainable
    assert "encoder.0.weight" not in trainable


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


class DummyPoseDecoder(torch.nn.Module):
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(
            latents.shape[0],
            48,
            dtype=latents.dtype,
            device=latents.device,
        )
        output[:, : latents.shape[1]] = latents
        output[:, 24 : 24 + latents.shape[1]] = latents
        return output


def make_online_model() -> EegPoseVAE:
    return EegPoseVAE(
        EegPoseModelConfig(
            n_channels=2,
            n_samples=200,
            eeg_feature_count=201,
            pose_latent_dim=3,
            context_packet_count=2,
            hidden_dim=16,
            model_latent_dim=5,
            wavelet_band_lengths=(13, 13, 25, 50, 100),
            standardize_pose_latents=False,
        )
    )


def make_online_frame(index: int, *, trusted: bool = True) -> PairedTrainingFrame:
    rng = np.random.default_rng(index)
    return PairedTrainingFrame(
        packet_id=index,
        eeg=rng.normal(size=(2, 200)).astype(np.float32),
        target_time_s=float(index),
        pose_latent=np.asarray([0.2, -0.1, 0.3], dtype=np.float32),
        pose_confidence=0.9 if trusted else 0.1,
        pose_reconstruction_error=0.02,
        interpolation_confidence=0.9,
    )


def test_online_calibrator_rejects_untrusted_frames() -> None:
    calibrator = OnlineEegCalibrator(
        make_online_model(),
        DummyPoseDecoder(),
        min_batch_size=2,
        update_every=2,
        steps_per_update=1,
    )

    accepted = calibrator.observe(make_online_frame(1, trusted=False))

    assert accepted is False
    assert calibrator.trusted_sample_count == 0
    assert calibrator.skipped_samples == 1


def test_online_calibrator_updates_after_trusted_batch() -> None:
    model = make_online_model()
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    calibrator = OnlineEegCalibrator(
        model,
        DummyPoseDecoder(),
        min_batch_size=2,
        update_every=2,
        steps_per_update=1,
        batch_size=2,
    )

    calibrator.observe(make_online_frame(1))
    calibrator.observe(make_online_frame(2))
    status = calibrator.status()

    assert status.trusted_samples == 2
    assert status.update_count == 1
    assert status.latest_loss is not None
    assert any(
        not torch.allclose(value, before[name])
        for name, value in model.state_dict().items()
        if name in before
    )


def test_online_calibrator_saves_session_artifacts() -> None:
    session_dir = output_path("online_calibration_session")
    calibrator = OnlineEegCalibrator(
        make_online_model(),
        DummyPoseDecoder(),
        min_batch_size=2,
        update_every=2,
        steps_per_update=1,
        batch_size=2,
    )
    calibrator.observe(make_online_frame(1))
    calibrator.observe(make_online_frame(2))

    calibrator.save(session_dir)

    assert (session_dir / "session_model.pt").exists()
    assert (session_dir / "trusted_calibration_samples.npz").exists()
    assert (session_dir / "online_status.json").exists()


def test_profile_start_uses_fallback_until_profile_exists() -> None:
    profiles_root = output_path("profile_start_root")
    fallback = output_path("fallback_model.pt")

    start = profile_start_checkpoint(
        "profile_user",
        fallback,
        profiles_root=profiles_root,
    )

    assert start == fallback


def test_install_profile_model_sets_primary_profile_model() -> None:
    profiles_root = output_path("install_profile_root")
    source = output_path("install_source_model.pt")
    save_model(source, make_online_model())

    profile_model = install_profile_model(
        "profile_user",
        source,
        profiles_root=profiles_root,
        update_kind="test_install",
    )

    assert profile_model.exists()
    assert profile_start_checkpoint(
        "profile_user",
        source,
        profiles_root=profiles_root,
    ) == profile_model


def make_profile_build_dataset(path: Path, seed: int, *, n_samples: int = 10) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        packet_id=np.arange(n_samples, dtype=np.int64),
        eeg=rng.normal(size=(n_samples, 2, 200)).astype(np.float32),
        target_time_s=np.arange(n_samples, dtype=np.float64),
        pose_latent=rng.normal(size=(n_samples, 3)).astype(np.float32),
        pose_confidence=np.full(n_samples, 0.95, dtype=np.float32),
        interpolation_confidence=np.full(n_samples, 0.9, dtype=np.float32),
        pose_reconstruction_error=np.full(n_samples, 0.04, dtype=np.float32),
    )


def test_reset_band_adapter_identity_restores_profile_save_state() -> None:
    model = make_online_model()
    with torch.no_grad():
        model.band_adapter.scale.fill_(1.5)
        model.band_adapter.bias.fill_(0.25)

    reset_band_adapter_identity(model)

    torch.testing.assert_close(model.band_adapter.scale, torch.ones_like(model.band_adapter.scale))
    torch.testing.assert_close(model.band_adapter.bias, torch.zeros_like(model.band_adapter.bias))


def unique_output_path(name: str) -> Path:
    return output_path(f"{name}_{uuid.uuid4().hex}")


def test_profile_build_creates_primary_profile_from_paired_data() -> None:
    root = unique_output_path("profile_build")
    profiles_root = root / "profiles"
    data_path = root / "session_01.npz"
    base_path = root / "base.pt"
    make_profile_build_dataset(data_path, 31)
    train_model(
        data_path,
        base_path,
        epochs=1,
        batch_size=4,
        hidden_dim=16,
        model_latent_dim=5,
        pose_checkpoint=None,
    )

    report = build_profile_model(
        user_id="profile_user",
        new_session_data=data_path,
        base_checkpoint=base_path,
        profiles_root=profiles_root,
        epochs=1,
        inner_epochs=1,
        query_epochs=1,
        batch_size=4,
        pose_checkpoint=None,
    )

    profile_model = profiles_root / "profile_user" / "profile_model.pt"
    history_dir = Path(report.history_dir)
    assert report.committed is True
    assert profile_model.exists()
    assert (history_dir / "proposed_profile_model.pt").exists()
    assert (history_dir / "profile_build_report.json").exists()
    assert (history_dir / "profile_build_summary.csv").exists()
    session_dirs = [
        path
        for path in (profiles_root / "profile_user" / "profile_history").iterdir()
        if (path / "paired_profile_session.npz").exists()
    ]
    assert len(session_dirs) == 1
    assert (session_dirs[0] / "paired_profile_session.npz").exists()
    model = load_model(profile_model).model
    torch.testing.assert_close(model.band_adapter.scale, torch.ones_like(model.band_adapter.scale))
    torch.testing.assert_close(model.band_adapter.bias, torch.zeros_like(model.band_adapter.bias))


def test_profile_finalization_failure_leaves_primary_profile_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = unique_output_path("profile_build_atomic_failure")
    profiles_root = root / "profiles"
    data_path = root / "session_01.npz"
    base_path = root / "base.pt"
    make_profile_build_dataset(data_path, 39)
    train_model(
        data_path,
        base_path,
        epochs=1,
        batch_size=4,
        hidden_dim=16,
        model_latent_dim=5,
        pose_checkpoint=None,
    )
    profile_model = profiles_root / "profile_user" / "profile_model.pt"
    real_replace = Path.replace

    def fail_profile_replace(path: Path, target: Path):
        if Path(target) == profile_model:
            raise OSError("simulated profile finalization failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_profile_replace)

    with pytest.raises(OSError, match="simulated profile finalization failure"):
        build_profile_model(
            user_id="profile_user",
            new_session_data=data_path,
            base_checkpoint=base_path,
            profiles_root=profiles_root,
            epochs=1,
            inner_epochs=1,
            query_epochs=1,
            batch_size=4,
            pose_checkpoint=None,
        )

    assert not profile_model.exists()
    assert not (profile_model.parent / "profile_metrics.json").exists()
    assert not list(profile_model.parent.glob(".profile_*.pending"))


def test_second_profile_build_uses_existing_profile_and_session_history() -> None:
    root = unique_output_path("profile_build_existing")
    profiles_root = root / "profiles"
    first_data = root / "session_01.npz"
    second_data = root / "session_02.npz"
    base_path = root / "base.pt"
    make_profile_build_dataset(first_data, 41)
    make_profile_build_dataset(second_data, 42)
    train_model(
        first_data,
        base_path,
        epochs=1,
        batch_size=4,
        hidden_dim=16,
        model_latent_dim=5,
        pose_checkpoint=None,
    )
    first = build_profile_model(
        user_id="profile_user",
        new_session_data=first_data,
        base_checkpoint=base_path,
        profiles_root=profiles_root,
        epochs=1,
        inner_epochs=1,
        query_epochs=1,
        batch_size=4,
        pose_checkpoint=None,
    )

    second = build_profile_model(
        user_id="profile_user",
        new_session_data=second_data,
        base_checkpoint=base_path,
        profiles_root=profiles_root,
        epochs=1,
        inner_epochs=1,
        query_epochs=1,
        batch_size=4,
        pose_checkpoint=None,
    )

    assert second.start_checkpoint == str(profiles_root / "profile_user" / "profile_model.pt")
    assert second.session_count == 2
    assert Path(first.history_dir) != Path(second.history_dir)


def test_profile_session_uses_file_level_query_split() -> None:
    root = unique_output_path("profile_file_split")
    first = root / "source_00.npz"
    second = root / "source_01.npz"
    make_profile_build_dataset(first, 51, n_samples=6)
    make_profile_build_dataset(second, 52, n_samples=4)

    session = load_profile_session([first, second], query_fraction=0.5)

    np.testing.assert_array_equal(session.support_indices, np.arange(6))
    np.testing.assert_array_equal(session.query_indices, np.arange(6, 10))


def test_profile_session_uses_block_level_query_split_when_available() -> None:
    root = unique_output_path("profile_block_split")
    path = root / "guided_session.npz"
    make_profile_build_dataset(path, 53, n_samples=9)
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays.update(
        {
            "profile_posture": np.asarray("standing"),
            "profile_block_id": np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 3], dtype=np.int64),
            "profile_block_name": np.asarray(
                [
                    "neutral_rest",
                    "neutral_rest",
                    "left_arm_raise",
                    "left_arm_raise",
                    "right_arm_raise",
                    "right_arm_raise",
                    "torso_side_lean",
                    "torso_side_lean",
                    "torso_side_lean",
                ]
            ),
            "profile_block_repeat_index": np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "profile_block_accepted": np.ones(9, dtype=np.bool_),
            "profile_block_summary_json": np.asarray(
                '[{"movement_name":"neutral_rest","accepted":true},'
                '{"movement_name":"left_arm_raise","accepted":true},'
                '{"movement_name":"right_arm_raise","accepted":true},'
                '{"movement_name":"torso_side_lean","accepted":true}]'
            ),
        }
    )
    np.savez_compressed(path, **arrays)

    session = load_profile_session(path, query_fraction=0.25)

    np.testing.assert_array_equal(session.support_indices, np.arange(6))
    np.testing.assert_array_equal(session.query_indices, np.arange(6, 9))
    assert session.posture == "standing"
    assert session.block_summary


def test_profile_session_excludes_rejected_block_samples() -> None:
    root = unique_output_path("profile_rejected_blocks")
    path = root / "guided_session.npz"
    make_profile_build_dataset(path, 54, n_samples=6)
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays.update(
        {
            "profile_block_id": np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
            "profile_block_accepted": np.asarray([True, True, False, False, True, True]),
            "profile_block_summary_json": np.asarray(
                '[{"movement_name":"neutral_rest","accepted":true},'
                '{"movement_name":"left_arm_raise","accepted":false,'
                '"reject_reason":"insufficient_target_motion"},'
                '{"movement_name":"right_arm_raise","accepted":true}]'
            ),
        }
    )
    np.savez_compressed(path, **arrays)

    session = load_profile_session(path, query_fraction=0.5)

    assert len(session.eeg) == 4
    np.testing.assert_array_equal(session.block_ids, np.asarray([0, 0, 2, 2]))


def test_profile_build_treats_data_folders_as_distinct_sessions() -> None:
    root = unique_output_path("profile_build_folder_sessions")
    profiles_root = root / "profiles"
    first_dir = root / "session_a"
    second_dir = root / "session_b"
    base_path = root / "base.pt"
    make_profile_build_dataset(first_dir / "run_01.npz", 61)
    make_profile_build_dataset(first_dir / "run_02.npz", 62)
    make_profile_build_dataset(second_dir / "run_01.npz", 63)
    make_profile_build_dataset(second_dir / "run_02.npz", 64)
    train_model(
        first_dir / "run_01.npz",
        base_path,
        epochs=1,
        batch_size=4,
        hidden_dim=16,
        model_latent_dim=5,
        pose_checkpoint=None,
    )

    report = build_profile_model(
        user_id="profile_user",
        new_session_data=[first_dir, second_dir],
        base_checkpoint=base_path,
        profiles_root=profiles_root,
        epochs=1,
        inner_epochs=1,
        query_epochs=1,
        batch_size=4,
        pose_checkpoint=None,
    )

    assert report.session_count == 2
    history_dirs = [
        path
        for path in (profiles_root / "profile_user" / "profile_history").iterdir()
        if (path / "paired_profile_session.npz").exists()
    ]
    assert len(history_dirs) == 2
    assert all((path / "source_00.npz").exists() for path in history_dirs)
    assert all((path / "source_01.npz").exists() for path in history_dirs)


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
        "decoded_pose_error": 0.3,
        "stationary_false_positive_score": 0.2,
        "movement_response_score": 0.7,
        "readiness_score": 0.8,
    }

    assert not should_commit_profile_update(
        old_metrics,
        {
            "decoded_pose_error": 0.35,
            "stationary_false_positive_score": 0.2,
            "movement_response_score": 0.7,
            "readiness_score": 0.8,
        },
    )
    assert should_commit_profile_update(
        old_metrics,
        {
            "decoded_pose_error": 0.25,
            "stationary_false_positive_score": 0.15,
            "movement_response_score": 0.75,
            "readiness_score": 0.85,
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


def test_torso_side_lean_keeps_hip_root_and_aligns_guide_head() -> None:
    neutral = dummy_positions_for_block(
        "torso_side_lean",
        elapsed_s=0.0,
        duration_s=8.0,
    )
    leaned = dummy_positions_for_block(
        "torso_side_lean",
        elapsed_s=2.0,
        duration_s=8.0,
    )

    np.testing.assert_allclose(leaned[[6, 7]], neutral[[6, 7]])
    assert leaned[[0, 1], 0].mean() > neutral[[0, 1], 0].mean()

    neck, head = _aligned_head_guide_positions(leaned)
    hip_center = leaned[[6, 7]].mean(axis=0)
    shoulder_center = leaned[[0, 1]].mean(axis=0)
    torso_axis = shoulder_center - hip_center
    np.testing.assert_allclose(
        np.cross(torso_axis, head - shoulder_center),
        np.zeros(3),
        atol=1e-6,
    )
    assert np.dot(head - neck, torso_axis) > 0.0


def test_profile_block_validator_accepts_torso_side_lean() -> None:
    vectors = feature_vectors_for_block("torso_side_lean")

    result = validate_profile_block(
        block_id=7,
        block=CalibrationMovementBlock(
            "torso_side_lean",
            "shoulders_core",
            8.0,
        ),
        repeat_index=0,
        start_time_s=10.0,
        end_time_s=18.0,
        feature_vectors=vectors,
        pose_confidences=[0.85] * len(vectors),
        paired_sample_count=20,
    )

    assert result.accepted is True
    assert result.reject_reason == ""


def feature_vectors_for_block(name: str, *, duration_s: float = 8.0, samples: int = 12) -> list[np.ndarray]:
    vectors = []
    for elapsed_s in np.linspace(0.0, duration_s, samples):
        vector = np.zeros(48, dtype=np.float32)
        vector[:24] = dummy_positions_for_block(
            name,
            elapsed_s=float(elapsed_s),
            duration_s=duration_s,
        ).reshape(-1)
        vectors.append(vector)
    return vectors


def test_profile_block_validator_accepts_imperfect_followed_movement() -> None:
    vectors = feature_vectors_for_block("left_arm_raise")
    for vector in vectors:
        vector[:24] += 0.04

    result = validate_profile_block(
        block_id=1,
        block=CalibrationMovementBlock("left_arm_raise", "left_arm", 8.0),
        repeat_index=0,
        start_time_s=10.0,
        end_time_s=18.0,
        feature_vectors=vectors,
        pose_confidences=[0.85] * len(vectors),
        paired_sample_count=20,
    )

    assert result.accepted is True
    assert result.reject_reason == ""


def test_profile_block_validator_rejects_low_pose_quality() -> None:
    vectors = feature_vectors_for_block("right_arm_raise")

    result = validate_profile_block(
        block_id=2,
        block=CalibrationMovementBlock("right_arm_raise", "right_arm", 8.0),
        repeat_index=0,
        start_time_s=10.0,
        end_time_s=18.0,
        feature_vectors=vectors,
        pose_confidences=[0.2] * len(vectors),
        paired_sample_count=20,
    )

    assert result.accepted is False
    assert result.reject_reason == "low_pose_quality"


def test_profile_block_validator_rejects_impossible_jumps() -> None:
    vectors = feature_vectors_for_block("left_arm_raise")
    vectors[5] = vectors[5].copy()
    vectors[5][0] += 4.0

    result = validate_profile_block(
        block_id=3,
        block=CalibrationMovementBlock("left_arm_raise", "left_arm", 8.0),
        repeat_index=0,
        start_time_s=10.0,
        end_time_s=18.0,
        feature_vectors=vectors,
        pose_confidences=[0.85] * len(vectors),
        paired_sample_count=20,
    )

    assert result.accepted is False
    assert result.reject_reason == "impossible_pose_jumps"


def test_profile_block_validator_rejects_inactive_movement_block() -> None:
    neutral = np.zeros(48, dtype=np.float32)
    neutral[:24] = dummy_positions_for_block(
        "neutral_rest",
        elapsed_s=0.0,
        duration_s=8.0,
    ).reshape(-1)
    vectors = [neutral.copy() for _ in range(12)]

    result = validate_profile_block(
        block_id=4,
        block=CalibrationMovementBlock("left_forward_reach", "left_arm", 8.0),
        repeat_index=0,
        start_time_s=10.0,
        end_time_s=18.0,
        feature_vectors=vectors,
        pose_confidences=[0.85] * len(vectors),
        paired_sample_count=20,
    )

    assert result.accepted is False
    assert result.reject_reason == "insufficient_target_motion"


def test_profile_block_validator_rejects_overactive_rest_block() -> None:
    vectors = feature_vectors_for_block("torso_side_lean")

    result = validate_profile_block(
        block_id=5,
        block=CalibrationMovementBlock("neutral_rest", "rest", 8.0, rest=True),
        repeat_index=0,
        start_time_s=10.0,
        end_time_s=18.0,
        feature_vectors=vectors,
        pose_confidences=[0.85] * len(vectors),
        paired_sample_count=20,
    )

    assert result.accepted is False
    assert result.reject_reason == "rest_too_active"


def test_profile_block_validator_rejects_wrong_motion_pattern() -> None:
    vectors = feature_vectors_for_block("torso_side_lean")

    result = validate_profile_block(
        block_id=6,
        block=CalibrationMovementBlock("left_arm_raise", "left_arm", 8.0),
        repeat_index=0,
        start_time_s=10.0,
        end_time_s=18.0,
        feature_vectors=vectors,
        pose_confidences=[0.85] * len(vectors),
        paired_sample_count=20,
    )

    assert result.accepted is False
    assert result.reject_reason in {"insufficient_target_motion", "wrong_motion_pattern"}


def test_positions_from_feature_vector_uses_position_features_only() -> None:
    vector = np.arange(48, dtype=np.float32)

    positions = positions_from_feature_vector(vector)

    assert positions.shape == (8, 3)
    np.testing.assert_allclose(positions.reshape(-1), np.arange(24, dtype=np.float32))


def test_calibration_overlay_renders_truth_dummy_and_eeg_layers() -> None:
    class FakeCv2:
        LINE_AA = 16
        FONT_HERSHEY_SIMPLEX = 0

        def __init__(self) -> None:
            self.lines = []
            self.circles = []
            self.weights = []
            self.rectangles = []
            self.text = []

        def line(self, *args, **kwargs) -> None:
            self.lines.append((args, kwargs))

        def circle(self, *args, **kwargs) -> None:
            self.circles.append((args, kwargs))

        def addWeighted(self, overlay, alpha, frame, beta, gamma, dst) -> None:
            self.weights.append(alpha)
            dst[:] = overlay

        def rectangle(self, *args, **kwargs) -> None:
            self.rectangles.append((args, kwargs))

        def putText(self, *args, **kwargs) -> None:
            self.text.append((args, kwargs))

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    overlay = CalibrationOverlayState()
    truth = np.zeros(48, dtype=np.float32)
    eeg = np.zeros(48, dtype=np.float32)
    eeg[0] = 1.0
    overlay.update_truth(truth)
    overlay.update_eeg(eeg)
    overlay.update_dummy(select_next_movement_block(RegionScores(1, 0, 0, 0, 0, 0)), 0.5)
    overlay.update_status(
        CalibrationDisplayStatus(
            readiness_score=0.75,
            ready=False,
            trusted_samples=12,
            skipped_samples=3,
            update_count=2,
            latest_loss=0.4,
        )
    )
    cv2 = FakeCv2()

    overlay.render(frame, cv2, pose_result=None, mirror_x=False)

    assert cv2.lines
    assert cv2.circles
    assert cv2.weights
    assert cv2.rectangles
    assert cv2.text


def test_calibration_overlay_mirrors_skeleton_geometry() -> None:
    class FakeCv2:
        LINE_AA = 16

        def __init__(self) -> None:
            self.lines = []

        def line(self, *args, **kwargs) -> None:
            self.lines.append((args, kwargs))

        def circle(self, *args, **kwargs) -> None:
            pass

        def addWeighted(self, overlay, alpha, frame, beta, gamma, dst) -> None:
            dst[:] = overlay

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    truth = np.zeros(48, dtype=np.float32)
    truth[:6] = np.asarray([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    overlay = CalibrationOverlayState()
    overlay.update_truth(truth)
    cv2 = FakeCv2()

    overlay.render(frame, cv2, pose_result=None, mirror_x=True)

    assert cv2.lines[0][0][1][0] > cv2.lines[0][0][2][0]


def test_calibration_overlay_fits_truth_to_image_space_landmarks() -> None:
    class FakeCv2:
        LINE_AA = 16

        def __init__(self) -> None:
            self.lines = []

        def line(self, *args, **kwargs) -> None:
            self.lines.append((args, kwargs))

        def circle(self, *args, **kwargs) -> None:
            pass

        def addWeighted(self, overlay, alpha, frame, beta, gamma, dst) -> None:
            dst[:] = overlay

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    truth_positions = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.4, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [-1.6, 1.0, 0.0],
            [1.6, 1.0, 0.0],
            [-0.8, 1.2, 0.0],
            [0.8, 1.2, 0.0],
        ],
        dtype=np.float32,
    )
    expected_points = truth_positions[:, :2] * 30.0 + np.asarray([90.0, 70.0])
    landmarks = [PoseLandmark(None, None, None, None, None) for _ in range(25)]
    for point, landmark_index in zip(
        expected_points,
        (11, 12, 13, 14, 15, 16, 23, 24),
    ):
        landmarks[landmark_index] = PoseLandmark(
            float(point[0] / 200.0),
            float(point[1] / 200.0),
            0.0,
            1.0,
            1.0,
        )
    result = PoseResult(1, 1.0, 200, 200, landmarks, [], True)
    feature_vector = np.zeros(48, dtype=np.float32)
    feature_vector[:24] = truth_positions.reshape(-1)
    overlay = CalibrationOverlayState()
    overlay.update_truth(feature_vector)
    cv2 = FakeCv2()

    overlay.render(frame, cv2, pose_result=result, mirror_x=False)

    assert cv2.lines[0][0][1] == (60, 40)
    assert cv2.lines[0][0][2] == (120, 40)


def test_calibration_eeg_overlay_is_display_smoothed() -> None:
    overlay = CalibrationOverlayState(eeg_display_smoothing=0.25)
    first = np.zeros(48, dtype=np.float32)
    second = np.zeros(48, dtype=np.float32)
    second[0] = 4.0

    overlay.update_eeg(first)
    overlay.update_eeg(second)

    assert overlay._eeg_positions is not None
    assert overlay._eeg_positions[0, 0] == 1.0


def test_pose_encoding_exports_public_wrapper() -> None:
    assert PoseLatentStream is not None
