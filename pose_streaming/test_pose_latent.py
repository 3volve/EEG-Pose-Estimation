from pathlib import Path

import numpy as np
import pytest
import torch

from pose_async import PoseLandmark, PoseResult
from pose_autoencoder import PoseAutoencoder, load_checkpoint, save_checkpoint
from pose_features import PoseFeatureExtractor, feature_dim
from pose_latent_stream import PoseLatentStream
from train_pose_autoencoder import load_feature_matrix, train_autoencoder


def make_landmarks(
    wrist_offset: float = 0.0,
    coordinate_scale: float = 1.0,
) -> list[PoseLandmark]:
    landmarks = [
        PoseLandmark(
            float(index) * coordinate_scale,
            float(index * 2) * coordinate_scale,
            0.0,
            0.8,
            0.6,
        )
        for index in range(33)
    ]
    wrist = landmarks[15]
    landmarks[15] = PoseLandmark(
        wrist.x + wrist_offset,
        wrist.y,
        wrist.z,
        wrist.visibility,
        wrist.presence,
    )
    return landmarks


def make_pose(
    timestamp_ms: int = 10,
    wrist_offset: float = 0.0,
    world_landmarks: list[PoseLandmark] | None = None,
) -> PoseResult:
    return PoseResult(
        timestamp_ms,
        1.5,
        640,
        480,
        make_landmarks(wrist_offset),
        world_landmarks or [],
        True,
    )


def test_feature_dimensions() -> None:
    assert feature_dim(False) == 24
    assert feature_dim() == 48


def test_features_are_centered_scaled_and_include_time_velocity() -> None:
    extractor = PoseFeatureExtractor(use_world_landmarks=False)
    first = extractor.extract(make_pose(timestamp_ms=100))
    second = extractor.extract(
        make_pose(timestamp_ms=200, wrist_offset=1.0)
    )

    first_positions = first.vector[:24].reshape(-1, 3)
    velocity = second.vector[24:].reshape(-1, 3)
    assert np.allclose((first_positions[-2] + first_positions[-1]) / 2, 0.0)
    assert np.isclose(np.linalg.norm(first_positions[0] - first_positions[1]), 1)
    assert np.allclose(first.vector[24:], 0.0)
    assert np.isclose(velocity[4, 0], 10.0 / np.sqrt(5.0))
    assert np.count_nonzero(velocity) == 1
    assert second.confidence == pytest.approx(0.7)


def test_velocity_accounts_for_elapsed_time() -> None:
    fast = PoseFeatureExtractor(use_world_landmarks=False)
    fast.extract(make_pose(timestamp_ms=100))
    fast_frame = fast.extract(
        make_pose(timestamp_ms=200, wrist_offset=1.0)
    )

    slow = PoseFeatureExtractor(use_world_landmarks=False)
    slow.extract(make_pose(timestamp_ms=100))
    slow_frame = slow.extract(
        make_pose(timestamp_ms=300, wrist_offset=1.0)
    )

    assert np.allclose(fast_frame.vector[24:], slow_frame.vector[24:] * 2)


def test_world_landmarks_are_preferred() -> None:
    image_only = PoseFeatureExtractor(
        include_velocity=False,
        use_world_landmarks=False,
    ).extract(make_pose())
    world = make_landmarks(wrist_offset=2.0, coordinate_scale=3.0)
    preferred = PoseFeatureExtractor(
        include_velocity=False,
        use_world_landmarks=True,
    ).extract(make_pose(world_landmarks=world))

    assert not np.allclose(image_only.vector, preferred.vector)


def test_velocity_can_be_disabled() -> None:
    frame = PoseFeatureExtractor(
        include_velocity=False,
        use_world_landmarks=False,
    ).extract(make_pose())

    assert frame.vector.shape == (24,)


def test_no_pose_returns_zero_frame_and_resets_velocity() -> None:
    extractor = PoseFeatureExtractor()
    extractor.extract(make_pose(timestamp_ms=100))
    no_pose = PoseResult(150, 1.6, 640, 480, [], [], False)

    empty = extractor.extract(no_pose)
    after_gap = extractor.extract(
        make_pose(timestamp_ms=200, wrist_offset=1.0)
    )

    assert empty.pose_detected is False
    assert empty.confidence == 0.0
    assert np.allclose(empty.vector, 0.0)
    assert np.allclose(after_gap.vector[24:], 0.0)


def test_autoencoder_architecture_and_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    model = PoseAutoencoder(latent_dim=4, hidden_dims=(16, 8))
    features = torch.randn(2, 48)
    reconstruction, latent = model(features)
    path = tmp_path / "pose.pt"

    save_checkpoint(path, model, {"source": "test"})
    restored, config = load_checkpoint(path)
    restored_reconstruction, restored_latent = restored(features)

    assert reconstruction.shape == features.shape
    assert latent.shape == (2, 4)
    assert torch.allclose(reconstruction, restored_reconstruction)
    assert torch.allclose(latent, restored_latent)
    assert config["source"] == "test"
    assert config["hidden_dims"] == (16, 8)


def test_training_tracks_validation_loss() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(32, 48)).astype(np.float32)

    model, history = train_autoencoder(
        features,
        latent_dim=4,
        epochs=20,
        batch_size=16,
        lr=5e-3,
        val_split=0.2,
    )

    assert model.input_dim == 48
    assert len(history["train_loss"]) == 20
    assert len(history["val_loss"]) == 20
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_feature_loader_rejects_empty_dataset(tmp_path: Path) -> None:
    path = tmp_path / "empty.npz"
    np.savez(path, features=np.empty((0, 48), dtype=np.float32))

    with pytest.raises(ValueError, match="empty"):
        load_feature_matrix(path)


def test_feature_loader_reads_npz_features(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    features = np.ones((3, 48), dtype=np.float32)
    np.savez(
        path,
        features=features,
        timestamp_ms=np.array([10, 20, 30], dtype=np.int64),
    )

    assert np.array_equal(load_feature_matrix(path), features)


def test_feature_loader_rejects_missing_features_array(tmp_path: Path) -> None:
    path = tmp_path / "missing_features.npz"
    np.savez(path, timestamp_ms=np.array([10], dtype=np.int64))

    with pytest.raises(ValueError, match="features"):
        load_feature_matrix(path)


def test_latent_stream_polls_estimator_and_returns_full_frame() -> None:
    class FakeEstimator:
        def __init__(self) -> None:
            self.latest = make_pose()
            self.queued = [self.latest]

        def get_latest(self):
            return self.latest

        def get_nowait(self):
            return self.queued.pop(0) if self.queued else None

    estimator = FakeEstimator()
    model = PoseAutoencoder(latent_dim=4)
    extractor = PoseFeatureExtractor(use_world_landmarks=False)
    stream = PoseLatentStream(estimator, model, extractor)

    latest = stream.get_latest()
    repeated = stream.get_latest()
    queued = stream.get_nowait()

    assert latest is repeated
    assert queued is latest
    assert latest.timestamp_ms == 10
    assert latest.feature_vector.shape == (48,)
    assert latest.latent.shape == (4,)
    assert latest.reconstruction.shape == (48,)
    assert latest.reconstruction_error >= 0
    assert latest.pose_detected is True


def test_latent_stream_from_checkpoint(tmp_path: Path) -> None:
    class FakeEstimator:
        def get_latest(self):
            return None

        def get_nowait(self):
            return None

    checkpoint = tmp_path / "pose.pt"
    save_checkpoint(checkpoint, PoseAutoencoder(input_dim=24), {})

    stream = PoseLatentStream.from_checkpoint(
        FakeEstimator(),
        checkpoint,
        include_velocity=False,
    )

    assert stream.autoencoder.input_dim == 24
    assert stream.feature_extractor.include_velocity is False
