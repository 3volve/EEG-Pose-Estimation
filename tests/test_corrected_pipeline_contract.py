from pathlib import Path
from uuid import uuid4

import numpy as np

from config import (
    EEG_BANDSTOP_HZ,
    EEG_BANDSTOP_ORDER,
    EEG_PREPROCESSING_VERSION,
    EEG_SAMPLE_RATE,
    EEG_SOURCE_CHANNEL_INDICES,
)
from eeg_encoding.model import (
    EegPoseModelConfig,
    EegPoseVAE,
    build_grouped_context_windows,
    load_model,
    train_model,
)
from eeg_encoding.personalization import (
    _session_eeg_contexts,
    load_profile_session,
    profile_update_gate,
)


def _write_four_round_archive(path: Path) -> None:
    rng = np.random.default_rng(7)
    sample_count = 16
    roles = np.repeat(
        np.asarray(["support", "query", "validation", "test"]),
        4,
    )
    block_ids = np.repeat(np.arange(8, dtype=np.int64), 2)
    block_names = np.asarray(
        [
            "left_arm_raise",
            "left_arm_raise",
            "rest_between_blocks",
            "rest_between_blocks",
        ]
        * 4
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        eeg=rng.normal(size=(sample_count, 4, 200)).astype(np.float32),
        pose_latent=rng.normal(size=(sample_count, 6)).astype(np.float32),
        pose_confidence=np.full(sample_count, 0.9, dtype=np.float32),
        interpolation_confidence=np.full(sample_count, 0.9, dtype=np.float32),
        pose_reconstruction_error=np.full(sample_count, 0.01, dtype=np.float32),
        profile_round_role=roles,
        profile_block_id=block_ids,
        profile_block_name=block_names,
        profile_block_repeat_index=np.repeat(np.arange(4), 4),
        profile_block_accepted=np.asarray(
            [True, True, False, False] + [True] * 12,
            dtype=np.bool_,
        ),
        profile_block_is_rest=np.asarray(
            ["rest" in name for name in block_names],
            dtype=np.bool_,
        ),
        profile_frame_trusted=np.ones(sample_count, dtype=np.bool_),
        profile_frame_split_eligible=np.ones(sample_count, dtype=np.bool_),
        pipeline_version=np.asarray(EEG_PREPROCESSING_VERSION),
        source_channel_indices=np.asarray(EEG_SOURCE_CHANNEL_INDICES),
        sample_rate_hz=np.asarray(EEG_SAMPLE_RATE),
        bandstop_low_hz=np.asarray(EEG_BANDSTOP_HZ[0]),
        bandstop_high_hz=np.asarray(EEG_BANDSTOP_HZ[1]),
        bandstop_order=np.asarray(EEG_BANDSTOP_ORDER),
    )


def _output_dir() -> Path:
    path = (
        Path(__file__).resolve().parents[1]
        / "test_outputs"
        / f"corrected_pipeline_{uuid4().hex}"
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_four_round_base_training_excludes_test_and_saves_signature(
) -> None:
    tmp_path = _output_dir()
    archive_path = tmp_path / "four_round.npz"
    model_path = tmp_path / "base.pt"
    _write_four_round_archive(archive_path)

    predictor = train_model(
        archive_path,
        model_path,
        epochs=1,
        batch_size=4,
        hidden_dim=12,
        model_latent_dim=4,
        pose_checkpoint=None,
    )

    assert predictor.training_report is not None
    assert predictor.training_report.train.n_samples == 8
    assert predictor.training_report.validation is not None
    assert predictor.training_report.validation.n_samples == 4
    reloaded = load_model(model_path)
    assert reloaded.model.config.pipeline_version == EEG_PREPROCESSING_VERSION
    assert (
        reloaded.model.config.source_channel_indices
        == EEG_SOURCE_CHANNEL_INDICES
    )
    assert reloaded.model.config.bandstop_low_hz == EEG_BANDSTOP_HZ[0]
    assert reloaded.model.config.bandstop_high_hz == EEG_BANDSTOP_HZ[1]


def test_new_profile_loader_keeps_diagnostically_rejected_frames(
) -> None:
    tmp_path = _output_dir()
    archive_path = tmp_path / "four_round.npz"
    _write_four_round_archive(archive_path)

    session = load_profile_session(archive_path)

    assert len(session.eeg) == 16
    assert len(session.support_indices) == 4
    assert len(session.query_indices) == 4
    assert len(session.validation_indices) == 4
    assert len(session.test_indices) == 4
    assert session.block_accepted is not None
    assert not np.all(session.block_accepted)


def test_profile_contexts_keep_untrusted_history_and_break_at_split_boundary(
) -> None:
    tmp_path = _output_dir()
    archive_path = tmp_path / "four_round.npz"
    _write_four_round_archive(archive_path)
    with np.load(archive_path) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    arrays["eeg"] = np.broadcast_to(
        np.arange(16, dtype=np.float32)[:, None, None],
        (16, 4, 200),
    ).copy()
    arrays["profile_frame_trusted"] = np.ones(16, dtype=bool)
    arrays["profile_frame_trusted"][1] = False
    arrays["profile_frame_split_eligible"] = np.ones(16, dtype=bool)
    arrays["profile_frame_split_eligible"][5] = False
    np.savez_compressed(archive_path, **arrays)

    session = load_profile_session(archive_path)

    assert len(session.eeg) == 16
    np.testing.assert_array_equal(
        session.target_eligible[:8],
        [True, False, True, True, True, False, True, True],
    )
    np.testing.assert_array_equal(
        session.history_eligible[:8],
        [True, True, True, True, True, False, True, True],
    )
    np.testing.assert_array_equal(session.support_indices, [0, 2, 3])
    np.testing.assert_array_equal(session.query_indices, [4, 6, 7])

    config = EegPoseModelConfig(
        n_channels=4,
        n_samples=200,
        eeg_feature_count=200,
        pose_latent_dim=6,
        context_packet_count=2,
        model_latent_dim=3,
        hidden_dim=8,
        use_wavelet=False,
        standardize_input=False,
        use_band_adapter=False,
    )
    contexts, target_indices = _session_eeg_contexts(
        EegPoseVAE(config),
        session,
    )
    context_by_raw_index = {
        int(raw_index): contexts[position]
        for position, raw_index in enumerate(target_indices)
    }

    np.testing.assert_array_equal(
        context_by_raw_index[2][:, 0, 0],
        [1.0, 2.0],
    )
    np.testing.assert_array_equal(
        context_by_raw_index[6][:, 0, 0],
        [6.0, 6.0],
    )


def test_context_history_resets_only_when_round_role_changes() -> None:
    features = np.arange(6, dtype=np.float32).reshape(6, 1, 1)
    roles = np.asarray(["support"] * 3 + ["test"] * 3)

    contexts = build_grouped_context_windows(
        features,
        roles,
        context_packet_count=2,
    )

    np.testing.assert_array_equal(contexts[2, :, 0, 0], [1.0, 2.0])
    np.testing.assert_array_equal(contexts[3, :, 0, 0], [3.0, 3.0])


def test_permanent_holdout_can_veto_validation_approved_update() -> None:
    old = {
        "decoded_pose_error": 0.30,
        "stationary_false_positive_score": 0.20,
        "movement_response_score": 0.70,
        "readiness_score": 0.80,
    }
    improved = {
        "decoded_pose_error": 0.25,
        "stationary_false_positive_score": 0.18,
        "movement_response_score": 0.75,
        "readiness_score": 0.85,
    }
    regressed_holdout = {
        **old,
        "decoded_pose_error": 0.35,
    }

    committed, validation_passed, holdout_passed = profile_update_gate(
        old,
        improved,
        old,
        regressed_holdout,
    )

    assert validation_passed
    assert not holdout_passed
    assert not committed
