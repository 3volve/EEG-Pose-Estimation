from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from config import (
    EEG_BANDSTOP_HZ,
    EEG_BANDSTOP_ORDER,
    EEG_PREPROCESSING_VERSION,
    EEG_SAMPLE_RATE,
    EEG_SOURCE_CHANNEL_INDICES,
)
from eeg_encoding.model import (
    build_eligible_grouped_context_windows,
    train_model,
)


def _output_dir() -> Path:
    path = Path("test_outputs") / f"model_context_eligibility_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_untrusted_target_remains_in_later_eeg_context_history() -> None:
    features = np.arange(6, dtype=np.float32).reshape(6, 1, 1)
    roles = np.asarray(["support"] * 6)
    target_trusted = np.asarray([True, False, True, True, True, True])
    split_eligible = np.ones(6, dtype=bool)

    contexts, target_indices = build_eligible_grouped_context_windows(
        features,
        roles,
        target_trusted,
        split_eligible,
        context_packet_count=3,
    )

    np.testing.assert_array_equal(target_indices, [0, 2, 3, 4, 5])
    np.testing.assert_array_equal(contexts[1, :, 0, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(contexts[2, :, 0, 0], [1.0, 2.0, 3.0])


def test_split_ineligible_boundary_rows_reset_new_role_context() -> None:
    features = np.arange(8, dtype=np.float32).reshape(8, 1, 1)
    roles = np.asarray(["support"] * 3 + ["validation"] * 5)
    target_trusted = np.ones(8, dtype=bool)
    split_eligible = np.asarray(
        [True, True, True, False, False, True, True, True]
    )

    contexts, target_indices = build_eligible_grouped_context_windows(
        features,
        roles,
        target_trusted,
        split_eligible,
        context_packet_count=3,
    )

    np.testing.assert_array_equal(target_indices, [0, 1, 2, 5, 6, 7])
    np.testing.assert_array_equal(contexts[3, :, 0, 0], [5.0, 5.0, 5.0])
    np.testing.assert_array_equal(contexts[4, :, 0, 0], [5.0, 5.0, 6.0])


def test_four_role_training_uses_only_eligible_support_query_targets() -> None:
    output_dir = _output_dir()
    archive_path = output_dir / "four_role.npz"
    model_path = output_dir / "base.pt"
    rng = np.random.default_rng(27)
    roles = np.repeat(
        np.asarray(["support", "query", "validation", "test"]),
        3,
    )
    target_trusted = np.ones(12, dtype=bool)
    target_trusted[1] = False
    split_eligible = np.ones(12, dtype=bool)
    split_eligible[6] = False
    np.savez_compressed(
        archive_path,
        eeg=rng.normal(size=(12, 4, 200)).astype(np.float32),
        pose_latent=rng.normal(size=(12, 4)).astype(np.float32),
        profile_round_role=roles,
        profile_frame_trusted=target_trusted,
        profile_frame_split_eligible=split_eligible,
        pipeline_version=np.asarray(EEG_PREPROCESSING_VERSION),
        source_channel_indices=np.asarray(EEG_SOURCE_CHANNEL_INDICES),
        sample_rate_hz=np.asarray(EEG_SAMPLE_RATE),
        bandstop_low_hz=np.asarray(EEG_BANDSTOP_HZ[0]),
        bandstop_high_hz=np.asarray(EEG_BANDSTOP_HZ[1]),
        bandstop_order=np.asarray(EEG_BANDSTOP_ORDER),
    )

    predictor = train_model(
        archive_path,
        model_path,
        epochs=1,
        batch_size=3,
        hidden_dim=8,
        model_latent_dim=3,
        context_packet_count=3,
        pose_checkpoint=None,
    )

    assert predictor.training_report is not None
    assert predictor.training_report.train.n_samples == 5
    assert predictor.training_report.validation is not None
    assert predictor.training_report.validation.n_samples == 2


def test_rest_metadata_never_changes_model_inputs_or_checkpoint() -> None:
    output_dir = _output_dir()
    rng = np.random.default_rng(33)
    roles = np.repeat(
        np.asarray(["support", "query", "validation", "test"]),
        3,
    )
    shared = {
        "eeg": rng.normal(size=(12, 4, 200)).astype(np.float32),
        "pose_latent": rng.normal(size=(12, 4)).astype(np.float32),
        "profile_round_role": roles,
        "profile_frame_trusted": np.ones(12, dtype=bool),
        "profile_frame_split_eligible": np.ones(12, dtype=bool),
        "pipeline_version": np.asarray(EEG_PREPROCESSING_VERSION),
        "source_channel_indices": np.asarray(EEG_SOURCE_CHANNEL_INDICES),
        "sample_rate_hz": np.asarray(EEG_SAMPLE_RATE),
        "bandstop_low_hz": np.asarray(EEG_BANDSTOP_HZ[0]),
        "bandstop_high_hz": np.asarray(EEG_BANDSTOP_HZ[1]),
        "bandstop_order": np.asarray(EEG_BANDSTOP_ORDER),
    }
    first_archive = output_dir / "rest_false.npz"
    second_archive = output_dir / "rest_true.npz"
    np.savez_compressed(
        first_archive,
        **shared,
        profile_block_is_rest=np.zeros(12, dtype=bool),
    )
    np.savez_compressed(
        second_archive,
        **shared,
        profile_block_is_rest=np.ones(12, dtype=bool),
    )
    first_model = output_dir / "rest_false.pt"
    second_model = output_dir / "rest_true.pt"
    training_kwargs = {
        "epochs": 1,
        "batch_size": 3,
        "hidden_dim": 8,
        "model_latent_dim": 3,
        "context_packet_count": 3,
        "pose_checkpoint": None,
        "seed": 44,
    }

    train_model(first_archive, first_model, **training_kwargs)
    train_model(second_archive, second_model, **training_kwargs)

    first_state = torch.load(
        first_model,
        map_location="cpu",
        weights_only=True,
    )["state_dict"]
    second_state = torch.load(
        second_model,
        map_location="cpu",
        weights_only=True,
    )["state_dict"]
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name])
