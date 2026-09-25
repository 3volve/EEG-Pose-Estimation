from __future__ import annotations

import json
from pathlib import Path
import uuid

import numpy as np
import pytest
import torch

from eeg_encoding.permanent_holdout import (
    PERMANENT_HOLDOUT_CATEGORIES,
    HoldoutCandidate,
    candidates_from_test_blocks,
    canonical_preprocessing_signature,
    checksum_preprocessing_signature,
    load_manifest,
    normalize_holdout_category,
    replace_next_slot_after_decision,
    seed_manifest_from_first_session,
)
from eeg_encoding.model import EegPoseModelConfig, EegPoseVAE, save_model
from eeg_encoding.personalization import (
    PostAdaptationMetrics,
    ProfileBuildReport,
    _forked_torch_seed,
    _load_exact_holdout_slot,
    _write_profile_build_report,
    build_profile_model,
    profile_update_gate,
    summarize_permanent_holdout_metrics,
)


def output_dir(name: str) -> Path:
    path = Path("test_outputs") / f"permanent_holdout_{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def make_candidate(
    category: str,
    *,
    session_id: str = "session_01",
    block_id: int = 1,
    preprocessing: dict[str, object] | None = None,
    eligible: bool = True,
) -> HoldoutCandidate:
    signature = canonical_preprocessing_signature(
        preprocessing or {"context_packets": 2, "transform": "fft-v1"}
    )
    return HoldoutCandidate(
        category=category,
        source_session_id=session_id,
        source_path=f"C:/profiles/{session_id}/paired_profile_session.npz",
        source_block_id=block_id,
        source_block_name=(
            "rest_between_blocks" if category == "rest" else category
        ),
        source_repeat_index=0,
        sample_indices=(10, 11),
        sample_count=2,
        source_metadata={"profile_posture": "standing", "quality": 0.9},
        preprocessing_signature=signature,
        preprocessing_checksum=checksum_preprocessing_signature(signature),
        data_checksum=f"data-{session_id}-{block_id}",
        eligible=eligible,
        ineligible_reason="" if eligible else "quality_gate_failed",
    )


def make_complete_candidates(
    *,
    session_id: str = "session_01",
) -> list[HoldoutCandidate]:
    return [
        make_candidate(
            category,
            session_id=session_id,
            block_id=index,
        )
        for index, category in enumerate(PERMANENT_HOLDOUT_CATEGORIES)
    ]


def write_complete_holdout_archive(path: Path) -> np.ndarray:
    category_names = [
        "rest_between_blocks" if category == "rest" else category
        for category in PERMANENT_HOLDOUT_CATEGORIES
    ]
    roles = ["support", "query", "validation"]
    block_ids = [100, 101, 102]
    block_names = ["left_arm_raise"] * 3
    for index, name in enumerate(category_names):
        roles.extend(["test", "test"])
        block_ids.extend([index, index])
        block_names.extend([name, name])
    sample_count = len(roles)
    test_indices = np.flatnonzero(np.asarray(roles) == "test").astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        eeg=np.arange(sample_count * 6, dtype=np.float32).reshape(
            sample_count,
            2,
            3,
        ),
        pose_latent=np.arange(sample_count * 4, dtype=np.float32).reshape(
            sample_count,
            4,
        ),
        pose_confidence=np.full(sample_count, 0.99, dtype=np.float32),
        interpolation_confidence=np.full(sample_count, 0.99, dtype=np.float32),
        pose_reconstruction_error=np.full(sample_count, 0.01, dtype=np.float32),
        profile_session_id=np.asarray("exact_session"),
        profile_posture=np.asarray("standing"),
        profile_round_role=np.asarray(roles),
        profile_block_id=np.asarray(block_ids, dtype=np.int64),
        profile_block_name=np.asarray(block_names),
        profile_block_repeat_index=np.zeros(sample_count, dtype=np.int64),
        profile_block_accepted=np.ones(sample_count, dtype=bool),
        profile_block_is_rest=np.asarray(
            [name == "rest_between_blocks" for name in block_names],
            dtype=bool,
        ),
        profile_frame_trusted=np.ones(sample_count, dtype=bool),
        profile_frame_split_eligible=np.ones(sample_count, dtype=bool),
    )
    return test_indices


def write_legacy_checkpoint(path: Path) -> None:
    config = EegPoseModelConfig(
        n_channels=2,
        n_samples=3,
        eeg_feature_count=6,
        pose_latent_dim=4,
        context_packet_count=2,
        model_latent_dim=3,
        hidden_dim=8,
        use_wavelet=False,
        use_band_adapter=False,
    )
    save_model(path, EegPoseVAE(config))


def test_holdout_categories_normalize_two_rest_block_names() -> None:
    assert len(PERMANENT_HOLDOUT_CATEGORIES) == 10
    assert len(set(PERMANENT_HOLDOUT_CATEGORIES)) == 10
    assert normalize_holdout_category("neutral_rest") == "rest"
    assert normalize_holdout_category("rest_between_blocks") == "rest"
    assert normalize_holdout_category("hip_shift") is None


def test_candidates_from_test_blocks_keep_provenance_and_quality() -> None:
    archive_path = (
        output_dir("candidate_provenance")
        / "session_01"
        / "paired_profile_session.npz"
    )
    archive_path.parent.mkdir()
    block_summary = [
        {
            "block_id": 4,
            "movement_name": "left_arm_raise",
            "accepted": True,
            "acceptance_score": 0.95,
        },
        {
            "block_id": 5,
            "movement_name": "rest_between_blocks",
            "accepted": True,
            "acceptance_score": 0.8,
        },
        {
            "block_id": 6,
            "movement_name": "hip_shift",
            "accepted": True,
        },
    ]
    np.savez_compressed(
        archive_path,
        eeg=np.arange(6 * 2 * 3, dtype=np.float32).reshape(6, 2, 3),
        pose_latent=np.arange(6 * 4, dtype=np.float32).reshape(6, 4),
        pose_confidence=np.full(6, 0.99, dtype=np.float32),
        interpolation_confidence=np.full(6, 0.98, dtype=np.float32),
        pose_reconstruction_error=np.full(6, 0.02, dtype=np.float32),
        profile_session_id=np.asarray("session_01"),
        profile_posture=np.asarray("standing"),
        profile_block_id=np.asarray([4, 4, 5, 5, 6, 6], dtype=np.int64),
        profile_block_name=np.asarray(
            [
                "left_arm_raise",
                "left_arm_raise",
                "rest_between_blocks",
                "rest_between_blocks",
                "hip_shift",
                "hip_shift",
            ]
        ),
        profile_block_repeat_index=np.asarray([2, 2, 8, 8, 1, 1], dtype=np.int64),
        profile_block_accepted=np.ones(6, dtype=bool),
        profile_block_summary_json=np.asarray(json.dumps(block_summary)),
    )

    candidates = candidates_from_test_blocks(
        archive_path,
        [0, 1, 2],
        preprocessing={"context_packets": 2, "transform": "fft-v1"},
    )

    assert [candidate.category for candidate in candidates] == [
        "left_arm_raise",
        "rest",
    ]
    movement, rest = candidates
    assert movement.eligible
    assert movement.source_session_id == "session_01"
    assert movement.source_path == str(archive_path.resolve())
    assert movement.source_block_id == 4
    assert movement.source_block_name == "left_arm_raise"
    assert movement.source_repeat_index == 2
    assert movement.sample_indices == (0, 1)
    assert movement.source_metadata["profile_posture"] == "standing"
    assert movement.source_metadata["block_summary"]["acceptance_score"] == 0.95
    assert len(movement.preprocessing_checksum) == 64
    assert len(movement.data_checksum) == 64
    assert not rest.eligible
    assert rest.ineligible_reason == "partial_test_block"


def test_seed_manifest_has_ten_slots_and_persists_cursor() -> None:
    path = output_dir("seed") / "permanent_holdout_manifest.json"
    candidates = make_complete_candidates()

    manifest = seed_manifest_from_first_session(path, candidates)
    loaded = load_manifest(path)

    assert manifest == loaded
    assert loaded.cursor == 0
    assert loaded.next_category == "left_arm_raise"
    assert tuple(loaded.slots) == PERMANENT_HOLDOUT_CATEGORIES
    assert all(slot is not None for slot in loaded.slots.values())
    assert (
        loaded.slots["rest"].source_block_name
        == "rest_between_blocks"
    )


def test_replace_advances_only_after_eligible_matching_candidate(
) -> None:
    path = output_dir("replace") / "permanent_holdout_manifest.json"
    seed_manifest_from_first_session(
        path,
        make_complete_candidates(session_id="first"),
    )

    before_absent = path.read_bytes()
    absent = replace_next_slot_after_decision(
        path,
        [make_candidate("right_arm_raise", session_id="second", block_id=2)],
        decision_id="build_02",
        committed=True,
    )
    assert not absent.updated
    assert absent.reason == "candidate_absent"
    assert absent.cursor_before == absent.cursor_after == 0
    assert path.read_bytes() == before_absent

    ineligible_candidate = make_candidate(
        "left_arm_raise",
        session_id="second",
        block_id=3,
        eligible=False,
    )
    ineligible = replace_next_slot_after_decision(
        path,
        [ineligible_candidate],
        decision_id="build_02",
        committed=True,
    )
    assert not ineligible.updated
    assert ineligible.reason == "candidate_ineligible"
    assert load_manifest(path).cursor == 0

    new_left = make_candidate(
        "left_arm_raise",
        session_id="second",
        block_id=4,
    )
    replaced = replace_next_slot_after_decision(
        path,
        [new_left],
        decision_id="build_02",
        committed=False,
    )
    assert replaced.updated
    assert replaced.reason == "replaced"
    assert replaced.cursor_before == 0
    assert replaced.cursor_after == 1

    loaded = load_manifest(path)
    slot = loaded.slots["left_arm_raise"]
    assert slot is not None
    assert slot.source_session_id == "second"
    assert slot.source_block_id == 4
    assert slot.installed_after_decision_id == "build_02"
    assert slot.installed_after_committed is False
    assert loaded.next_category == "right_arm_raise"


def test_preprocessing_mismatch_does_not_advance_cursor() -> None:
    path = output_dir("mismatch") / "permanent_holdout_manifest.json"
    seed_manifest_from_first_session(
        path,
        make_complete_candidates(),
    )
    different_preprocessing = make_candidate(
        "left_arm_raise",
        session_id="second",
        block_id=2,
        preprocessing={"context_packets": 3, "transform": "fft-v1"},
    )

    update = replace_next_slot_after_decision(
        path,
        [different_preprocessing],
        decision_id="build_02",
        committed=True,
    )

    assert not update.updated
    assert update.reason == "preprocessing_mismatch"
    assert load_manifest(path).cursor == 0


def test_manifest_rejects_tampered_preprocessing_checksum() -> None:
    path = output_dir("tamper") / "permanent_holdout_manifest.json"
    seed_manifest_from_first_session(
        path,
        make_complete_candidates(),
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["preprocessing_checksum"] = "not-the-signature-checksum"
    path.write_text(json.dumps(raw), encoding="utf-8")

    try:
        load_manifest(path)
    except ValueError as error:
        assert "preprocessing checksum is invalid" in str(error)
    else:
        raise AssertionError("tampered manifest should be rejected")


def test_seed_rejects_incomplete_first_session() -> None:
    path = output_dir("incomplete") / "permanent_holdout_manifest.json"

    with pytest.raises(ValueError, match="missing"):
        seed_manifest_from_first_session(
            path,
            [make_candidate("left_arm_raise")],
        )

    assert not path.exists()


def test_rotation_rejects_session_already_used_by_another_category() -> None:
    path = output_dir("diversity") / "permanent_holdout_manifest.json"
    seed_manifest_from_first_session(
        path,
        make_complete_candidates(session_id="bootstrap"),
    )
    first = replace_next_slot_after_decision(
        path,
        [
            make_candidate(
                "left_arm_raise",
                session_id="daily_01",
                block_id=100,
            )
        ],
        decision_id="build_01",
        committed=True,
    )
    assert first.updated
    assert first.cursor_after == 1

    repeated_session = replace_next_slot_after_decision(
        path,
        [
            make_candidate(
                "right_arm_raise",
                session_id="daily_01",
                block_id=101,
            )
        ],
        decision_id="build_02",
        committed=True,
    )
    assert not repeated_session.updated
    assert repeated_session.reason == "source_session_already_used"
    assert repeated_session.cursor_before == repeated_session.cursor_after == 1

    distinct_session = replace_next_slot_after_decision(
        path,
        [
            make_candidate(
                "right_arm_raise",
                session_id="daily_02",
                block_id=102,
            )
        ],
        decision_id="build_03",
        committed=True,
    )
    assert distinct_session.updated
    assert distinct_session.cursor_after == 2


def test_ten_replacements_establish_and_preserve_distinct_session_coverage() -> None:
    path = output_dir("full_diversity") / "permanent_holdout_manifest.json"
    seed_manifest_from_first_session(
        path,
        make_complete_candidates(session_id="bootstrap"),
    )

    for index, category in enumerate(PERMANENT_HOLDOUT_CATEGORIES):
        update = replace_next_slot_after_decision(
            path,
            [
                make_candidate(
                    category,
                    session_id=f"daily_{index:02d}",
                    block_id=100 + index,
                )
            ],
            decision_id=f"build_{index:02d}",
            committed=True,
        )
        assert update.updated

    manifest = load_manifest(path)
    session_ids = {
        slot.source_session_id
        for slot in manifest.slots.values()
        if slot is not None
    }
    assert len(session_ids) == 10
    assert "bootstrap" not in session_ids

    next_update = replace_next_slot_after_decision(
        path,
        [
            make_candidate(
                PERMANENT_HOLDOUT_CATEGORIES[0],
                session_id="daily_10",
                block_id=200,
            )
        ],
        decision_id="build_10",
        committed=False,
    )
    assert next_update.updated
    rotated = load_manifest(path)
    assert len(
        {
            slot.source_session_id
            for slot in rotated.slots.values()
            if slot is not None
        }
    ) == 10


def test_holdout_aggregation_uses_rest_and_movement_roles_separately() -> None:
    path = output_dir("aggregation") / "manifest.json"
    manifest = seed_manifest_from_first_session(
        path,
        make_complete_candidates(),
    )
    metrics = [
        PostAdaptationMetrics(
            session=category,
            query_samples=2,
            pose_mae=float(index),
            decoded_pose_error=float(index + 1),
            stationary_false_positive_score=float(index + 10),
            movement_response_score=float(index + 20),
            readiness_score=float(index + 30),
            ready=False,
        )
        for index, category in enumerate(PERMANENT_HOLDOUT_CATEGORIES)
    ]

    summary = summarize_permanent_holdout_metrics(metrics, manifest)

    assert summary["pose_mae"] == np.mean(np.arange(10))
    assert summary["decoded_pose_error"] == np.mean(np.arange(1, 11))
    assert summary["stationary_false_positive_score"] == 19.0
    assert summary["movement_response_score"] == np.mean(np.arange(20, 29))
    assert summary["readiness_score"] == np.mean(np.arange(30, 40))


def test_profile_gate_fails_closed_when_holdout_is_expected() -> None:
    unchanged = {
        "decoded_pose_error": 0.2,
        "stationary_false_positive_score": 0.1,
        "movement_response_score": 0.7,
        "readiness_score": 0.8,
    }

    committed, validation_passed, holdout_passed = profile_update_gate(
        unchanged,
        unchanged,
        None,
        None,
        holdout_expected=True,
    )

    assert validation_passed
    assert not holdout_passed
    assert not committed


def test_profile_report_serializes_holdout_and_test_evidence() -> None:
    output = output_dir("report")
    metric = PostAdaptationMetrics(
        session="session:test",
        query_samples=2,
        pose_mae=0.1,
        decoded_pose_error=0.2,
        stationary_false_positive_score=0.03,
        movement_response_score=0.7,
        readiness_score=0.8,
        ready=False,
    )
    report = ProfileBuildReport(
        user_id="Evo",
        profile_model="profile.pt",
        proposed_model="proposed.pt",
        committed=False,
        start_checkpoint="start.pt",
        history_dir=str(output),
        session_count=1,
        old_metrics={"decoded_pose_error": 0.2},
        new_metrics={"decoded_pose_error": 0.3},
        old_session_metrics=(metric,),
        session_metrics=(metric,),
        session_block_summaries={},
        validation_passed=False,
        holdout_passed=True,
        old_holdout_metrics={"decoded_pose_error": 0.2},
        new_holdout_metrics={"decoded_pose_error": 0.19},
        test_metrics=(metric,),
        holdout_update={"updated": True, "category": "left_arm_raise"},
    )

    _write_profile_build_report(output, report)

    payload = json.loads(
        (output / "profile_build_report.json").read_text(encoding="utf-8")
    )
    assert payload["validation_passed"] is False
    assert payload["holdout_passed"] is True
    assert payload["old_holdout_metrics"]["decoded_pose_error"] == 0.2
    assert payload["new_holdout_metrics"]["decoded_pose_error"] == 0.19
    assert payload["test_metrics"][0]["session"] == "session:test"
    assert payload["holdout_update"]["category"] == "left_arm_raise"


def test_exact_slot_loader_uses_manifest_indices_and_checks_source() -> None:
    output = output_dir("exact_source")
    archive_path = output / "paired_profile_session.npz"
    test_indices = write_complete_holdout_archive(archive_path)
    with np.load(archive_path) as archive:
        initial = {key: np.asarray(archive[key]) for key in archive.files}
    initial["profile_frame_trusted"] = initial["profile_frame_trusted"].copy()
    initial["profile_frame_trusted"][test_indices[0]] = False
    np.savez_compressed(archive_path, **initial)
    legacy_preprocessing = {
        "pipeline_version": "legacy-first-four-unfiltered",
        "source_channel_indices": (0, 1, 2, 3),
        "sample_rate_hz": 250,
        "bandstop_low_hz": None,
        "bandstop_high_hz": None,
        "bandstop_order": None,
    }
    candidates = candidates_from_test_blocks(
        archive_path,
        test_indices,
        preprocessing=legacy_preprocessing,
    )
    manifest_path = output / "manifest.json"
    manifest = seed_manifest_from_first_session(manifest_path, candidates)
    slot = manifest.slots["left_arm_raise"]
    assert slot is not None

    session, evaluation_indices = _load_exact_holdout_slot(slot, manifest)

    assert slot.sample_count == 2
    assert len(session.eeg) == len(initial["eeg"])
    assert len(evaluation_indices) == 1
    assert evaluation_indices[0] == slot.sample_indices[1]
    np.testing.assert_array_equal(
        session.block_ids[evaluation_indices],
        np.full(1, slot.source_block_id),
    )

    with np.load(archive_path) as archive:
        changed = {key: np.asarray(archive[key]) for key in archive.files}
    changed["eeg"] = changed["eeg"].copy()
    changed["eeg"][slot.sample_indices[0], 0, 0] += 1.0
    np.savez_compressed(archive_path, **changed)

    with pytest.raises(ValueError, match="source checksum changed"):
        _load_exact_holdout_slot(slot, manifest)


def test_candidates_reject_ambiguous_multi_source_archive() -> None:
    output = output_dir("multi_source")
    archive_path = output / "paired_profile_session.npz"
    test_indices = write_complete_holdout_archive(archive_path)
    (output / "source_00.npz").write_bytes(archive_path.read_bytes())
    (output / "source_01.npz").write_bytes(archive_path.read_bytes())

    with pytest.raises(ValueError, match="merges 2 sources"):
        candidates_from_test_blocks(
            archive_path,
            test_indices,
            preprocessing={"pipeline": "test"},
        )


def test_forked_adaptation_seed_is_repeatable_and_restores_rng() -> None:
    torch.manual_seed(91)
    expected_after = torch.rand(3)
    torch.manual_seed(91)

    with _forked_torch_seed(12):
        first = torch.rand(4)
    with _forked_torch_seed(12):
        second = torch.rand(4)
    actual_after = torch.rand(3)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(actual_after, expected_after)


def test_four_round_profile_requires_bootstrap_before_persisting_history() -> None:
    output = output_dir("bootstrap_required")
    archive_path = output / "session.npz"
    write_complete_holdout_archive(archive_path)
    checkpoint = output / "base.pt"
    write_legacy_checkpoint(checkpoint)
    profiles_root = output / "profiles"

    with pytest.raises(RuntimeError, match="run bootstrap-base first"):
        build_profile_model(
            user_id="user",
            new_session_data=archive_path,
            base_checkpoint=checkpoint,
            profiles_root=profiles_root,
            epochs=0,
            inner_epochs=0,
            query_epochs=0,
            pose_checkpoint=None,
        )

    assert not (profiles_root / "user" / "profile_history").exists()


def test_preprocessing_mismatch_does_not_poison_profile_history() -> None:
    output = output_dir("staged_validation")
    archive_path = output / "session.npz"
    write_complete_holdout_archive(archive_path)
    with np.load(archive_path) as archive:
        changed = {key: np.asarray(archive[key]) for key in archive.files}
    changed.update(
        {
            "pipeline_version": np.asarray("different-pipeline"),
            "source_channel_indices": np.asarray([0, 1]),
            "sample_rate_hz": np.asarray(250),
            "bandstop_low_hz": np.asarray(55.0),
            "bandstop_high_hz": np.asarray(65.0),
            "bandstop_order": np.asarray(4),
        }
    )
    np.savez_compressed(archive_path, **changed)
    checkpoint = output / "base.pt"
    write_legacy_checkpoint(checkpoint)
    profiles_root = output / "profiles"

    with pytest.raises(ValueError, match="preprocessing mismatch"):
        build_profile_model(
            user_id="user",
            new_session_data=archive_path,
            base_checkpoint=checkpoint,
            profiles_root=profiles_root,
            epochs=0,
            inner_epochs=0,
            query_epochs=0,
            pose_checkpoint=None,
        )

    assert not (profiles_root / "user" / "profile_history").exists()


def test_holdout_preprocessing_mismatch_does_not_poison_profile_history(
) -> None:
    output = output_dir("holdout_prevalidation")
    archive_path = output / "session.npz"
    write_complete_holdout_archive(archive_path)
    checkpoint = output / "base.pt"
    write_legacy_checkpoint(checkpoint)
    profiles_root = output / "profiles"
    seed_manifest_from_first_session(
        profiles_root / "user" / "permanent_holdout" / "manifest.json",
        make_complete_candidates(),
    )

    with pytest.raises(
        ValueError,
        match="Permanent holdout preprocessing does not match",
    ):
        build_profile_model(
            user_id="user",
            new_session_data=archive_path,
            base_checkpoint=checkpoint,
            profiles_root=profiles_root,
            epochs=0,
            inner_epochs=0,
            query_epochs=0,
            pose_checkpoint=None,
        )

    assert not (profiles_root / "user" / "profile_history").exists()
