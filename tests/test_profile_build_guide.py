from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from streaming.eeg import EegPacket
from streaming.calibration import (
    DEFAULT_CALIBRATION_BLOCKS,
    PROFILE_BUILD_BLOCKS,
    PROFILE_BUILD_REPEATS,
    PROFILE_BUILD_ROLES,
    PROFILE_MOVEMENT_TITLES,
    CalibrationMovementBlock,
    RegionScores,
    profile_build_sequence,
    select_next_movement_block,
    validate_profile_block,
)


def _load_cli_module():
    module_path = Path(__file__).resolve().parents[1] / "__main__.py"
    spec = importlib.util.spec_from_file_location(
        f"eeg_pose_cli_{uuid4().hex}",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_build_sequence_has_four_shuffled_role_chunks() -> None:
    sequence = profile_build_sequence(seed=123)
    expected_names = {block.name for block in PROFILE_BUILD_BLOCKS}

    assert PROFILE_BUILD_REPEATS == 4
    assert PROFILE_BUILD_ROLES == ("support", "query", "validation", "test")
    chunk_orders = []
    for role in PROFILE_BUILD_ROLES:
        chunk = [block for block in sequence if block.role == role]
        assert {block.name for block in chunk} == expected_names
        assert all(block.duration_s == 8.0 for block in chunk)
        assert [block.name for block in chunk if block.rest] == ["neutral_rest"]
        chunk_orders.append(tuple(block.name for block in chunk))

    assert len(set(chunk_orders)) == len(PROFILE_BUILD_ROLES)
    assert len(sequence) == 40
    assert sum(block.duration_s for block in sequence) == 320.0
    assert all(block.role in PROFILE_BUILD_ROLES for block in sequence)
    assert set(PROFILE_MOVEMENT_TITLES) == expected_names
    assert profile_build_sequence(seed=123) == sequence


def test_profile_repeat_indices_are_enclosing_round_indices() -> None:
    cli = _load_cli_module()
    sequence = profile_build_sequence(seed=123)
    role_indices = {
        role: round_index
        for round_index, role in enumerate(PROFILE_BUILD_ROLES)
    }

    assert [
        cli._repeat_index(sequence, index)
        for index in range(len(sequence))
    ] == [
        role_indices[block.role]
        for block in sequence
    ]


def test_short_profile_duration_is_rejected_before_hardware_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli_module()
    movement_duration_s = sum(
        block.duration_s
        for block in profile_build_sequence()
    )

    def unexpected_hardware_initialization():
        raise AssertionError("EEG hardware initialized before duration validation")

    monkeypatch.setattr(cli, "SignalStreamer", unexpected_hardware_initialization)

    with pytest.raises(ValueError, match="shorter than the minimum movement-and-rest"):
        cli.collect_profile_build_session(
            SimpleNamespace(duration=movement_duration_s - 0.1),
            Path("unused") / "paired_profile_session.npz",
        )


def test_profile_guide_roles_survive_block_validation_results() -> None:
    block = CalibrationMovementBlock(
        "left_arm_raise",
        "left_arm",
        8.0,
        role="validation",
    )

    result = validate_profile_block(
        block_id=3,
        block=block,
        repeat_index=2,
        start_time_s=10.0,
        end_time_s=18.0,
        feature_vectors=[],
        pose_confidences=[],
        paired_sample_count=0,
    )

    assert result.role == "validation"
    assert asdict(result)["role"] == "validation"


def test_profile_capture_drains_late_final_block_packets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli_module()

    class Clock:
        now = 100.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, _seconds: float) -> None:
            self.now += 0.25

    clock = Clock()

    def packet(packet_id: int, start_time_s: float) -> EegPacket:
        return EegPacket(
            packet_id=packet_id,
            samples=np.zeros((4, 200), dtype=np.float32),
            start_time_s=start_time_s,
            end_time_s=start_time_s + 0.796,
            sample_rate=250,
        )

    class FakeEegStream:
        def __init__(self) -> None:
            self.scheduled = [
                (101.75, packet(1, 100.8)),
                (102.0, packet(2, 101.2)),
            ]

        def pop_packet(self):
            if self.scheduled and self.scheduled[0][0] <= clock.now:
                return self.scheduled.pop(0)[1]
            return None

    class FakePoseStream:
        @staticmethod
        def get_latest():
            return SimpleNamespace(
                timestamp_ms=round(clock.now * 1000),
                received_time_s=clock.now,
                feature_vector=np.zeros(48, dtype=np.float32),
                latent=np.zeros(4, dtype=np.float32),
                reconstruction_error=0.0,
                pose_detected=True,
                confidence=0.9,
            )

    class FakeOverlay:
        @staticmethod
        def update_dummy(*_args) -> None:
            pass

        @staticmethod
        def update_truth(*_args) -> None:
            pass

        @staticmethod
        def update_neutral_rest(*_args) -> None:
            pass

        @staticmethod
        def update_movement_title(*_args, **_kwargs) -> None:
            pass

    class AlwaysReadyRestGate:
        status = None

        @staticmethod
        def begin_return_to_rest() -> None:
            pass

        @staticmethod
        def update(*_args) -> bool:
            return True

    final_block = CalibrationMovementBlock(
        "torso_side_lean",
        "shoulders_core",
        1.0,
        role="test",
    )
    monkeypatch.setattr(cli, "profile_build_sequence", lambda: [final_block])
    monkeypatch.setattr(cli.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(cli.time, "sleep", clock.sleep)
    monkeypatch.setattr(
        cli,
        "pair_packet",
        lambda packet, _pose_buffer: packet.packet_id,
    )

    frames, results = cli.collect_guided_profile_build_frames(
        FakeEegStream(),
        FakePoseStream(),
        duration_s=0.0,
        max_pose_gap_s=0.12,
        overlay_state=FakeOverlay(),
        rest_gate=AlwaysReadyRestGate(),
    )

    assert frames == [1]
    assert len(results) == 1
    assert results[0].paired_sample_count == 1


def test_active_guides_use_observable_torso_side_lean() -> None:
    assert "hip_shift" not in {block.name for block in DEFAULT_CALIBRATION_BLOCKS}
    assert "hip_shift" not in {block.name for block in PROFILE_BUILD_BLOCKS}
    assert "hip_shift" not in {block.name for block in profile_build_sequence()}
    assert "torso_shift" not in {block.name for block in DEFAULT_CALIBRATION_BLOCKS}
    assert "torso_shift" not in {block.name for block in PROFILE_BUILD_BLOCKS}
    assert "torso_shift" not in {block.name for block in profile_build_sequence()}

    block = select_next_movement_block(
        RegionScores(
            left_arm=1.0,
            right_arm=2.0,
            shoulders_core=3.0,
            hips=100.0,
            rest_false_positive=0.0,
            movement_response=0.0,
        )
    )

    assert block.name == "torso_side_lean"


@pytest.mark.parametrize("repeats", [0, 5])
def test_profile_build_sequence_rejects_unlabeled_round_counts(repeats: int) -> None:
    with pytest.raises(ValueError, match="repeats must be between 1 and 4"):
        profile_build_sequence(repeats=repeats)
