from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from streaming import CalibrationMovementBlock
from streaming.eeg import EegPacket


def _load_cli_module():
    module_path = Path(__file__).resolve().parents[1] / "__main__.py"
    spec = importlib.util.spec_from_file_location(
        f"eeg_pose_guide_flow_{uuid4().hex}",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_capture_excludes_neutral_wait_and_boundary_packets() -> None:
    cli = _load_cli_module()

    class Clock:
        now = 100.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, _seconds: float) -> None:
            self.now += 0.25

    clock = Clock()

    def packet(
        packet_id: int,
        start_time_s: float,
        end_time_s: float,
    ) -> EegPacket:
        return EegPacket(
            packet_id=packet_id,
            samples=np.zeros((4, 200), dtype=np.float32),
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            sample_rate=250,
        )

    class FakeEegStream:
        def __init__(self) -> None:
            self.scheduled = [
                (101.75, packet(1, 100.8, 101.5)),
                (101.75, packet(2, 100.5, 101.2)),
                (102.75, packet(3, 101.9, 102.6)),
                (104.5, packet(4, 103.6, 104.3)),
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

    class DelayedRestGate:
        status = None

        def __init__(self) -> None:
            self.ready_at: float | None = None

        def begin_return_to_rest(self) -> None:
            self.ready_at = clock.now + 1.0

        def update(self, _feature_vector, now_s: float) -> bool:
            return self.ready_at is None or now_s >= self.ready_at

    sequence = (
        CalibrationMovementBlock(
            "left_arm_raise",
            "left_arm",
            1.0,
            role="support",
        ),
        CalibrationMovementBlock(
            "right_arm_raise",
            "right_arm",
            1.0,
            role="query",
        ),
    )

    with (
        patch.object(cli.time, "monotonic", clock.monotonic),
        patch.object(cli.time, "sleep", clock.sleep),
        patch.object(cli, "pair_packet", lambda packet, _buffer: packet.packet_id),
    ):
        frames, results = cli.collect_guided_profile_build_frames(
            FakeEegStream(),
            FakePoseStream(),
            duration_s=0.0,
            max_pose_gap_s=0.12,
            overlay_state=FakeOverlay(),
            sequence=sequence,
            rest_gate=DelayedRestGate(),
        )

    assert frames == [1, 4]
    assert [result.paired_sample_count for result in results] == [1, 1]
    assert results[0].start_time_s == 100.75
    assert results[0].end_time_s <= results[1].start_time_s - 1.75
