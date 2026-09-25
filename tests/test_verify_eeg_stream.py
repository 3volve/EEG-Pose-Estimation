from __future__ import annotations

import importlib.util
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from streaming.eeg import EegPacket


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


class FakeEegStream:
    preprocessing_signature = {
        "pipeline_version": "test-filtered-v1",
        "source_channel_indices": (1, 2),
        "sample_rate_hz": 250,
        "bandstop_low_hz": 55.0,
        "bandstop_high_hz": 65.0,
        "bandstop_order": 4,
    }

    def __init__(self, packets: list[EegPacket]) -> None:
        self.packets = list(packets)
        self.started = False
        self.stopped = False

    def start_streaming(self) -> None:
        self.started = True

    def stop_streaming(self) -> None:
        self.stopped = True

    def pop_packet(self) -> EegPacket | None:
        return self.packets.pop(0) if self.packets else None


def _packet(packet_id: int, samples: list[list[float]]) -> EegPacket:
    return EegPacket(
        packet_id=packet_id,
        samples=np.asarray(samples, dtype=np.float32),
        start_time_s=float(packet_id),
        end_time_s=float(packet_id) + 0.8,
        sample_rate=250,
    )


def test_eeg_stream_verification_summarizes_fake_packets() -> None:
    cli = _load_cli_module()
    stream = FakeEegStream(
        [
            _packet(0, [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]),
            _packet(1, [[2.0, 3.0, 4.0], [12.0, 13.0, 14.0]]),
        ]
    )

    summary = cli._collect_eeg_stream_verification(
        stream,
        duration_s=0.01,
        poll_delay_s=0.0,
    )

    assert stream.started
    assert stream.stopped
    assert summary["preprocessing_signature"] == stream.preprocessing_signature
    assert summary["packet_count"] == 2
    assert summary["packet_shape"] == [2, 3]
    np.testing.assert_allclose(
        summary["channel_standard_deviations"],
        np.std(
            np.asarray(
                [
                    [0.0, 1.0, 2.0, 2.0, 3.0, 4.0],
                    [10.0, 11.0, 12.0, 12.0, 13.0, 14.0],
                ]
            ),
            axis=1,
        ),
    )
    assert np.all(np.isfinite(summary["channel_standard_deviations"]))


def test_eeg_stream_verification_fails_clearly_without_packets() -> None:
    cli = _load_cli_module()
    stream = FakeEegStream([])

    with pytest.raises(RuntimeError, match="No EEG packet arrived"):
        cli._collect_eeg_stream_verification(
            stream,
            duration_s=0.001,
            poll_delay_s=0.0,
        )

    assert stream.started
    assert stream.stopped
