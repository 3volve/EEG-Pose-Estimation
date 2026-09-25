from __future__ import annotations

import sys
import threading
from types import ModuleType

import numpy as np
import pytest

from config import (
    EEG_BANDSTOP_HZ,
    EEG_BANDSTOP_ORDER,
    EEG_PREPROCESSING_VERSION,
    EEG_SAMPLE_RATE,
    EEG_SOURCE_CHANNEL_INDICES,
)
from streaming.eeg import SignalStreamer


class _FakeEegStream:
    def __init__(self, *, channel_count: int = 8, sample_rate: float = 250.0):
        self._channel_count = channel_count
        self._sample_rate = sample_rate

    def type(self) -> str:
        return "EEG"

    def name(self) -> str:
        return "test-eeg"

    def channel_count(self) -> int:
        return self._channel_count

    def nominal_srate(self) -> float:
        return self._sample_rate


def _install_fake_pylsl(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stream: _FakeEegStream,
    inlet_factory,
) -> None:
    module = ModuleType("pylsl")
    module.resolve_streams = lambda: [stream]
    module.StreamInlet = inlet_factory
    module.local_clock = lambda: 100.0
    module.proc_clocksync = 1
    module.proc_dejitter = 2
    module.proc_monotonize = 4
    monkeypatch.setitem(sys.modules, "pylsl", module)


def _eight_column_signal(
    selected_signal: np.ndarray,
) -> np.ndarray:
    sample_count = len(selected_signal)
    raw = np.zeros((sample_count, 8), dtype=np.float32)
    for channel_index in EEG_SOURCE_CHANNEL_INDICES:
        raw[:, channel_index] = selected_signal * channel_index
    return raw


def test_default_preprocessing_signature_describes_corrected_ingress() -> None:
    streamer = SignalStreamer()

    assert streamer.channel_indices == (1, 2, 3, 4)
    assert streamer.n_channels == 4
    assert streamer.preprocessing_signature == {
        "pipeline_version": EEG_PREPROCESSING_VERSION,
        "source_channel_indices": EEG_SOURCE_CHANNEL_INDICES,
        "sample_rate_hz": EEG_SAMPLE_RATE,
        "bandstop_low_hz": EEG_BANDSTOP_HZ[0],
        "bandstop_high_hz": EEG_BANDSTOP_HZ[1],
        "bandstop_order": EEG_BANDSTOP_ORDER,
    }


def test_channel_selection_uses_exact_configured_columns() -> None:
    streamer = SignalStreamer(
        packet_size=3,
        packet_stride=3,
        channel_indices=(1, 4),
        bandstop_hz=None,
    )
    raw = np.arange(15, dtype=np.float32).reshape(3, 5)

    streamer._append_samples(raw)
    packet = streamer.pop_packet()

    assert packet is not None
    np.testing.assert_array_equal(packet.samples, raw[:, (1, 4)].T)


def test_received_chunk_rejects_missing_configured_column() -> None:
    streamer = SignalStreamer(
        channel_indices=(1, 4),
        bandstop_hz=None,
    )

    with pytest.raises(ValueError, match=r"has 4 columns.*require column 4"):
        streamer._append_samples(np.zeros((10, 4), dtype=np.float32))


def test_lsl_metadata_rejects_wrong_rate_and_insufficient_width() -> None:
    streamer = SignalStreamer()

    with pytest.raises(RuntimeError, match=r"249 Hz; expected 250 Hz"):
        streamer._validate_stream_metadata(
            stream_name="test-eeg",
            channel_count=8,
            nominal_sample_rate=249.0,
        )
    with pytest.raises(ValueError, match=r"has 4 columns.*require column 4"):
        streamer._validate_stream_metadata(
            stream_name="test-eeg",
            channel_count=4,
            nominal_sample_rate=250.0,
        )


def test_lsl_metadata_failure_is_exposed_to_packet_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamer = SignalStreamer()
    _install_fake_pylsl(
        monkeypatch,
        stream=_FakeEegStream(channel_count=4),
        inlet_factory=lambda _stream: pytest.fail(
            "metadata validation should fail before opening an inlet"
        ),
    )

    with pytest.raises(ValueError, match=r"has 4 columns.*require column 4") as direct:
        streamer.start_streaming()
    with pytest.raises(RuntimeError, match="fix the stream and restart") as surfaced:
        streamer.pop_packet()

    assert surfaced.value.__cause__ is direct.value


def test_pull_processing_failure_crosses_worker_thread_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MalformedInlet:
        def __init__(self, _stream, **kwargs) -> None:
            pass

        def flush(self) -> None:
            pass

        def open_stream(self, timeout):
            pass

        def time_correction(self, timeout):
            return 0.0

        def close_stream(self):
            pass

        def pull_chunk(self, _timeout_s, _packet_stride):
            return [[0.0, 1.0, 2.0, 3.0]], [0.0]

    streamer = SignalStreamer()
    _install_fake_pylsl(
        monkeypatch,
        stream=_FakeEegStream(),
        inlet_factory=MalformedInlet,
    )
    worker_errors: list[Exception] = []

    def run_worker() -> None:
        try:
            streamer.start_streaming()
        except Exception as error:
            worker_errors.append(error)

    worker = threading.Thread(target=run_worker)
    worker.start()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert len(worker_errors) == 1
    with pytest.raises(RuntimeError, match=r"received EEG LSL chunk has 4 columns") as surfaced:
        streamer.pop_packet()
    assert surfaced.value.__cause__ is worker_errors[0]


def test_bandstop_attenuates_60_hz_and_preserves_10_hz() -> None:
    sample_count = EEG_SAMPLE_RATE * 10
    time_s = np.arange(sample_count, dtype=np.float64) / EEG_SAMPLE_RATE
    selected = (
        np.sin(2.0 * np.pi * 10.0 * time_s)
        + np.sin(2.0 * np.pi * 60.0 * time_s)
    ).astype(np.float32)
    raw = _eight_column_signal(selected)
    streamer = SignalStreamer(
        packet_size=sample_count,
        packet_stride=sample_count,
        channel_indices=(1,),
    )

    streamer._append_samples(raw)
    packet = streamer.pop_packet()

    assert packet is not None
    output = packet.samples[0]
    frequencies = np.fft.rfftfreq(sample_count, d=1.0 / EEG_SAMPLE_RATE)
    input_spectrum = np.abs(np.fft.rfft(raw[:, 1]))
    output_spectrum = np.abs(np.fft.rfft(output))

    def response_db(frequency_hz: float) -> float:
        index = int(np.argmin(np.abs(frequencies - frequency_hz)))
        return float(20.0 * np.log10(output_spectrum[index] / input_spectrum[index]))

    assert response_db(60.0) < -20.0
    assert abs(response_db(10.0)) < 1.0


def test_bandstop_state_is_chunk_invariant_and_packet_overlap_is_exact() -> None:
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(500, 8)).astype(np.float32)
    single_chunk = SignalStreamer(packet_size=500, packet_stride=500)
    irregular_chunks = SignalStreamer(packet_size=500, packet_stride=500)

    single_chunk._append_samples(raw)
    for start, end in ((0, 17), (17, 138), (138, 139), (139, 411), (411, 500)):
        irregular_chunks._append_samples(raw[start:end])

    single_packet = single_chunk.pop_packet()
    chunked_packet = irregular_chunks.pop_packet()
    assert single_packet is not None
    assert chunked_packet is not None
    np.testing.assert_allclose(
        chunked_packet.samples,
        single_packet.samples,
        rtol=0.0,
        atol=1e-6,
    )

    overlapping = SignalStreamer(packet_size=200, packet_stride=50)
    overlapping._append_samples(raw[:300])
    first = overlapping.pop_packet()
    second = overlapping.pop_packet()
    third = overlapping.pop_packet()

    assert first is not None
    assert second is not None
    assert third is not None
    np.testing.assert_array_equal(first.samples[:, 50:], second.samples[:, :150])
    np.testing.assert_array_equal(second.samples[:, 50:], third.samples[:, :150])
