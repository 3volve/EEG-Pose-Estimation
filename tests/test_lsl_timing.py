from types import ModuleType
import time
import json
import sys

import numpy as np
import pytest

from streaming.eeg import SignalStreamer
from streaming.debug_capture import RawPoseEegDebugCapture


class Stream:
    def type(self): return "EEG"
    def name(self): return "test"
    def channel_count(self): return 2
    def nominal_srate(self): return 250.0


def test_live_clock_mapping_raw_diagnostics_and_read_sizes(monkeypatch, tmp_path):
    capture = RawPoseEegDebugCapture()
    streamer = SignalStreamer(
        channel_indices=(0, 1), bandstop_hz=None, packet_size=8, packet_stride=2,
        corrected_sample_observer=capture.record_corrected_eeg,
        raw_lsl_observer=capture.record_raw_lsl,
        clock_observer=capture.record_eeg_clock,
    )
    raw = np.arange(20, dtype=np.float32).reshape(10, 2)
    corrected = 100.0 + np.arange(10) / 250
    original = corrected.copy()
    original[3] = original[2] - .01
    inlets = []

    class Inlet:
        def __init__(self, info, **kwargs):
            self.flags = kwargs["processing_flags"]
            self.cursor = 0
            self.closed = False
            self.reads = []
            inlets.append(self)
        def open_stream(self, timeout): pass
        def flush(self): pass
        def time_correction(self, timeout): return 17.0
        def close_stream(self): self.closed = True
        def pull_chunk(self, timeout, max_samples):
            self.reads.append((timeout, max_samples))
            if self.flags:
                start = self.cursor
                self.cursor += 5
                if self.cursor == 10:
                    streamer.stop_streaming()
                return raw[start:self.cursor].tolist(), corrected[start:self.cursor].tolist()
            if self.cursor:
                return [], []
            self.cursor = 1
            # Different startup row count and batch size; raw data must stay independent.
            return raw[1:].tolist(), original[1:].tolist()

    module = ModuleType("pylsl")
    module.StreamInlet = Inlet
    module.resolve_streams = lambda: [Stream()]
    module.local_clock = lambda: 100.0
    module.proc_clocksync, module.proc_dejitter, module.proc_monotonize = 1, 2, 4
    monkeypatch.setitem(sys.modules, "pylsl", module)
    monkeypatch.setattr("streaming.eeg.signal_streamer.time.monotonic", lambda: 110.0)
    streamer.start_streaming()
    packets = [streamer.pop_packet(), streamer.pop_packet()]
    assert all(p is not None for p in packets)
    assert packets[0].start_time_s == 110.0
    assert packets[0].end_time_s == pytest.approx(110.028)
    assert packets[1].start_time_s == pytest.approx(110.008)
    np.testing.assert_array_equal(packets[0].samples[:, 2:], packets[1].samples[:, :6])
    assert [inlet.flags for inlet in inlets] == [0, 7]
    assert all(inlet.closed for inlet in inlets)
    assert inlets[1].reads == [(.02, 5), (.02, 5)]
    path = capture.save(tmp_path / "debug.npz", paired_frames=[], metadata=streamer.timing_signature)
    with np.load(path) as data:
        np.testing.assert_array_equal(data["eeg_native_samples"], raw)
        np.testing.assert_allclose(data["eeg_native_monotonic_time_s"], corrected + 10)
        np.testing.assert_array_equal(data["eeg_native_corrected_lsl_time_s"], corrected)
        assert np.isnan(data["eeg_native_source_time_s"]).all()
        np.testing.assert_array_equal(data["eeg_raw_lsl_samples"], raw[1:])
        np.testing.assert_array_equal(data["eeg_raw_lsl_source_time_s"], original[1:])
        clock = json.loads(str(data["eeg_clock_measurements_json"]))[0]
        assert clock["applied_offset_s"] == 10
        assert clock["source_clock_correction_s"] == 17


@pytest.mark.parametrize("times", [[1., float("nan")], [1., .9]])
def test_corrected_ingress_rejects_invalid_timestamps(times):
    streamer = SignalStreamer(channel_indices=(0,), bandstop_hz=None)
    with pytest.raises(ValueError, match="Corrected EEG LSL timestamps"):
        streamer._append_samples(np.ones((2, 1)), sample_times=np.array(times),
                                 corrected_lsl_times=np.array(times))


def test_corrected_ingress_rejects_boundary_reversal():
    streamer = SignalStreamer(channel_indices=(0,), bandstop_hz=None)
    times = np.array([1., 1.004])
    streamer._append_samples(np.ones((2, 1)), sample_times=times, corrected_lsl_times=times)
    with pytest.raises(ValueError, match="did not advance"):
        streamer._append_samples(np.ones((2, 1)), sample_times=times, corrected_lsl_times=times)


def test_clock_measurement_uses_shortest_bracket(monkeypatch):
    # Nine delayed measurements must not bias the one clean clock mapping.
    ticks = iter([value for i in range(10) for value in (110., 110. + (0. if i == 4 else .1))])
    monkeypatch.setattr("streaming.eeg.signal_streamer.time.monotonic", lambda: next(ticks))
    measurement = SignalStreamer()._measure_clock(lambda: 100.)
    assert measurement["measured_offset_s"] == 10.
    assert measurement["read_uncertainty_s"] == time.get_clock_info("monotonic").resolution / 2
