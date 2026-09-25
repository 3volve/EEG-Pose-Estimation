from __future__ import annotations

import math
import queue
import time
from collections.abc import Sequence
from typing import Callable, Final

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, sosfilt_zi

from config import (
    EEG_BANDSTOP_HZ,
    EEG_BANDSTOP_ORDER,
    EEG_PACKET_SIZE,
    EEG_PACKET_STRIDE,
    EEG_PREPROCESSING_VERSION,
    EEG_SAMPLE_RATE,
    EEG_SOURCE_CHANNEL_INDICES,
    EEG_ACQUISITION_SAMPLES,
    EEG_ACQUISITION_TIMEOUT_S,
)

from .packets import EegPacket


_USE_CONFIG_BANDSTOP: Final = object()

RawSampleObserver = Callable[
    [
        NDArray[np.float32],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
    ],
    None,
]


class SignalStreamer:
    """Stream EEG chunks from LSL and emit fixed-size timestamped packets."""

    def __init__(
        self,
        *,
        sample_rate: int = EEG_SAMPLE_RATE,
        packet_size: int = EEG_PACKET_SIZE,
        packet_stride: int = EEG_PACKET_STRIDE,
        channel_indices: Sequence[int] | None = None,
        n_channels: int | None = None,
        bandstop_hz: tuple[float, float] | None | object = _USE_CONFIG_BANDSTOP,
        bandstop_order: int = EEG_BANDSTOP_ORDER,
        raw_sample_observer: RawSampleObserver | None = None,
        corrected_sample_observer: RawSampleObserver | None = None,
        raw_lsl_observer: Callable[[NDArray[np.float32], NDArray[np.float64], float], None] | None = None,
        clock_observer: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        if channel_indices is not None and n_channels is not None:
            raise ValueError("Specify channel_indices or n_channels, not both.")
        if channel_indices is None:
            channel_indices = (
                tuple(range(n_channels))
                if n_channels is not None
                else EEG_SOURCE_CHANNEL_INDICES
            )
        resolved_indices = tuple(channel_indices)
        if not resolved_indices:
            raise ValueError("EEG source channel indices must not be empty.")
        if any(
            not isinstance(index, (int, np.integer)) or isinstance(index, bool)
            for index in resolved_indices
        ):
            raise TypeError(
                "EEG source channel indices must contain only integer column indices."
            )
        if any(index < 0 for index in resolved_indices):
            raise ValueError(
                f"EEG source channel indices must be non-negative; got "
                f"{resolved_indices}."
            )
        if len(set(resolved_indices)) != len(resolved_indices):
            raise ValueError(
                f"EEG source channel indices must be unique; got {resolved_indices}."
            )
        if sample_rate <= 0:
            raise ValueError(f"EEG sample rate must be positive; got {sample_rate}.")
        if packet_size <= 0 or packet_stride <= 0:
            raise ValueError(
                "EEG packet size and stride must both be positive; got "
                f"size={packet_size}, stride={packet_stride}."
            )
        if bandstop_order <= 0:
            raise ValueError(
                f"EEG band-stop order must be positive; got {bandstop_order}."
            )

        if bandstop_hz is _USE_CONFIG_BANDSTOP:
            # Custom low-rate streamers are used by small packetization tests. The
            # production/default 250 Hz path always enables the configured filter.
            resolved_bandstop = (
                EEG_BANDSTOP_HZ if sample_rate == EEG_SAMPLE_RATE else None
            )
        else:
            resolved_bandstop = bandstop_hz
        if resolved_bandstop is not None:
            low_hz, high_hz = resolved_bandstop
            nyquist_hz = sample_rate / 2.0
            if not 0.0 < low_hz < high_hz < nyquist_hz:
                raise ValueError(
                    "EEG band-stop frequencies must satisfy "
                    f"0 < low < high < Nyquist ({nyquist_hz:g} Hz); got "
                    f"{resolved_bandstop}."
                )

        self.sample_rate = sample_rate
        self.packet_size = packet_size
        self.packet_stride = packet_stride
        self.channel_indices = tuple(int(index) for index in resolved_indices)
        self.n_channels = len(self.channel_indices)
        self.bandstop_hz = resolved_bandstop
        self.bandstop_order = bandstop_order
        self.raw_sample_observer = raw_sample_observer
        self.corrected_sample_observer = corrected_sample_observer
        self.raw_lsl_observer = raw_lsl_observer
        self.clock_observer = clock_observer
        self.acquisition_samples = EEG_ACQUISITION_SAMPLES
        self.stream_timeout_s = EEG_ACQUISITION_TIMEOUT_S
        self._last_live_time_s: float | None = None
        self._packets: queue.SimpleQueue[EegPacket] = queue.SimpleQueue()
        self._sample_buffer: NDArray[np.float32] | None = None
        self._time_buffer: NDArray[np.float64] | None = None
        self._stop_requested = False
        self._stream_error: Exception | None = None
        self._next_packet_id = 0
        self._bandstop_sos: NDArray[np.float64] | None = None
        self._bandstop_state: NDArray[np.float64] | None = None
        if self.bandstop_hz is not None:
            self._bandstop_sos = np.asarray(
                butter(
                    self.bandstop_order,
                    self.bandstop_hz,
                    btype="bandstop",
                    fs=self.sample_rate,
                    output="sos",
                ),
                dtype=np.float64,
            )

    @property
    def preprocessing_signature(self) -> dict[str, object]:
        """Return the serializable EEG ingress contract for saved data/models."""
        bandstop_low_hz: float | None = None
        bandstop_high_hz: float | None = None
        if self.bandstop_hz is not None:
            bandstop_low_hz, bandstop_high_hz = self.bandstop_hz
        return {
            "pipeline_version": EEG_PREPROCESSING_VERSION,
            "source_channel_indices": self.channel_indices,
            "sample_rate_hz": self.sample_rate,
            "bandstop_low_hz": bandstop_low_hz,
            "bandstop_high_hz": bandstop_high_hz,
            "bandstop_order": (
                self.bandstop_order if self.bandstop_hz is not None else None
            ),
        }

    @property
    def timing_signature(self) -> dict[str, object]:
        return {
            "eeg_timing_version": "lsl-clocksync-dejitter-monotonize-python-clock-v1",
            "eeg_acquisition_samples": self.acquisition_samples,
            "eeg_acquisition_timeout_s": self.stream_timeout_s,
            "eeg_clock_mapping": "fixed-session-offset-measured-local-clocks",
            "eeg_raw_lsl_diagnostics": self.raw_lsl_observer is not None,
        }

    def _measure_clock(self, local_clock) -> dict[str, float]:
        # Bracket the LSL read; choose the measurement least affected by scheduling.
        measurements = []
        clock_resolution_s = time.get_clock_info("monotonic").resolution
        for _ in range(10):
            before = time.monotonic()
            lsl_time = local_clock()
            after = time.monotonic()
            measurements.append({
                "lsl_time_s": lsl_time,
                "python_time_s": (before + after) / 2.0,
                "read_uncertainty_s": (after - before + clock_resolution_s) / 2.0,
                "python_clock_resolution_s": clock_resolution_s,
                "measured_offset_s": (before + after) / 2.0 - lsl_time,
            })
        return min(measurements, key=lambda item: item["read_uncertainty_s"])

    def start_streaming(self) -> None:
        inlet = None
        raw_inlet = None
        try:
            from pylsl import (
                StreamInlet, resolve_streams, local_clock,
                proc_clocksync, proc_dejitter, proc_monotonize,
            )

            eeg_streams = [stream for stream in resolve_streams() if stream.type() == "EEG"]
            if not eeg_streams:
                raise RuntimeError(
                    "No EEG streams found. Make sure OpenBCI GUI or CLI is streaming."
                )
            eeg_stream = eeg_streams[0]
            self._validate_stream_metadata(
                stream_name=eeg_stream.name(),
                channel_count=eeg_stream.channel_count(),
                nominal_sample_rate=eeg_stream.nominal_srate(),
            )
            # The raw inlet is independent: never zip its batches with processed batches.
            if self.raw_lsl_observer is not None:
                raw_inlet = StreamInlet(eeg_stream, processing_flags=0)
                raw_inlet.open_stream(timeout=5.0)
            inlet = StreamInlet(
                eeg_stream,
                max_chunklen=self.acquisition_samples,
                processing_flags=proc_clocksync | proc_dejitter | proc_monotonize,
            )
            inlet.open_stream(timeout=5.0)
            # Prime clock correction; proc_clocksync applies it, not our Python code.
            inlet.time_correction(timeout=5.0)
            inlet.flush()
            measurement = self._measure_clock(local_clock)
            offset = measurement["measured_offset_s"]
            next_clock_check = measurement["python_time_s"]
            print("Connected to LSL stream:", eeg_stream.name())

            while not self._stop_requested:
                if time.monotonic() >= next_clock_check:
                    measurement = self._measure_clock(local_clock)
                    measurement["applied_offset_s"] = offset
                    measurement["source_clock_correction_s"] = inlet.time_correction(timeout=0.0)
                    if self.clock_observer is not None:
                        self.clock_observer(measurement)
                    next_clock_check = measurement["python_time_s"] + 5.0
                samples, timestamps = inlet.pull_chunk(
                    self.stream_timeout_s, self.acquisition_samples,
                )
                if samples:
                    corrected_times = np.asarray(timestamps, dtype=np.float64)
                    self._append_samples(
                        np.asarray(samples, dtype=np.float32),
                        sample_times=corrected_times + offset,
                        corrected_lsl_times=corrected_times,
                    )
                if raw_inlet is not None:
                    # Nonblocking and bounded work keeps diagnostics off the live wait path.
                    raw_samples, raw_times = raw_inlet.pull_chunk(0.0, 1024)
                    if raw_samples:
                        self.raw_lsl_observer(
                            np.asarray(raw_samples, dtype=np.float32),
                            np.asarray(raw_times, dtype=np.float64),
                            time.monotonic(),
                        )
        except Exception as error:
            self._stream_error = error
            raise
        finally:
            if inlet is not None:
                inlet.close_stream()
            if raw_inlet is not None:
                raw_inlet.close_stream()

    def stop_streaming(self) -> None:
        self._stop_requested = True

    def pop_packet(self) -> EegPacket | None:
        if self._stream_error is not None:
            error = self._stream_error
            raise RuntimeError(
                "EEG streaming stopped after an LSL ingress failure; fix the "
                f"stream and restart this capture. Cause: {type(error).__name__}: "
                f"{error}"
            ) from error
        try:
            return self._packets.get_nowait()
        except queue.Empty:
            return None

    def _append_samples(
        self,
        raw_samples: NDArray[np.float32],
        *,
        source_timestamps: NDArray[np.float64] | None = None,
        sample_times: NDArray[np.float64] | None = None,
        corrected_lsl_times: NDArray[np.float64] | None = None,
    ) -> None:
        if raw_samples.ndim != 2:
            raise ValueError(
                "EEG LSL samples must be a 2D samples-by-columns array; got shape "
                f"{raw_samples.shape}."
            )
        self._validate_channel_count(
            raw_samples.shape[1],
            source_context="received EEG LSL chunk",
        )
        if len(raw_samples) == 0:
            return
        if source_timestamps is not None and source_timestamps.shape != (
            len(raw_samples),
        ):
            raise ValueError(
                "EEG LSL source timestamps must have one value per sample; got "
                f"shape {source_timestamps.shape} for {len(raw_samples)} samples."
            )

        received_time_s = time.monotonic()
        if sample_times is None:
            # Offline/test injection only. Live ingress always supplies mapped LSL time.
            sample_offsets = np.arange(len(raw_samples), dtype=np.float64) - len(raw_samples) + 1
            sample_times = received_time_s + sample_offsets / self.sample_rate
        else:
            if sample_times.shape != (len(raw_samples),) or not np.isfinite(sample_times).all():
                raise ValueError("Corrected EEG LSL timestamps must be finite, one per sample.")
            if np.any(np.diff(sample_times) <= 0) or (
                self._last_live_time_s is not None and sample_times[0] <= self._last_live_time_s
            ):
                raise ValueError("Corrected EEG LSL timestamps did not advance; restart the capture.")
            self._last_live_time_s = float(sample_times[-1])
            assert corrected_lsl_times is not None
            if self.corrected_sample_observer is not None:
                self.corrected_sample_observer(
                    raw_samples, sample_times, corrected_lsl_times, received_time_s,
                )
        if self.raw_sample_observer is not None:
            self.raw_sample_observer(
                raw_samples,
                sample_times,
                (
                    source_timestamps
                    if source_timestamps is not None
                    else np.full(len(raw_samples), np.nan, dtype=np.float64)
                ),
                received_time_s,
            )

        signals = raw_samples[:, self.channel_indices]
        signals = self._apply_bandstop(signals)
        signals = signals.T.astype(np.float32, copy=False)

        if self._sample_buffer is None:
            self._sample_buffer = signals
            self._time_buffer = sample_times
        else:
            self._sample_buffer = np.concatenate((self._sample_buffer, signals), axis=1)
            assert self._time_buffer is not None
            self._time_buffer = np.concatenate((self._time_buffer, sample_times))

        self._emit_ready_packets()

    def _apply_bandstop(
        self,
        signals: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        if self._bandstop_sos is None:
            return signals.astype(np.float32, copy=False)
        if self._bandstop_state is None:
            initial_state = sosfilt_zi(self._bandstop_sos)
            self._bandstop_state = (
                initial_state[:, :, np.newaxis]
                * signals[0][np.newaxis, np.newaxis, :]
            )
        filtered, self._bandstop_state = sosfilt(
            self._bandstop_sos,
            signals,
            axis=0,
            zi=self._bandstop_state,
        )
        return filtered.astype(np.float32, copy=False)

    def _validate_stream_metadata(
        self,
        *,
        stream_name: str,
        channel_count: int,
        nominal_sample_rate: float,
    ) -> None:
        self._validate_channel_count(
            channel_count,
            source_context=f"EEG LSL stream {stream_name!r}",
        )
        if not math.isclose(
            nominal_sample_rate,
            self.sample_rate,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                f"EEG LSL stream {stream_name!r} advertises "
                f"{nominal_sample_rate:g} Hz; expected {self.sample_rate} Hz."
            )

    def _validate_channel_count(
        self,
        channel_count: int,
        *,
        source_context: str,
    ) -> None:
        highest_index = max(self.channel_indices)
        if highest_index >= channel_count:
            raise ValueError(
                f"{source_context} has {channel_count} columns, but configured EEG "
                f"source indices {self.channel_indices} require column "
                f"{highest_index}."
            )

    def _emit_ready_packets(self) -> None:
        assert self._sample_buffer is not None
        assert self._time_buffer is not None

        while self._sample_buffer.shape[1] >= self.packet_size:
            samples = self._sample_buffer[:, : self.packet_size].copy()
            times = self._time_buffer[: self.packet_size]
            self._packets.put(
                EegPacket(
                    packet_id=self._next_packet_id,
                    samples=samples,
                    start_time_s=float(times[0]),
                    end_time_s=float(times[-1]),
                    sample_rate=self.sample_rate,
                )
            )
            self._next_packet_id += 1
            self._sample_buffer = self._sample_buffer[:, self.packet_stride :]
            self._time_buffer = self._time_buffer[self.packet_stride :]
