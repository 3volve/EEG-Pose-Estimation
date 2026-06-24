from __future__ import annotations

import queue
import time

import numpy as np
from numpy.typing import NDArray

from config import (
    EEG_CHANNELS,
    EEG_PACKET_SIZE,
    EEG_PACKET_STRIDE,
    EEG_SAMPLE_RATE,
    EEG_STREAM_TIMEOUT_MARGIN_S,
)

from .packets import EegPacket


class SignalStreamer:
    """Stream EEG chunks from LSL and emit fixed-size timestamped packets."""

    def __init__(
        self,
        *,
        sample_rate: int = EEG_SAMPLE_RATE,
        packet_size: int = EEG_PACKET_SIZE,
        packet_stride: int = EEG_PACKET_STRIDE,
        n_channels: int = EEG_CHANNELS,
    ) -> None:
        self.sample_rate = sample_rate
        self.packet_size = packet_size
        self.packet_stride = packet_stride
        self.n_channels = n_channels
        self.stream_timeout_s = packet_stride / sample_rate + EEG_STREAM_TIMEOUT_MARGIN_S
        self._packets: queue.SimpleQueue[EegPacket] = queue.SimpleQueue()
        self._sample_buffer: NDArray[np.float32] | None = None
        self._time_buffer: NDArray[np.float64] | None = None
        self._stop_requested = False
        self._next_packet_id = 0

    def start_streaming(self) -> None:
        from pylsl import StreamInlet, resolve_streams

        streams = resolve_streams()
        eeg_streams = [stream for stream in streams if stream.type() == "EEG"]
        if not eeg_streams:
            raise RuntimeError(
                "No EEG streams found. Make sure OpenBCI GUI or CLI is streaming."
            )

        inlet = StreamInlet(eeg_streams[0])
        inlet.flush()
        print("Connected to LSL stream:", eeg_streams[0].name())

        while not self._stop_requested:
            samples, _ = inlet.pull_chunk(self.stream_timeout_s, self.packet_stride)
            if not samples:
                continue
            self._append_samples(np.asarray(samples, dtype=np.float32))

    def stop_streaming(self) -> None:
        self._stop_requested = True

    def pop_packet(self) -> EegPacket | None:
        try:
            return self._packets.get_nowait()
        except queue.Empty:
            return None

    def _append_samples(self, raw_samples: NDArray[np.float32]) -> None:
        signals = raw_samples[:, : self.n_channels]
        if signals.size == 0:
            return

        end_time_s = time.monotonic()
        sample_offsets = np.arange(len(signals), dtype=np.float64) - len(signals) + 1
        sample_times = end_time_s + sample_offsets / self.sample_rate
        signals = signals.T.astype(np.float32, copy=False)

        if self._sample_buffer is None:
            self._sample_buffer = signals
            self._time_buffer = sample_times
        else:
            self._sample_buffer = np.concatenate((self._sample_buffer, signals), axis=1)
            assert self._time_buffer is not None
            self._time_buffer = np.concatenate((self._time_buffer, sample_times))

        self._emit_ready_packets()

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
