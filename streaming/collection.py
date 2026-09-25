from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

from config import PAIRING_MAX_POSE_GAP_S, PAIRING_POLL_DELAY_S
from streaming.eeg import EegPacket

from .dataset import save_paired_frames
from .pose_buffer import PoseLatentBuffer
from .records import PairedTrainingFrame

if TYPE_CHECKING:
    from pose_encoding import PoseLatentFrame
    from .debug_capture import RawPoseEegDebugCapture


class EegPacketStream(Protocol):
    def pop_packet(self) -> EegPacket | None: ...


class PoseLatentStreamLike(Protocol):
    def get_latest(self) -> "PoseLatentFrame | None": ...


def pair_packet(
    packet: EegPacket,
    pose_buffer: PoseLatentBuffer,
) -> PairedTrainingFrame | None:
    interpolated = pose_buffer.latent_at(packet.end_time_s)
    if interpolated is None:
        return None
    return PairedTrainingFrame(
        packet_id=packet.packet_id,
        eeg=packet.samples,
        target_time_s=packet.end_time_s,
        pose_latent=interpolated.latent,
        pose_confidence=interpolated.pose_confidence,
        pose_reconstruction_error=interpolated.pose_reconstruction_error,
        interpolation_confidence=interpolated.interpolation_confidence,
    )


def collect_paired_frames(
    eeg_stream: EegPacketStream,
    pose_stream: PoseLatentStreamLike,
    *,
    duration_s: float,
    max_pose_gap_s: float = PAIRING_MAX_POSE_GAP_S,
    poll_delay_s: float = PAIRING_POLL_DELAY_S,
    debug_capture: "RawPoseEegDebugCapture | None" = None,
) -> list[PairedTrainingFrame]:
    pose_buffer = PoseLatentBuffer(max_gap_s=max_pose_gap_s)
    frames: list[PairedTrainingFrame] = []
    pending_packets: list[EegPacket] = []
    deadline = time.monotonic() + duration_s

    while time.monotonic() < deadline:
        pose_frame = pose_stream.get_latest()
        pose_buffer.add(pose_frame)
        if debug_capture is not None:
            debug_capture.record_processed_pose(pose_frame)
        packet = eeg_stream.pop_packet()
        if packet is not None:
            if debug_capture is not None:
                debug_capture.record_eeg_packet(packet)
            pending_packets.append(packet)

        still_pending: list[EegPacket] = []
        latest_pose_time_s = pose_buffer.latest_time_s
        for pending_packet in pending_packets:
            paired = pair_packet(pending_packet, pose_buffer)
            if paired is not None:
                frames.append(paired)
            elif (
                latest_pose_time_s is None
                or latest_pose_time_s <= pending_packet.end_time_s
            ):
                still_pending.append(pending_packet)
        pending_packets = still_pending
        time.sleep(poll_delay_s)
    return frames


def collect_and_save_paired_frames(
    eeg_stream: EegPacketStream,
    pose_stream: PoseLatentStreamLike,
    *,
    duration_s: float,
    out_path: str,
    max_pose_gap_s: float = PAIRING_MAX_POSE_GAP_S,
    metadata: dict[str, object] | None = None,
    debug_capture: "RawPoseEegDebugCapture | None" = None,
) -> list[PairedTrainingFrame]:
    frames = collect_paired_frames(
        eeg_stream,
        pose_stream,
        duration_s=duration_s,
        max_pose_gap_s=max_pose_gap_s,
        debug_capture=debug_capture,
    )
    save_paired_frames(
        out_path, frames,
        metadata={**(metadata or {}), "pairing_pose_time_basis": "capture_timestamp_ms"},
    )
    return frames
