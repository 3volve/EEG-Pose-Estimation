from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from streaming.pose import PoseResult


UPPER_BODY_NAMES: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)
UPPER_BODY_LANDMARK_IDS: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 23, 24)
UPPER_BODY_BONES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (2, 4),
    (1, 3),
    (3, 5),
    (0, 6),
    (1, 7),
    (6, 7),
)
REGION_LANDMARKS: dict[str, tuple[int, ...]] = {
    "left_arm": (0, 2, 4),
    "right_arm": (1, 3, 5),
    "shoulders_core": (0, 1, 6, 7),
    "hips": (6, 7),
}
_POSITION_DIM = len(UPPER_BODY_NAMES) * 3


@dataclass(frozen=True, slots=True)
class CalibrationMovementBlock:
    name: str
    region: str
    duration_s: float
    rest: bool = False


@dataclass(frozen=True, slots=True)
class ProfileBlockResult:
    block_id: int
    movement_name: str
    repeat_index: int
    start_time_s: float
    end_time_s: float
    accepted: bool
    acceptance_score: float
    reject_reason: str
    paired_sample_count: int
    pose_sample_count: int
    mean_pose_confidence: float
    target_motion: float
    rest_motion: float
    direction_score: float
    jump_score: float
    guide_distance: float


@dataclass(frozen=True, slots=True)
class RegionScores:
    left_arm: float
    right_arm: float
    shoulders_core: float
    hips: float
    rest_false_positive: float
    movement_response: float


@dataclass(frozen=True, slots=True)
class CalibrationDisplayStatus:
    readiness_score: float
    ready: bool
    trusted_samples: int
    skipped_samples: int
    update_count: int
    latest_loss: float | None = None


DEFAULT_CALIBRATION_BLOCKS: tuple[CalibrationMovementBlock, ...] = (
    CalibrationMovementBlock("rest", "rest", 4.0, rest=True),
    CalibrationMovementBlock("left_arm_raise", "left_arm", 6.0),
    CalibrationMovementBlock("right_arm_raise", "right_arm", 6.0),
    CalibrationMovementBlock("elbow_bends", "arms", 6.0),
    CalibrationMovementBlock("wrist_forearm", "arms", 6.0),
    CalibrationMovementBlock("torso_shift", "shoulders_core", 6.0),
    CalibrationMovementBlock("hip_shift", "hips", 6.0),
)

PROFILE_BUILD_REPEATS: int = 3
PROFILE_BUILD_REST_DURATION_S: float = 4.0
PROFILE_BUILD_BLOCKS: tuple[CalibrationMovementBlock, ...] = (
    CalibrationMovementBlock("neutral_rest", "rest", 20.0, rest=True),
    CalibrationMovementBlock("left_arm_raise", "left_arm", 8.0),
    CalibrationMovementBlock("right_arm_raise", "right_arm", 8.0),
    CalibrationMovementBlock("left_elbow_bend", "left_arm", 8.0),
    CalibrationMovementBlock("right_elbow_bend", "right_arm", 8.0),
    CalibrationMovementBlock("left_forward_reach", "left_arm", 8.0),
    CalibrationMovementBlock("right_forward_reach", "right_arm", 8.0),
    CalibrationMovementBlock("both_arms_raise", "arms", 8.0),
    CalibrationMovementBlock("arms_open_close", "arms", 8.0),
    CalibrationMovementBlock("torso_shift", "shoulders_core", 8.0),
    CalibrationMovementBlock("hip_shift", "hips", 8.0),
)


def positions_from_feature_vector(
    feature_vector: NDArray[np.float32],
) -> NDArray[np.float32]:
    return np.asarray(feature_vector[:_POSITION_DIM], dtype=np.float32).reshape(8, 3)


def neutral_dummy_positions() -> NDArray[np.float32]:
    return np.asarray(
        [
            [-0.5, -1.55, 0.0],
            [0.5, -1.55, 0.0],
            [-0.9, -0.9, 0.0],
            [0.9, -0.9, 0.0],
            [-1.05, -0.2, 0.0],
            [1.05, -0.2, 0.0],
            [-0.4, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


def dummy_positions_for_block(
    block_name: str,
    elapsed_s: float,
    duration_s: float,
) -> NDArray[np.float32]:
    positions = neutral_dummy_positions()
    phase = _smooth_phase(elapsed_s, duration_s)

    if block_name == "left_arm_raise":
        positions[2] += np.asarray([-0.12, -0.75 * phase, 0.0], dtype=np.float32)
        positions[4] += np.asarray([-0.06, -1.35 * phase, 0.0], dtype=np.float32)
    elif block_name == "right_arm_raise":
        positions[3] += np.asarray([0.12, -0.75 * phase, 0.0], dtype=np.float32)
        positions[5] += np.asarray([0.06, -1.35 * phase, 0.0], dtype=np.float32)
    elif block_name == "left_elbow_bend":
        bend = np.sin(2.0 * np.pi * _cycle(elapsed_s, duration_s))
        positions[4] += np.asarray([0.32 * bend, -0.35 * abs(bend), 0.0], dtype=np.float32)
    elif block_name == "right_elbow_bend":
        bend = np.sin(2.0 * np.pi * _cycle(elapsed_s, duration_s))
        positions[5] += np.asarray([-0.32 * bend, -0.35 * abs(bend), 0.0], dtype=np.float32)
    elif block_name == "left_forward_reach":
        reach = _smooth_phase(elapsed_s, duration_s)
        positions[2] += np.asarray([-0.05, -0.15 * reach, -0.2 * reach], dtype=np.float32)
        positions[4] += np.asarray([-0.08, -0.35 * reach, -0.75 * reach], dtype=np.float32)
    elif block_name == "right_forward_reach":
        reach = _smooth_phase(elapsed_s, duration_s)
        positions[3] += np.asarray([0.05, -0.15 * reach, -0.2 * reach], dtype=np.float32)
        positions[5] += np.asarray([0.08, -0.35 * reach, -0.75 * reach], dtype=np.float32)
    elif block_name == "both_arms_raise":
        positions[2] += np.asarray([-0.12, -0.75 * phase, 0.0], dtype=np.float32)
        positions[4] += np.asarray([-0.06, -1.35 * phase, 0.0], dtype=np.float32)
        positions[3] += np.asarray([0.12, -0.75 * phase, 0.0], dtype=np.float32)
        positions[5] += np.asarray([0.06, -1.35 * phase, 0.0], dtype=np.float32)
    elif block_name == "arms_open_close":
        open_amount = 0.45 * np.sin(2.0 * np.pi * _cycle(elapsed_s, duration_s))
        positions[[2, 4], 0] -= open_amount
        positions[[3, 5], 0] += open_amount
    elif block_name == "elbow_bends":
        bend = np.sin(2.0 * np.pi * _cycle(elapsed_s, duration_s))
        positions[4] += np.asarray([0.25 * bend, -0.35 * abs(bend), 0.0], dtype=np.float32)
        positions[5] += np.asarray([-0.25 * bend, -0.35 * abs(bend), 0.0], dtype=np.float32)
    elif block_name == "wrist_forearm":
        wave = np.sin(4.0 * np.pi * _cycle(elapsed_s, duration_s))
        positions[4] += np.asarray([0.2 * wave, -0.15 * wave, 0.0], dtype=np.float32)
        positions[5] += np.asarray([-0.2 * wave, 0.15 * wave, 0.0], dtype=np.float32)
    elif block_name == "torso_shift":
        shift = 0.25 * np.sin(2.0 * np.pi * _cycle(elapsed_s, duration_s))
        positions[[0, 1, 6, 7], 0] += shift
    elif block_name == "hip_shift":
        shift = 0.3 * np.sin(2.0 * np.pi * _cycle(elapsed_s, duration_s))
        positions[[6, 7], 0] += shift
        positions[[0, 1], 0] -= 0.15 * shift

    return positions


def profile_build_sequence(
    *,
    repeats: int = PROFILE_BUILD_REPEATS,
) -> tuple[CalibrationMovementBlock, ...]:
    sequence: list[CalibrationMovementBlock] = [PROFILE_BUILD_BLOCKS[0]]
    rest = CalibrationMovementBlock(
        "rest_between_blocks",
        "rest",
        PROFILE_BUILD_REST_DURATION_S,
        rest=True,
    )
    movements = [block for block in PROFILE_BUILD_BLOCKS if not block.rest]
    for repeat_index in range(repeats):
        for block in movements:
            sequence.append(block)
            if repeat_index != repeats - 1 or block != movements[-1]:
                sequence.append(rest)
    return tuple(sequence)


def validate_profile_block(
    *,
    block_id: int,
    block: CalibrationMovementBlock,
    repeat_index: int,
    start_time_s: float,
    end_time_s: float,
    feature_vectors: list[NDArray[np.float32]],
    pose_confidences: list[float],
    paired_sample_count: int,
) -> ProfileBlockResult:
    if not feature_vectors:
        return _profile_block_result(
            block_id,
            block,
            repeat_index,
            start_time_s,
            end_time_s,
            accepted=False,
            score=0.0,
            reason="low_pose_quality",
            paired_sample_count=paired_sample_count,
        )

    positions = np.stack(
        [positions_from_feature_vector(vector) for vector in feature_vectors],
        axis=0,
    )
    confidence = float(np.mean(pose_confidences)) if pose_confidences else 0.0
    diffs = np.diff(positions, axis=0)
    frame_motion = np.linalg.norm(diffs, axis=2) if len(positions) > 1 else np.zeros((0, 8))
    landmark_motion = np.linalg.norm(np.ptp(positions, axis=0), axis=1)
    target_indices = _target_landmarks_for_block(block)
    other_indices = tuple(index for index in range(8) if index not in target_indices)
    target_motion = float(np.mean(landmark_motion[list(target_indices)]))
    other_motion = float(np.mean(landmark_motion[list(other_indices)])) if other_indices else 0.0
    rest_motion = float(np.mean(landmark_motion))
    jump_score = _impossible_jump_score(frame_motion)
    direction_score = _direction_agreement(block.name, positions)
    guide_distance = _guide_distance(block.name, positions, start_time_s, end_time_s)

    if confidence < 0.45 or len(positions) < 3:
        accepted = False
        reason = "low_pose_quality"
    elif jump_score > 1.0:
        accepted = False
        reason = "impossible_pose_jumps"
    elif block.rest and rest_motion > 0.2:
        accepted = False
        reason = "rest_too_active"
    elif not block.rest and target_motion < 0.12:
        accepted = False
        reason = "insufficient_target_motion"
    elif not block.rest and target_motion < other_motion * 0.65:
        accepted = False
        reason = "insufficient_target_motion"
    elif not block.rest and direction_score < 0.15:
        accepted = False
        reason = "wrong_motion_pattern"
    else:
        accepted = True
        reason = ""

    pose_score = np.clip((confidence - 0.45) / 0.45, 0.0, 1.0)
    jump_component = np.clip(1.0 - jump_score, 0.0, 1.0)
    if block.rest:
        motion_component = np.clip(1.0 - rest_motion / 0.2, 0.0, 1.0)
        direction_component = 1.0
    else:
        motion_component = np.clip(target_motion / 0.45, 0.0, 1.0)
        direction_component = np.clip(direction_score, 0.0, 1.0)
    guide_component = np.clip(1.0 - guide_distance / 2.0, 0.0, 1.0)
    score = float(
        0.3 * pose_score
        + 0.3 * jump_component
        + 0.25 * motion_component
        + 0.12 * direction_component
        + 0.03 * guide_component
    )
    if not accepted:
        score = min(score, 0.49)

    return ProfileBlockResult(
        block_id=block_id,
        movement_name=block.name,
        repeat_index=repeat_index,
        start_time_s=float(start_time_s),
        end_time_s=float(end_time_s),
        accepted=accepted,
        acceptance_score=score,
        reject_reason=reason,
        paired_sample_count=paired_sample_count,
        pose_sample_count=len(feature_vectors),
        mean_pose_confidence=confidence,
        target_motion=target_motion,
        rest_motion=rest_motion,
        direction_score=direction_score,
        jump_score=jump_score,
        guide_distance=guide_distance,
    )


def _profile_block_result(
    block_id: int,
    block: CalibrationMovementBlock,
    repeat_index: int,
    start_time_s: float,
    end_time_s: float,
    *,
    accepted: bool,
    score: float,
    reason: str,
    paired_sample_count: int,
) -> ProfileBlockResult:
    return ProfileBlockResult(
        block_id=block_id,
        movement_name=block.name,
        repeat_index=repeat_index,
        start_time_s=float(start_time_s),
        end_time_s=float(end_time_s),
        accepted=accepted,
        acceptance_score=score,
        reject_reason=reason,
        paired_sample_count=paired_sample_count,
        pose_sample_count=0,
        mean_pose_confidence=0.0,
        target_motion=0.0,
        rest_motion=0.0,
        direction_score=0.0,
        jump_score=0.0,
        guide_distance=float("inf"),
    )


def _target_landmarks_for_block(block: CalibrationMovementBlock) -> tuple[int, ...]:
    if block.name.startswith("left_"):
        return REGION_LANDMARKS["left_arm"]
    if block.name.startswith("right_"):
        return REGION_LANDMARKS["right_arm"]
    if block.name in ("both_arms_raise", "arms_open_close", "elbow_bends", "wrist_forearm"):
        return REGION_LANDMARKS["left_arm"] + REGION_LANDMARKS["right_arm"]
    if block.region in REGION_LANDMARKS:
        return REGION_LANDMARKS[block.region]
    return tuple(range(8))


def _impossible_jump_score(frame_motion: NDArray[np.float32]) -> float:
    if len(frame_motion) == 0:
        return 0.0
    jump = float(np.percentile(frame_motion, 98))
    return max(0.0, jump - 0.45) / 0.45


def _direction_agreement(block_name: str, positions: NDArray[np.float32]) -> float:
    if block_name in ("left_arm_raise", "both_arms_raise"):
        return _upward_range_score(positions, (2, 4))
    if block_name == "right_arm_raise":
        return _upward_range_score(positions, (3, 5))
    if block_name == "left_elbow_bend":
        return _upward_range_score(positions, (4,))
    if block_name == "right_elbow_bend":
        return _upward_range_score(positions, (5,))
    if block_name == "left_forward_reach":
        return _range_score(positions, (4,), axis=2) + _upward_range_score(positions, (4,))
    if block_name == "right_forward_reach":
        return _range_score(positions, (5,), axis=2) + _upward_range_score(positions, (5,))
    if block_name == "arms_open_close":
        left_range = float(np.ptp(positions[:, 4, 0]))
        right_range = float(np.ptp(positions[:, 5, 0]))
        return min(left_range + right_range, 1.0)
    if block_name in ("torso_shift", "hip_shift"):
        return min(float(np.ptp(positions[:, :, 0].mean(axis=1))), 1.0)
    if "rest" in block_name:
        return 1.0
    return 0.5


def _upward_range_score(positions: NDArray[np.float32], indices: tuple[int, ...]) -> float:
    starts = positions[0, list(indices), 1]
    highest = np.min(positions[:, list(indices), 1], axis=0)
    return float(max(0.0, np.mean(starts - highest)))


def _range_score(
    positions: NDArray[np.float32],
    indices: tuple[int, ...],
    *,
    axis: int,
) -> float:
    return float(np.mean(np.ptp(positions[:, list(indices), axis], axis=0)))


def _guide_distance(
    block_name: str,
    positions: NDArray[np.float32],
    start_time_s: float,
    end_time_s: float,
) -> float:
    if len(positions) == 0:
        return float("inf")
    duration = max(end_time_s - start_time_s, 1e-6)
    guide = np.stack(
        [
            dummy_positions_for_block(
                block_name,
                elapsed_s=duration * index / max(len(positions) - 1, 1),
                duration_s=duration,
            )
            for index in range(len(positions))
        ],
        axis=0,
    )
    guide_delta = guide - neutral_dummy_positions()
    actual_delta = positions - positions[0:1]
    return float(np.mean(np.linalg.norm(actual_delta - guide_delta, axis=2)))


def region_scores(
    truth_positions: NDArray[np.float32],
    predicted_positions: NDArray[np.float32],
    *,
    stillness_truth_velocity: float | None = None,
    predicted_velocity: float | None = None,
    movement_response: float = 0.0,
) -> RegionScores:
    errors = np.linalg.norm(predicted_positions - truth_positions, axis=1)
    rest_false_positive = 0.0
    if stillness_truth_velocity is not None and predicted_velocity is not None:
        if stillness_truth_velocity < 0.08:
            rest_false_positive = max(0.0, predicted_velocity - 0.12)
    return RegionScores(
        left_arm=float(np.mean(errors[list(REGION_LANDMARKS["left_arm"])])),
        right_arm=float(np.mean(errors[list(REGION_LANDMARKS["right_arm"])])),
        shoulders_core=float(np.mean(errors[list(REGION_LANDMARKS["shoulders_core"])])),
        hips=float(np.mean(errors[list(REGION_LANDMARKS["hips"])])),
        rest_false_positive=float(rest_false_positive),
        movement_response=float(movement_response),
    )


def select_next_movement_block(
    scores: RegionScores,
    *,
    recent_block_names: tuple[str, ...] = (),
) -> CalibrationMovementBlock:
    if scores.rest_false_positive > max(
        scores.left_arm,
        scores.right_arm,
        scores.shoulders_core,
        scores.hips,
    ):
        return _block_by_name("rest")

    region_values = {
        "left_arm": scores.left_arm,
        "right_arm": scores.right_arm,
        "shoulders_core": scores.shoulders_core,
        "hips": scores.hips,
    }
    target_region = max(region_values, key=region_values.get)
    candidates = [
        block
        for block in DEFAULT_CALIBRATION_BLOCKS
        if block.region == target_region and block.name not in recent_block_names[-2:]
    ]
    if not candidates:
        candidates = [
            block
            for block in DEFAULT_CALIBRATION_BLOCKS
            if block.region == target_region
        ]
    if candidates:
        return candidates[0]
    return _block_by_name("rest")


class CalibrationOverlayState:
    def __init__(
        self,
        *,
        error_scale: float = 0.75,
        eeg_display_smoothing: float = 0.7,
    ) -> None:
        self.error_scale = error_scale
        self.eeg_display_smoothing = eeg_display_smoothing
        self._lock = Lock()
        self._truth_positions: NDArray[np.float32] | None = None
        self._eeg_positions: NDArray[np.float32] | None = None
        self._dummy_positions: NDArray[np.float32] | None = None
        self._status: CalibrationDisplayStatus | None = None

    def update_truth(self, feature_vector: NDArray[np.float32]) -> None:
        with self._lock:
            self._truth_positions = positions_from_feature_vector(feature_vector)

    def update_eeg(self, decoded_feature_vector: NDArray[np.float32]) -> None:
        with self._lock:
            latest = positions_from_feature_vector(decoded_feature_vector)
            if self._eeg_positions is None:
                self._eeg_positions = latest
            else:
                alpha = self.eeg_display_smoothing
                self._eeg_positions = (
                    alpha * latest + (1.0 - alpha) * self._eeg_positions
                ).astype(np.float32, copy=False)

    def update_dummy(
        self,
        block: CalibrationMovementBlock,
        elapsed_s: float,
    ) -> None:
        with self._lock:
            self._dummy_positions = dummy_positions_for_block(
                block.name,
                elapsed_s,
                block.duration_s,
            )

    def update_status(self, status: CalibrationDisplayStatus) -> None:
        with self._lock:
            self._status = status

    def render(
        self,
        frame: Any,
        cv2: ModuleType,
        pose_result: PoseResult | None,
        mirror_x: bool,
    ) -> None:
        with self._lock:
            truth = None if self._truth_positions is None else self._truth_positions.copy()
            eeg = None if self._eeg_positions is None else self._eeg_positions.copy()
            dummy = None if self._dummy_positions is None else self._dummy_positions.copy()
            status = self._status

        anchor = _display_anchor(frame, pose_result, mirror_x, truth)
        if dummy is not None and truth is not None:
            dummy = _fit_dummy_to_truth(dummy, truth)
        if dummy is not None:
            _draw_plain_skeleton(
                frame,
                cv2,
                dummy,
                anchor,
                mirror_x=mirror_x,
                color=(255, 40, 230),
                alpha=0.7,
                thickness=4,
            )
        if truth is not None:
            _draw_plain_skeleton(
                frame,
                cv2,
                truth,
                anchor,
                mirror_x=mirror_x,
                color=(40, 230, 40),
                alpha=0.9,
                thickness=3,
            )
        if truth is not None and eeg is not None:
            _draw_error_skeleton(
                frame,
                cv2,
                eeg,
                truth,
                anchor,
                mirror_x=mirror_x,
                error_scale=self.error_scale,
            )
        if status is not None:
            _draw_readiness_bar(frame, cv2, status)


def _display_anchor(
    frame: Any,
    pose_result: PoseResult | None,
    mirror_x: bool,
    reference_positions: NDArray[np.float32] | None = None,
) -> tuple[float, float, float]:
    height, width = frame.shape[:2]
    default_scale = min(width, height) * 0.16
    if pose_result is None or not pose_result.pose_detected or not pose_result.landmarks:
        return width * 0.5, height * 0.68, default_scale

    if reference_positions is not None:
        fitted = _fit_display_anchor_to_image_landmarks(
            reference_positions,
            pose_result,
            width=width,
            height=height,
            mirror_x=mirror_x,
        )
        if fitted is not None:
            return fitted

    points = []
    for landmark_index in (11, 12, 23, 24):
        landmark = pose_result.landmarks[landmark_index]
        assert landmark.x is not None
        assert landmark.y is not None
        x = 1.0 - landmark.x if mirror_x else landmark.x
        points.append(np.asarray([x * width, landmark.y * height], dtype=np.float32))
    left_shoulder, right_shoulder, left_hip, right_hip = points
    root = (left_hip + right_hip) / 2.0
    shoulder_scale = np.linalg.norm(left_shoulder - right_shoulder)
    scale = float(shoulder_scale if shoulder_scale > 1.0 else default_scale)
    return float(root[0]), float(root[1]), scale


def _fit_display_anchor_to_image_landmarks(
    positions: NDArray[np.float32],
    pose_result: PoseResult,
    *,
    width: int,
    height: int,
    mirror_x: bool,
) -> tuple[float, float, float] | None:
    source_points = []
    target_points = []
    for local_index, landmark_index in enumerate(UPPER_BODY_LANDMARK_IDS):
        if landmark_index >= len(pose_result.landmarks):
            return None
        landmark = pose_result.landmarks[landmark_index]
        if landmark.x is None or landmark.y is None:
            continue
        source_x = -positions[local_index, 0] if mirror_x else positions[local_index, 0]
        source_points.append([source_x, positions[local_index, 1]])
        target_x = (1.0 - landmark.x if mirror_x else landmark.x) * width
        target_points.append([target_x, landmark.y * height])

    if len(source_points) < 4:
        return None

    source = np.asarray(source_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    source_centered = source - source.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    denominator = float(np.sum(source_centered * source_centered))
    if denominator <= 1e-6:
        return None
    scale = float(np.sum(source_centered * target_centered) / denominator)
    if not np.isfinite(scale) or scale <= 1.0:
        return None
    offset = target.mean(axis=0) - scale * source.mean(axis=0)
    return float(offset[0]), float(offset[1]), scale


def _fit_dummy_to_truth(
    dummy_positions: NDArray[np.float32],
    truth_positions: NDArray[np.float32],
) -> NDArray[np.float32]:
    neutral = neutral_dummy_positions()
    movement_delta = dummy_positions - neutral
    return (truth_positions + movement_delta).astype(np.float32, copy=False)


def _points(
    positions: NDArray[np.float32],
    anchor: tuple[float, float, float],
    mirror_x: bool,
) -> list[tuple[int, int]]:
    root_x, root_y, scale = anchor
    return [
        (
            round(root_x + (-float(position[0]) if mirror_x else float(position[0])) * scale),
            round(root_y + float(position[1]) * scale),
        )
        for position in positions
    ]


def _draw_plain_skeleton(
    frame: Any,
    cv2: ModuleType,
    positions: NDArray[np.float32],
    anchor: tuple[float, float, float],
    *,
    mirror_x: bool,
    color: tuple[int, int, int],
    alpha: float,
    thickness: int,
) -> None:
    overlay = frame.copy()
    points = _points(positions, anchor, mirror_x)
    for start, end in UPPER_BODY_BONES:
        cv2.line(
            overlay,
            points[start],
            points[end],
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    for point in points:
        cv2.circle(
            overlay,
            point,
            radius=4,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def _draw_error_skeleton(
    frame: Any,
    cv2: ModuleType,
    eeg_positions: NDArray[np.float32],
    truth_positions: NDArray[np.float32],
    anchor: tuple[float, float, float],
    *,
    mirror_x: bool,
    error_scale: float,
) -> None:
    errors = np.linalg.norm(eeg_positions - truth_positions, axis=1)
    normalized = np.clip(errors / error_scale, 0.0, 1.0)
    colors = [_error_color(value) for value in normalized]
    points = _points(eeg_positions, anchor, mirror_x)

    for start, end in UPPER_BODY_BONES:
        alpha = float(0.25 + 0.55 * max(normalized[start], normalized[end]))
        _draw_gradient_line(
            frame,
            cv2,
            points[start],
            points[end],
            colors[start],
            colors[end],
            alpha=alpha,
            thickness=3,
        )
    for point, color, value in zip(points, colors, normalized):
        overlay = frame.copy()
        cv2.circle(
            overlay,
            point,
            radius=5,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        alpha = float(0.3 + 0.55 * value)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def _draw_readiness_bar(
    frame: Any,
    cv2: ModuleType,
    status: CalibrationDisplayStatus,
) -> None:
    height, width = frame.shape[:2]
    margin = 18
    bar_width = min(360, max(160, width - 2 * margin))
    bar_height = 18
    x0 = margin
    y0 = margin
    x1 = x0 + bar_width
    y1 = y0 + bar_height
    score = float(np.clip(status.readiness_score, 0.0, 1.0))
    fill_x = round(x0 + bar_width * score)
    fill_color = (40, 210, 60) if status.ready else (0, 180, 255)
    if score < 0.5:
        fill_color = (0, 70, 255)

    cv2.rectangle(frame, (x0, y0), (x1, y1), color=(45, 45, 45), thickness=-1)
    cv2.rectangle(frame, (x0, y0), (fill_x, y1), color=fill_color, thickness=-1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color=(235, 235, 235), thickness=1)

    loss_text = "" if status.latest_loss is None else f" loss {status.latest_loss:.3f}"
    text = (
        f"readiness {score:.2f} "
        f"{'READY' if status.ready else 'learning'} "
        f"trusted {status.trusted_samples} "
        f"updates {status.update_count}"
        f"{loss_text}"
    )
    cv2.putText(
        frame,
        text,
        (x0, y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def _draw_gradient_line(
    frame: Any,
    cv2: ModuleType,
    start: tuple[int, int],
    end: tuple[int, int],
    start_color: tuple[int, int, int],
    end_color: tuple[int, int, int],
    *,
    alpha: float,
    thickness: int,
    segments: int = 12,
) -> None:
    overlay = frame.copy()
    for index in range(segments):
        a = index / segments
        b = (index + 1) / segments
        p0 = _interpolate_point(start, end, a)
        p1 = _interpolate_point(start, end, b)
        color = _interpolate_color(start_color, end_color, (a + b) / 2.0)
        cv2.line(
            overlay,
            p0,
            p1,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def _error_color(value: float) -> tuple[int, int, int]:
    low = np.asarray([255, 220, 40], dtype=np.float32)
    high = np.asarray([0, 0, 255], dtype=np.float32)
    color = low * (1.0 - value) + high * value
    return tuple(int(channel) for channel in color)


def _interpolate_color(
    start_color: tuple[int, int, int],
    end_color: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    start = np.asarray(start_color, dtype=np.float32)
    end = np.asarray(end_color, dtype=np.float32)
    color = start * (1.0 - amount) + end * amount
    return tuple(int(channel) for channel in color)


def _interpolate_point(
    start: tuple[int, int],
    end: tuple[int, int],
    amount: float,
) -> tuple[int, int]:
    return (
        round(start[0] * (1.0 - amount) + end[0] * amount),
        round(start[1] * (1.0 - amount) + end[1] * amount),
    )


def _smooth_phase(elapsed_s: float, duration_s: float) -> float:
    cycle = _cycle(elapsed_s, duration_s)
    return float(0.5 - 0.5 * np.cos(2.0 * np.pi * cycle))


def _cycle(elapsed_s: float, duration_s: float) -> float:
    return float((elapsed_s % max(duration_s, 1e-6)) / max(duration_s, 1e-6))


def _block_by_name(name: str) -> CalibrationMovementBlock:
    for block in DEFAULT_CALIBRATION_BLOCKS:
        if block.name == name:
            return block
    raise AssertionError(f"Unknown calibration movement block: {name}")
