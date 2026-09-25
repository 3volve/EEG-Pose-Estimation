from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from threading import Lock
from types import ModuleType
from typing import Any, Literal

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
REGION_BONES: dict[str, tuple[tuple[int, int], ...]] = {
    "left_arm": ((0, 2), (2, 4)),
    "right_arm": ((1, 3), (3, 5)),
    "arms": ((0, 2), (2, 4), (1, 3), (3, 5)),
    "shoulders_core": ((0, 1), (0, 6), (1, 7), (6, 7)),
    "hips": ((6, 7),),
}
_POSITION_DIM = len(UPPER_BODY_NAMES) * 3
ProfileBuildRole = Literal["support", "query", "validation", "test"]
NeutralRestPhase = Literal["calibrating", "waiting", "ready"]
PROFILE_REST_HOLD_DURATION_S = 2.0
PROFILE_REST_VELOCITY_THRESHOLD = 0.35
PROFILE_REST_VELOCITY_WINDOW_S = 0.35
PROFILE_REST_VIOLATION_GRACE_S = 0.2
PROFILE_REST_WRIST_TOLERANCE = 0.3
PROFILE_MIN_DIRECTION_SCORE = 0.2
PROFILE_MOVEMENT_TITLE_DELAY_S = 0.75
PROFILE_MOVEMENT_TITLE_FADE_S = 1.5
PROFILE_MOVEMENT_TITLES: dict[str, str] = {
    "neutral_rest": "HOLD YOUR REST POSE",
    "left_arm_raise": "LEFT ARM OUT TO SIDE",
    "right_arm_raise": "RIGHT ARM OUT TO SIDE",
    "left_elbow_bend": "CURL LEFT FOREARM",
    "right_elbow_bend": "CURL RIGHT FOREARM",
    "left_forward_reach": "LEFT ARM TOWARD CAMERA",
    "right_forward_reach": "RIGHT ARM TOWARD CAMERA",
    "both_arms_raise": "BOTH ARMS OUT TO SIDES",
    "arms_open_close": "BOTH ARMS OUT - TWICE",
    "torso_side_lean": "LEAN SIDE TO SIDE",
}


@dataclass(frozen=True, slots=True)
class CalibrationMovementBlock:
    name: str
    region: str
    duration_s: float
    rest: bool = False
    role: ProfileBuildRole | None = None


@dataclass(frozen=True, slots=True)
class ProfileBlockResult:
    block_id: int
    movement_name: str
    repeat_index: int
    role: ProfileBuildRole | None
    is_rest: bool
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


@dataclass(frozen=True, slots=True)
class NeutralRestStatus:
    phase: NeutralRestPhase
    wrist_positions: NDArray[np.float32] | None
    circle_scale: float
    condition_satisfied: bool
    wrist_tolerance: float


class NeutralRestGate:
    def __init__(
        self,
        *,
        hold_duration_s: float = PROFILE_REST_HOLD_DURATION_S,
        velocity_threshold: float = PROFILE_REST_VELOCITY_THRESHOLD,
        velocity_window_s: float = PROFILE_REST_VELOCITY_WINDOW_S,
        violation_grace_s: float = PROFILE_REST_VIOLATION_GRACE_S,
        wrist_tolerance: float = PROFILE_REST_WRIST_TOLERANCE,
    ) -> None:
        if hold_duration_s <= 0.0:
            raise ValueError("hold_duration_s must be positive")
        if velocity_threshold <= 0.0:
            raise ValueError("velocity_threshold must be positive")
        if velocity_window_s <= 0.0:
            raise ValueError("velocity_window_s must be positive")
        if violation_grace_s < 0.0:
            raise ValueError("violation_grace_s cannot be negative")
        if wrist_tolerance <= 0.0:
            raise ValueError("wrist_tolerance must be positive")
        self.hold_duration_s = hold_duration_s
        self.velocity_threshold = velocity_threshold
        self.velocity_window_s = velocity_window_s
        self.violation_grace_s = violation_grace_s
        self.wrist_tolerance = wrist_tolerance
        self._neutral_wrists: NDArray[np.float32] | None = None
        self._candidate_wrists: NDArray[np.float32] | None = None
        self._condition_started_at: float | None = None
        self._violation_started_at: float | None = None
        self._velocity_samples: deque[tuple[float, float]] = deque()
        self._phase: NeutralRestPhase = "calibrating"
        self._circle_scale = 0.0
        self._condition_satisfied = False

    @property
    def neutral_wrists(self) -> NDArray[np.float32] | None:
        return (
            None
            if self._neutral_wrists is None
            else self._neutral_wrists.copy()
        )

    @property
    def status(self) -> NeutralRestStatus:
        displayed_wrists = (
            self._candidate_wrists
            if self._phase == "calibrating"
            else self._neutral_wrists
        )
        return NeutralRestStatus(
            phase=self._phase,
            wrist_positions=(
                None if displayed_wrists is None else displayed_wrists.copy()
            ),
            circle_scale=self._circle_scale,
            condition_satisfied=self._condition_satisfied,
            wrist_tolerance=self.wrist_tolerance,
        )

    def begin_return_to_rest(self) -> None:
        assert self._neutral_wrists is not None
        self._phase = "waiting"
        self._condition_started_at = None
        self._violation_started_at = None
        self._velocity_samples.clear()
        self._circle_scale = 1.0
        self._condition_satisfied = False

    def update(
        self,
        feature_vector: NDArray[np.float32],
        now_s: float,
    ) -> bool:
        positions, velocities = _positions_and_velocities(feature_vector)
        wrists = positions[[4, 5]]
        mean_velocity = float(np.linalg.norm(velocities, axis=1).mean())
        self._velocity_samples.append((now_s, mean_velocity))
        oldest_time_s = now_s - self.velocity_window_s
        while self._velocity_samples[0][0] < oldest_time_s:
            self._velocity_samples.popleft()
        robust_velocity = float(
            np.median([value for _, value in self._velocity_samples])
        )
        still = robust_velocity <= self.velocity_threshold

        if self._neutral_wrists is None:
            self._update_calibration(wrists, still, now_s)
        else:
            self._update_return_to_rest(wrists, still, now_s)
        return self._phase == "ready"

    def _update_calibration(
        self,
        wrists: NDArray[np.float32],
        still: bool,
        now_s: float,
    ) -> None:
        within_candidate = (
            self._candidate_wrists is not None
            and _wrists_within_target(
                wrists,
                self._candidate_wrists,
                self.wrist_tolerance,
            )
        )
        satisfied = still and (
            self._candidate_wrists is None or within_candidate
        )
        if not self._condition_holds_with_grace(satisfied, now_s):
            self._candidate_wrists = None
            self._condition_started_at = None
            self._circle_scale = 0.0
            self._condition_satisfied = False
            return
        if self._candidate_wrists is None:
            self._candidate_wrists = wrists.copy()
            self._condition_started_at = now_s
        assert self._condition_started_at is not None
        progress = np.clip(
            (now_s - self._condition_started_at) / self.hold_duration_s,
            0.0,
            1.0,
        )
        self._circle_scale = float(progress)
        self._condition_satisfied = True
        if progress >= 1.0:
            self._neutral_wrists = self._candidate_wrists.copy()
            self._phase = "ready"

    def _update_return_to_rest(
        self,
        wrists: NDArray[np.float32],
        still: bool,
        now_s: float,
    ) -> None:
        assert self._neutral_wrists is not None
        satisfied = still and _wrists_within_target(
            wrists,
            self._neutral_wrists,
            self.wrist_tolerance,
        )
        if not self._condition_holds_with_grace(satisfied, now_s):
            self._condition_started_at = None
            self._circle_scale = 1.0
            self._condition_satisfied = False
            return
        if self._condition_started_at is None:
            self._condition_started_at = now_s
        progress = np.clip(
            (now_s - self._condition_started_at) / self.hold_duration_s,
            0.0,
            1.0,
        )
        self._circle_scale = float(1.0 - progress)
        self._condition_satisfied = True
        if progress >= 1.0:
            self._phase = "ready"

    def _condition_holds_with_grace(
        self,
        satisfied: bool,
        now_s: float,
    ) -> bool:
        if satisfied:
            self._violation_started_at = None
            return True
        if self._condition_started_at is None:
            return False
        if self._violation_started_at is None:
            self._violation_started_at = now_s
        return now_s - self._violation_started_at <= self.violation_grace_s


def _positions_and_velocities(
    feature_vector: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    values = np.asarray(feature_vector, dtype=np.float32)
    expected_size = 2 * _POSITION_DIM
    if values.ndim != 1 or values.size < expected_size:
        raise ValueError(
            "Neutral rest detection requires a one-dimensional pose feature "
            f"vector with at least {expected_size} position-and-velocity values; "
            f"got shape {values.shape}."
        )
    return (
        values[:_POSITION_DIM].reshape(8, 3),
        values[_POSITION_DIM:expected_size].reshape(8, 3),
    )


def _wrists_within_target(
    wrists: NDArray[np.float32],
    target_wrists: NDArray[np.float32],
    tolerance: float,
) -> bool:
    distances = np.linalg.norm(wrists[:, :2] - target_wrists[:, :2], axis=1)
    return bool(np.all(distances <= tolerance))


DEFAULT_CALIBRATION_BLOCKS: tuple[CalibrationMovementBlock, ...] = (
    CalibrationMovementBlock("rest", "rest", 4.0, rest=True),
    CalibrationMovementBlock("left_arm_raise", "left_arm", 6.0),
    CalibrationMovementBlock("right_arm_raise", "right_arm", 6.0),
    CalibrationMovementBlock("elbow_bends", "arms", 6.0),
    CalibrationMovementBlock("wrist_forearm", "arms", 6.0),
    CalibrationMovementBlock("torso_side_lean", "shoulders_core", 6.0),
)

PROFILE_BUILD_ROLES: tuple[ProfileBuildRole, ...] = (
    "support",
    "query",
    "validation",
    "test",
)
PROFILE_BUILD_REPEATS: int = len(PROFILE_BUILD_ROLES)
PROFILE_BUILD_BLOCKS: tuple[CalibrationMovementBlock, ...] = (
    CalibrationMovementBlock("neutral_rest", "rest", 8.0, rest=True),
    CalibrationMovementBlock("left_arm_raise", "left_arm", 8.0),
    CalibrationMovementBlock("right_arm_raise", "right_arm", 8.0),
    CalibrationMovementBlock("left_elbow_bend", "left_arm", 8.0),
    CalibrationMovementBlock("right_elbow_bend", "right_arm", 8.0),
    CalibrationMovementBlock("left_forward_reach", "left_arm", 8.0),
    CalibrationMovementBlock("right_forward_reach", "right_arm", 8.0),
    CalibrationMovementBlock("both_arms_raise", "arms", 8.0),
    CalibrationMovementBlock("arms_open_close", "arms", 8.0),
    CalibrationMovementBlock("torso_side_lean", "shoulders_core", 8.0),
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
    return guide_positions_for_block(
        block_name,
        neutral_dummy_positions(),
        elapsed_s,
        duration_s,
    )


def guide_positions_for_block(
    block_name: str,
    start_positions: NDArray[np.float32],
    elapsed_s: float,
    duration_s: float,
) -> NDArray[np.float32]:
    assert start_positions.shape == (8, 3)
    positions = np.asarray(start_positions, dtype=np.float32).copy()
    phase = _smooth_phase(elapsed_s, duration_s)

    if block_name == "left_arm_raise":
        _pose_arm(
            positions,
            start_positions,
            shoulder=0,
            elbow=2,
            wrist=4,
            upper_target=(-1.0, -0.12, 0.0),
            forearm_target=(-1.0, -0.05, 0.0),
            amount=phase,
        )
    elif block_name == "right_arm_raise":
        _pose_arm(
            positions,
            start_positions,
            shoulder=1,
            elbow=3,
            wrist=5,
            upper_target=(1.0, -0.12, 0.0),
            forearm_target=(1.0, -0.05, 0.0),
            amount=phase,
        )
    elif block_name == "left_elbow_bend":
        _bend_elbow(positions, start_positions, shoulder=0, elbow=2, wrist=4, amount=phase)
    elif block_name == "right_elbow_bend":
        _bend_elbow(positions, start_positions, shoulder=1, elbow=3, wrist=5, amount=phase)
    elif block_name == "left_forward_reach":
        _pose_arm(
            positions,
            start_positions,
            shoulder=0,
            elbow=2,
            wrist=4,
            upper_target=(-0.12, 0.0, -1.0),
            forearm_target=(-0.05, 0.0, -1.0),
            amount=phase,
        )
    elif block_name == "right_forward_reach":
        _pose_arm(
            positions,
            start_positions,
            shoulder=1,
            elbow=3,
            wrist=5,
            upper_target=(0.12, 0.0, -1.0),
            forearm_target=(0.05, 0.0, -1.0),
            amount=phase,
        )
    elif block_name == "both_arms_raise":
        _pose_arm(
            positions,
            start_positions,
            shoulder=0,
            elbow=2,
            wrist=4,
            upper_target=(-1.0, -0.12, 0.0),
            forearm_target=(-1.0, -0.05, 0.0),
            amount=phase,
        )
        _pose_arm(
            positions,
            start_positions,
            shoulder=1,
            elbow=3,
            wrist=5,
            upper_target=(1.0, -0.12, 0.0),
            forearm_target=(1.0, -0.05, 0.0),
            amount=phase,
        )
    elif block_name == "arms_open_close":
        outward_amount = float(
            0.5 - 0.5 * np.cos(4.0 * np.pi * _cycle(elapsed_s, duration_s))
        )
        _pose_both_arms_out(positions, start_positions, outward_amount)
    elif block_name == "elbow_bends":
        _bend_elbow(positions, start_positions, shoulder=0, elbow=2, wrist=4, amount=phase)
        _bend_elbow(positions, start_positions, shoulder=1, elbow=3, wrist=5, amount=phase)
    elif block_name == "wrist_forearm":
        angle = np.deg2rad(45.0) * np.sin(4.0 * np.pi * _cycle(elapsed_s, duration_s))
        positions[4] = positions[2] + _rotate_z(start_positions[4] - start_positions[2], angle)
        positions[5] = positions[3] + _rotate_z(start_positions[5] - start_positions[3], -angle)
    elif block_name == "torso_side_lean":
        hip_center = start_positions[[6, 7]].mean(axis=0)
        angle = np.deg2rad(15.0) * np.sin(2.0 * np.pi * _cycle(elapsed_s, duration_s))
        positions[:6] = np.stack(
            [
                hip_center + _rotate_z(position - hip_center, angle)
                for position in start_positions[:6]
            ]
        )

    return positions.astype(np.float32, copy=False)


def _pose_arm(
    positions: NDArray[np.float32],
    start_positions: NDArray[np.float32],
    *,
    shoulder: int,
    elbow: int,
    wrist: int,
    upper_target: tuple[float, float, float],
    forearm_target: tuple[float, float, float],
    amount: float,
) -> None:
    if amount <= 0.0:
        return
    upper = start_positions[elbow] - start_positions[shoulder]
    forearm = start_positions[wrist] - start_positions[elbow]
    upper_length = float(np.linalg.norm(upper))
    forearm_length = float(np.linalg.norm(forearm))
    if upper_length <= 1e-6 or forearm_length <= 1e-6:
        return
    upper_direction = _slerp_direction(upper, np.asarray(upper_target), amount)
    forearm_direction = _slerp_direction(forearm, np.asarray(forearm_target), amount)
    positions[elbow] = positions[shoulder] + upper_length * upper_direction
    positions[wrist] = positions[elbow] + forearm_length * forearm_direction


def _bend_elbow(
    positions: NDArray[np.float32],
    start_positions: NDArray[np.float32],
    *,
    shoulder: int,
    elbow: int,
    wrist: int,
    amount: float,
) -> None:
    forearm = start_positions[wrist] - start_positions[elbow]
    forearm_length = float(np.linalg.norm(forearm))
    target = start_positions[shoulder] - start_positions[elbow]
    if forearm_length <= 1e-6 or float(np.linalg.norm(target)) <= 1e-6:
        return
    direction = _slerp_direction(forearm, target, amount)
    positions[wrist] = positions[elbow] + forearm_length * direction


def _pose_both_arms_out(
    positions: NDArray[np.float32],
    start_positions: NDArray[np.float32],
    amount: float,
) -> None:
    for shoulder, elbow, wrist, side in ((0, 2, 4, -1.0), (1, 3, 5, 1.0)):
        _pose_arm(
            positions,
            start_positions,
            shoulder=shoulder,
            elbow=elbow,
            wrist=wrist,
            upper_target=(side, 0.0, 0.0),
            forearm_target=(side, 0.0, 0.0),
            amount=amount,
        )


def _slerp_direction(
    start: NDArray[np.float32],
    target: NDArray[np.float32],
    amount: float,
) -> NDArray[np.float32]:
    start_direction = start / np.linalg.norm(start)
    target_direction = target / np.linalg.norm(target)
    dot = float(np.clip(np.dot(start_direction, target_direction), -1.0, 1.0))
    angle = float(np.arccos(dot))
    if angle <= 1e-5:
        return start_direction.astype(np.float32, copy=False)
    sin_angle = float(np.sin(angle))
    if abs(sin_angle) <= 1e-5:
        basis = (
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(float(start_direction[0])) < 0.9
            else np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        )
        perpendicular = np.cross(start_direction, basis)
        perpendicular /= np.linalg.norm(perpendicular)
        direction = (
            np.cos(np.pi * amount) * start_direction
            + np.sin(np.pi * amount) * perpendicular
        )
        return direction.astype(np.float32, copy=False)
    direction = (
        np.sin((1.0 - amount) * angle) / sin_angle * start_direction
        + np.sin(amount * angle) / sin_angle * target_direction
    )
    return (direction / np.linalg.norm(direction)).astype(np.float32, copy=False)


def _rotate_z(vector: NDArray[np.float32], angle: float) -> NDArray[np.float32]:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    x, y, z = vector
    return np.asarray(
        [cosine * x - sine * y, sine * x + cosine * y, z],
        dtype=np.float32,
    )


def profile_build_sequence(
    *,
    repeats: int = PROFILE_BUILD_REPEATS,
    seed: int | None = None,
) -> tuple[CalibrationMovementBlock, ...]:
    if not 1 <= repeats <= len(PROFILE_BUILD_ROLES):
        raise ValueError(
            f"repeats must be between 1 and {len(PROFILE_BUILD_ROLES)}, got {repeats}"
        )

    roles = PROFILE_BUILD_ROLES[:repeats]
    rng = np.random.default_rng(seed)
    sequence: list[CalibrationMovementBlock] = []
    seen_orders: set[tuple[str, ...]] = set()
    for role in roles:
        while True:
            indices = rng.permutation(len(PROFILE_BUILD_BLOCKS))
            order = tuple(PROFILE_BUILD_BLOCKS[index].name for index in indices)
            if order not in seen_orders:
                seen_orders.add(order)
                break
        sequence.extend(
            replace(PROFILE_BUILD_BLOCKS[index], role=role)
            for index in indices
        )
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
    elif not block.rest and direction_score < PROFILE_MIN_DIRECTION_SCORE:
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
        role=block.role,
        is_rest=block.rest,
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
        role=block.role,
        is_rest=block.rest,
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
    if block.name == "torso_side_lean":
        # Hip-centering makes global translation unobservable. The shoulder
        # midpoint moving relative to that fixed root is the side-lean signal.
        return (0, 1)
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
    if block_name == "left_arm_raise":
        return _upward_range_score(positions, (2, 4), reference_index=0)
    if block_name == "right_arm_raise":
        return _upward_range_score(positions, (3, 5), reference_index=1)
    if block_name == "both_arms_raise":
        return 0.5 * (
            _upward_range_score(positions, (2, 4), reference_index=0)
            + _upward_range_score(positions, (3, 5), reference_index=1)
        )
    if block_name == "left_elbow_bend":
        return _upward_range_score(positions, (4,), reference_index=2)
    if block_name == "right_elbow_bend":
        return _upward_range_score(positions, (5,), reference_index=3)
    if block_name == "left_forward_reach":
        return _relative_range_score(positions, 4, 0, axis=2) + _upward_range_score(
            positions,
            (4,),
            reference_index=0,
        )
    if block_name == "right_forward_reach":
        return _relative_range_score(positions, 5, 1, axis=2) + _upward_range_score(
            positions,
            (5,),
            reference_index=1,
        )
    if block_name == "arms_open_close":
        left_range = _relative_range_score(positions, 4, 0, axis=0)
        right_range = _relative_range_score(positions, 5, 1, axis=0)
        return min(left_range + right_range, 1.0)
    if block_name == "torso_side_lean":
        shoulder_center_x = positions[:, (0, 1), 0].mean(axis=1)
        return min(float(np.ptp(shoulder_center_x)), 1.0)
    if "rest" in block_name:
        return 1.0
    return 0.5


def _upward_range_score(
    positions: NDArray[np.float32],
    indices: tuple[int, ...],
    *,
    reference_index: int,
) -> float:
    relative_y = (
        positions[:, list(indices), 1]
        - positions[:, reference_index : reference_index + 1, 1]
    )
    starts = relative_y[0]
    highest = np.min(relative_y, axis=0)
    return float(max(0.0, np.mean(starts - highest)))


def _relative_range_score(
    positions: NDArray[np.float32],
    moving_index: int,
    reference_index: int,
    *,
    axis: int,
) -> float:
    relative = (
        positions[:, moving_index, axis]
        - positions[:, reference_index, axis]
    )
    return float(np.ptp(relative))


def _guide_distance(
    block_name: str,
    positions: NDArray[np.float32],
    start_time_s: float,
    end_time_s: float,
) -> float:
    if len(positions) == 0:
        return float("inf")
    duration = max(end_time_s - start_time_s, 1e-6)
    start_positions = positions[0]
    guide = np.stack(
        [
            guide_positions_for_block(
                block_name,
                start_positions,
                elapsed_s=duration * index / max(len(positions) - 1, 1),
                duration_s=duration,
            )
            for index in range(len(positions))
        ],
        axis=0,
    )
    guide_delta = guide - start_positions
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
    ):
        return _block_by_name("rest")

    region_values = {
        "left_arm": scores.left_arm,
        "right_arm": scores.right_arm,
        "shoulders_core": scores.shoulders_core,
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
        self._dummy_block_name: str | None = None
        self._dummy_block_rest: bool | None = None
        self._dummy_elapsed_s: float | None = None
        self._dummy_duration_s: float | None = None
        self._guide_origin_positions: NDArray[np.float32] | None = None
        self._guide_origin_pending = False
        self._guide_anchor: tuple[float, float, float] | None = None
        self._guide_generation = 0
        self._active_truth_landmarks: tuple[int, ...] = ()
        self._active_truth_bones: tuple[tuple[int, int], ...] = ()
        self._neutral_rest_status: NeutralRestStatus | None = None
        self._movement_title: str | None = None
        self._movement_title_alpha = 0.0
        self._status: CalibrationDisplayStatus | None = None

    def update_truth(self, feature_vector: NDArray[np.float32]) -> None:
        with self._lock:
            latest = positions_from_feature_vector(feature_vector)
            self._truth_positions = latest
            if self._guide_origin_pending:
                self._guide_origin_positions = latest.copy()
                self._guide_origin_pending = False

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
            starts_new_block = (
                self._dummy_block_name != block.name
                or (
                    self._dummy_elapsed_s is not None
                    and elapsed_s < self._dummy_elapsed_s
                )
            )
            if starts_new_block:
                (
                    self._active_truth_landmarks,
                    self._active_truth_bones,
                ) = _display_targets_for_block(block)
                starts_first_block = self._dummy_block_rest is None
                starts_after_rest = (
                    not block.rest and self._dummy_block_rest is True
                )
                if starts_first_block or starts_after_rest:
                    self._guide_origin_positions = (
                        None
                        if self._truth_positions is None
                        else self._truth_positions.copy()
                    )
                    self._guide_origin_pending = self._truth_positions is None
                    self._guide_anchor = None
                    self._guide_generation += 1
            self._dummy_positions = dummy_positions_for_block(
                block.name,
                elapsed_s,
                block.duration_s,
            )
            self._dummy_block_name = block.name
            self._dummy_block_rest = block.rest
            self._dummy_elapsed_s = elapsed_s
            self._dummy_duration_s = block.duration_s

    def update_status(self, status: CalibrationDisplayStatus) -> None:
        with self._lock:
            self._status = status

    def update_neutral_rest(self, status: NeutralRestStatus | None) -> None:
        with self._lock:
            self._neutral_rest_status = status

    def update_movement_title(
        self,
        block: CalibrationMovementBlock | None,
        alpha: float = 1.0,
    ) -> None:
        with self._lock:
            if block is None or alpha <= 0.0:
                self._movement_title = None
                self._movement_title_alpha = 0.0
                return
            self._movement_title = PROFILE_MOVEMENT_TITLES[block.name]
            self._movement_title_alpha = float(np.clip(alpha, 0.0, 1.0))

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
            dummy_block_name = self._dummy_block_name
            dummy_elapsed_s = self._dummy_elapsed_s
            dummy_duration_s = self._dummy_duration_s
            guide_origin = (
                None
                if self._guide_origin_positions is None
                else self._guide_origin_positions.copy()
            )
            guide_anchor = self._guide_anchor
            guide_generation = self._guide_generation
            active_truth_landmarks = self._active_truth_landmarks
            active_truth_bones = self._active_truth_bones
            neutral_rest_status = self._neutral_rest_status
            movement_title = self._movement_title
            movement_title_alpha = self._movement_title_alpha
            status = self._status

        truth_anchor = _display_anchor(frame, pose_result, mirror_x, truth)
        if guide_anchor is None and guide_origin is not None:
            guide_anchor = _display_anchor(
                frame,
                pose_result,
                mirror_x,
                guide_origin,
            )
            with self._lock:
                if self._guide_generation == guide_generation:
                    self._guide_anchor = guide_anchor
        if dummy is not None and guide_origin is not None:
            assert guide_anchor is not None
            assert dummy_block_name is not None
            assert dummy_elapsed_s is not None
            assert dummy_duration_s is not None
            dummy = guide_positions_for_block(
                dummy_block_name,
                guide_origin,
                dummy_elapsed_s,
                dummy_duration_s,
            )
            _draw_plain_skeleton(
                frame,
                cv2,
                dummy,
                guide_anchor,
                mirror_x=mirror_x,
                color=(255, 40, 230),
                alpha=0.7,
                thickness=4,
                draw_aligned_head=dummy_block_name == "torso_side_lean",
            )
        if truth is not None:
            _draw_selective_skeleton(
                frame,
                cv2,
                truth,
                truth_anchor,
                mirror_x=mirror_x,
                color=(40, 230, 40),
                alpha=0.9,
                active_alpha=0.35,
                thickness=3,
                active_landmarks=active_truth_landmarks,
                active_bones=active_truth_bones,
            )
        if neutral_rest_status is not None:
            rest_anchor = _display_anchor(
                frame,
                pose_result,
                mirror_x,
            )
            _draw_neutral_rest_circles(
                frame,
                cv2,
                neutral_rest_status,
                rest_anchor,
                mirror_x=mirror_x,
            )
        if truth is not None and eeg is not None:
            _draw_error_skeleton(
                frame,
                cv2,
                eeg,
                truth,
                truth_anchor,
                mirror_x=mirror_x,
                error_scale=self.error_scale,
            )
        if movement_title is not None:
            _draw_movement_title(
                frame,
                cv2,
                movement_title,
                alpha=movement_title_alpha,
            )
        if status is not None:
            _draw_readiness_bar(frame, cv2, status)


def _display_targets_for_block(
    block: CalibrationMovementBlock,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    if block.rest:
        return (), ()
    return REGION_LANDMARKS.get(block.region, ()), REGION_BONES.get(block.region, ())


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
    draw_aligned_head: bool = False,
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
    if draw_aligned_head:
        neck_position, head_position = _aligned_head_guide_positions(positions)
        neck_point, head_point = _points(
            np.stack((neck_position, head_position)),
            anchor,
            mirror_x,
        )
        cv2.line(
            overlay,
            neck_point,
            head_point,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            overlay,
            head_point,
            radius=max(6, round(anchor[2] * 0.18)),
            color=color,
            thickness=max(2, thickness // 2),
            lineType=cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def _draw_selective_skeleton(
    frame: Any,
    cv2: ModuleType,
    positions: NDArray[np.float32],
    anchor: tuple[float, float, float],
    *,
    mirror_x: bool,
    color: tuple[int, int, int],
    alpha: float,
    active_alpha: float,
    thickness: int,
    active_landmarks: tuple[int, ...],
    active_bones: tuple[tuple[int, int], ...],
) -> None:
    points = _points(positions, anchor, mirror_x)
    active_landmark_set = set(active_landmarks)
    active_bone_set = set(active_bones)
    _draw_skeleton_elements(
        frame,
        cv2,
        points,
        bones=tuple(bone for bone in UPPER_BODY_BONES if bone not in active_bone_set),
        landmark_indices=tuple(
            index for index in range(len(points)) if index not in active_landmark_set
        ),
        color=color,
        alpha=alpha,
        thickness=thickness,
    )
    _draw_skeleton_elements(
        frame,
        cv2,
        points,
        bones=tuple(bone for bone in UPPER_BODY_BONES if bone in active_bone_set),
        landmark_indices=tuple(
            index for index in range(len(points)) if index in active_landmark_set
        ),
        color=color,
        alpha=active_alpha,
        thickness=thickness,
    )


def _draw_skeleton_elements(
    frame: Any,
    cv2: ModuleType,
    points: list[tuple[int, int]],
    *,
    bones: tuple[tuple[int, int], ...],
    landmark_indices: tuple[int, ...],
    color: tuple[int, int, int],
    alpha: float,
    thickness: int,
) -> None:
    if not bones and not landmark_indices:
        return
    overlay = frame.copy()
    for start, end in bones:
        cv2.line(
            overlay,
            points[start],
            points[end],
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    for index in landmark_indices:
        cv2.circle(
            overlay,
            points[index],
            radius=4,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def _draw_neutral_rest_circles(
    frame: Any,
    cv2: ModuleType,
    status: NeutralRestStatus,
    anchor: tuple[float, float, float],
    *,
    mirror_x: bool,
) -> None:
    if status.phase == "calibrating":
        prompt = (
            "Hold still to set your natural rest position"
            if status.condition_satisfied
            else "Stand naturally with your hands at rest"
        )
    elif status.condition_satisfied:
        prompt = "Hold still for the next movement"
    else:
        prompt = "Return your hands to the rest circles"
    cv2.putText(
        frame,
        prompt,
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if status.wrist_positions is None or status.circle_scale <= 0.0:
        return
    if status.phase == "calibrating":
        color = (255, 200, 40)
    elif status.condition_satisfied:
        color = (40, 230, 40)
    else:
        color = (40, 150, 255)
    radius = max(
        1,
        round(anchor[2] * status.wrist_tolerance * status.circle_scale),
    )
    overlay = frame.copy()
    for point in _points(status.wrist_positions, anchor, mirror_x):
        cv2.circle(
            overlay,
            point,
            radius=radius,
            color=color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0.0, frame)


def _draw_movement_title(
    frame: Any,
    cv2: ModuleType,
    title: str,
    *,
    alpha: float,
) -> None:
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(1.5, max(0.8, width / 700.0))
    thickness = max(2, round(2.5 * font_scale))
    (text_width, text_height), baseline = cv2.getTextSize(
        title,
        font,
        font_scale,
        thickness,
    )
    maximum_width = round(width * 0.9)
    if text_width > maximum_width:
        font_scale *= maximum_width / text_width
        thickness = max(2, round(2.5 * font_scale))
        (text_width, text_height), baseline = cv2.getTextSize(
            title,
            font,
            font_scale,
            thickness,
        )

    center_y = height // 2
    text_x = (width - text_width) // 2
    text_y = center_y + text_height // 2
    horizontal_padding = max(16, round(18 * font_scale))
    vertical_padding = max(12, round(14 * font_scale))
    top_left = (
        max(0, text_x - horizontal_padding),
        max(0, text_y - text_height - vertical_padding),
    )
    bottom_right = (
        min(width - 1, text_x + text_width + horizontal_padding),
        min(height - 1, text_y + baseline + vertical_padding),
    )

    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (20, 20, 20), -1)
    cv2.rectangle(overlay, top_left, bottom_right, (255, 80, 230), 3)
    cv2.putText(
        overlay,
        title,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def _aligned_head_guide_positions(
    positions: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Place the guide neck and head along the current hip-to-shoulder axis."""
    hip_center = positions[[6, 7]].mean(axis=0)
    shoulder_center = positions[[0, 1]].mean(axis=0)
    torso_axis = shoulder_center - hip_center
    neck_position = shoulder_center + 0.12 * torso_axis
    head_position = shoulder_center + 0.34 * torso_axis
    return (
        neck_position.astype(np.float32, copy=False),
        head_position.astype(np.float32, copy=False),
    )


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
