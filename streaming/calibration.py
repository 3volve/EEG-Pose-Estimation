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
class RegionScores:
    left_arm: float
    right_arm: float
    shoulders_core: float
    hips: float
    rest_false_positive: float
    movement_response: float


DEFAULT_CALIBRATION_BLOCKS: tuple[CalibrationMovementBlock, ...] = (
    CalibrationMovementBlock("rest", "rest", 4.0, rest=True),
    CalibrationMovementBlock("left_arm_raise", "left_arm", 6.0),
    CalibrationMovementBlock("right_arm_raise", "right_arm", 6.0),
    CalibrationMovementBlock("elbow_bends", "arms", 6.0),
    CalibrationMovementBlock("wrist_forearm", "arms", 6.0),
    CalibrationMovementBlock("torso_shift", "shoulders_core", 6.0),
    CalibrationMovementBlock("hip_shift", "hips", 6.0),
)


def positions_from_feature_vector(
    feature_vector: NDArray[np.float32],
) -> NDArray[np.float32]:
    return np.asarray(feature_vector[:_POSITION_DIM], dtype=np.float32).reshape(8, 3)


def neutral_dummy_positions() -> NDArray[np.float32]:
    return np.asarray(
        [
            [-0.5, -1.0, 0.0],
            [0.5, -1.0, 0.0],
            [-0.85, -0.45, 0.0],
            [0.85, -0.45, 0.0],
            [-0.95, 0.15, 0.0],
            [0.95, 0.15, 0.0],
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
        positions[2] += np.asarray([-0.15, -0.65 * phase, 0.0], dtype=np.float32)
        positions[4] += np.asarray([-0.1, -1.25 * phase, 0.0], dtype=np.float32)
    elif block_name == "right_arm_raise":
        positions[3] += np.asarray([0.15, -0.65 * phase, 0.0], dtype=np.float32)
        positions[5] += np.asarray([0.1, -1.25 * phase, 0.0], dtype=np.float32)
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
    def __init__(self, *, error_scale: float = 0.75) -> None:
        self.error_scale = error_scale
        self._lock = Lock()
        self._truth_positions: NDArray[np.float32] | None = None
        self._eeg_positions: NDArray[np.float32] | None = None
        self._dummy_positions: NDArray[np.float32] | None = None

    def update_truth(self, feature_vector: NDArray[np.float32]) -> None:
        with self._lock:
            self._truth_positions = positions_from_feature_vector(feature_vector)

    def update_eeg(self, decoded_feature_vector: NDArray[np.float32]) -> None:
        with self._lock:
            self._eeg_positions = positions_from_feature_vector(decoded_feature_vector)

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

        anchor = _display_anchor(frame, pose_result, mirror_x)
        if dummy is not None:
            _draw_plain_skeleton(
                frame,
                cv2,
                dummy,
                anchor,
                color=(255, 180, 40),
                alpha=0.35,
                thickness=2,
            )
        if truth is not None:
            _draw_plain_skeleton(
                frame,
                cv2,
                truth,
                anchor,
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
                error_scale=self.error_scale,
            )


def _display_anchor(
    frame: Any,
    pose_result: PoseResult | None,
    mirror_x: bool,
) -> tuple[float, float, float]:
    height, width = frame.shape[:2]
    default_scale = min(width, height) * 0.16
    if pose_result is None or not pose_result.pose_detected or not pose_result.landmarks:
        return width * 0.5, height * 0.68, default_scale

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


def _points(
    positions: NDArray[np.float32],
    anchor: tuple[float, float, float],
) -> list[tuple[int, int]]:
    root_x, root_y, scale = anchor
    return [
        (round(root_x + float(position[0]) * scale), round(root_y + float(position[1]) * scale))
        for position in positions
    ]


def _draw_plain_skeleton(
    frame: Any,
    cv2: ModuleType,
    positions: NDArray[np.float32],
    anchor: tuple[float, float, float],
    *,
    color: tuple[int, int, int],
    alpha: float,
    thickness: int,
) -> None:
    overlay = frame.copy()
    points = _points(positions, anchor)
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
    error_scale: float,
) -> None:
    errors = np.linalg.norm(eeg_positions - truth_positions, axis=1)
    normalized = np.clip(errors / error_scale, 0.0, 1.0)
    colors = [_error_color(value) for value in normalized]
    points = _points(eeg_positions, anchor)

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
