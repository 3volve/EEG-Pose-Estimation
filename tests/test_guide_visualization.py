from __future__ import annotations

import numpy as np

from streaming import (
    CalibrationMovementBlock,
    CalibrationOverlayState,
    NeutralRestGate,
    NeutralRestStatus,
    PROFILE_MOVEMENT_TITLES,
    neutral_dummy_positions,
)
from streaming.calibration import _draw_neutral_rest_circles, guide_positions_for_block
from streaming.pose import PoseLandmark, PoseResult


class RecordingCv2:
    LINE_AA = 16
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self) -> None:
        self.lines: list[tuple[tuple, dict]] = []
        self.circles: list[tuple[tuple, dict]] = []
        self.text: list[tuple[tuple, dict]] = []
        self.rectangles: list[tuple[tuple, dict]] = []
        self.weights: list[float] = []

    def line(self, *args, **kwargs) -> None:
        self.lines.append((args, kwargs))

    def circle(self, *args, **kwargs) -> None:
        self.circles.append((args, kwargs))

    def putText(self, *args, **kwargs) -> None:
        self.text.append((args, kwargs))

    def rectangle(self, *args, **kwargs) -> None:
        self.rectangles.append((args, kwargs))

    @staticmethod
    def getTextSize(text, _font, font_scale, _thickness):
        return ((round(len(text) * 12 * font_scale), round(20 * font_scale)), 4)

    def addWeighted(self, overlay, alpha, frame, beta, gamma, dst) -> None:
        self.weights.append(alpha)
        dst[:] = overlay


def _feature_vector(positions: np.ndarray) -> np.ndarray:
    vector = np.zeros(48, dtype=np.float32)
    vector[:24] = positions.reshape(-1)
    return vector


def _rest_feature_vector(
    positions: np.ndarray,
    *,
    velocity: float = 0.0,
) -> np.ndarray:
    vector = _feature_vector(positions)
    vector[24:] = velocity
    return vector


def _pose_result(positions: np.ndarray, *, offset_x: float) -> PoseResult:
    image_points = positions[:, :2] * 30.0 + np.asarray([offset_x, 100.0])
    landmarks = [PoseLandmark(None, None, None, None, None) for _ in range(25)]
    for point, landmark_index in zip(
        image_points,
        (11, 12, 13, 14, 15, 16, 23, 24),
    ):
        landmarks[landmark_index] = PoseLandmark(
            float(point[0] / 240.0),
            float(point[1] / 240.0),
            0.0,
            1.0,
            1.0,
        )
    return PoseResult(1, 1.0, 240, 240, landmarks, [], True)


def test_guide_recaptures_only_when_movement_starts_after_rest() -> None:
    overlay = CalibrationOverlayState()
    first_block = CalibrationMovementBlock("left_arm_raise", "left_arm", 8.0)
    direct_block = CalibrationMovementBlock("right_arm_raise", "right_arm", 8.0)
    rest_block = CalibrationMovementBlock("rest", "rest", 4.0, rest=True)
    next_block = CalibrationMovementBlock("left_elbow_bend", "left_arm", 8.0)
    first_pose = neutral_dummy_positions()
    pose_at_rest = first_pose.copy()
    pose_at_rest[4, 0] += 0.5
    pose_after_rest = pose_at_rest.copy()
    pose_after_rest[5, 0] -= 0.4

    overlay.update_truth(_feature_vector(first_pose))
    overlay.update_dummy(first_block, 0.0)
    initial_generation = overlay._guide_generation

    overlay.update_truth(_feature_vector(pose_at_rest))
    overlay.update_dummy(direct_block, 0.0)

    np.testing.assert_allclose(overlay._guide_origin_positions, first_pose)
    assert overlay._guide_generation == initial_generation

    overlay.update_dummy(rest_block, 0.0)

    np.testing.assert_allclose(overlay._guide_origin_positions, first_pose)
    assert overlay._guide_generation == initial_generation

    overlay.update_truth(_feature_vector(pose_after_rest))
    overlay.update_dummy(next_block, 0.0)

    np.testing.assert_allclose(overlay._guide_origin_positions, pose_after_rest)
    assert overlay._guide_generation == initial_generation + 1


def test_guide_display_anchor_stays_fixed_during_movement() -> None:
    overlay = CalibrationOverlayState()
    block = CalibrationMovementBlock("left_arm_raise", "left_arm", 8.0)
    positions = neutral_dummy_positions()
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    cv2 = RecordingCv2()

    overlay.update_truth(_feature_vector(positions))
    overlay.update_dummy(block, 0.0)
    overlay.render(
        frame,
        cv2,
        pose_result=_pose_result(positions, offset_x=90.0),
        mirror_x=False,
    )
    initial_anchor = overlay._guide_anchor

    overlay.update_truth(_feature_vector(positions))
    overlay.update_dummy(block, 2.0)
    overlay.render(
        frame,
        cv2,
        pose_result=_pose_result(positions, offset_x=140.0),
        mirror_x=False,
    )

    assert overlay._guide_anchor == initial_anchor


def test_active_user_limb_is_translucent_above_guide() -> None:
    overlay = CalibrationOverlayState()
    positions = neutral_dummy_positions()
    overlay.update_truth(_feature_vector(positions))
    overlay.update_dummy(
        CalibrationMovementBlock("left_arm_raise", "left_arm", 8.0),
        2.0,
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2 = RecordingCv2()

    overlay.render(frame, cv2, pose_result=None, mirror_x=False)

    assert cv2.lines[0][1]["color"] == (255, 40, 230)
    assert cv2.weights == [0.7, 0.9, 0.35]


def test_rest_keeps_the_entire_user_pose_opaque() -> None:
    overlay = CalibrationOverlayState()
    positions = neutral_dummy_positions()
    overlay.update_truth(_feature_vector(positions))
    overlay.update_dummy(
        CalibrationMovementBlock("rest", "rest", 4.0, rest=True),
        1.0,
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2 = RecordingCv2()

    overlay.render(frame, cv2, pose_result=None, mirror_x=False)

    assert cv2.weights == [0.7, 0.9]


def test_standalone_arm_raise_is_full_sized_and_preserves_limb_lengths() -> None:
    start = neutral_dummy_positions()
    raised = guide_positions_for_block("left_arm_raise", start, 4.0, 8.0)
    returned = guide_positions_for_block("left_arm_raise", start, 8.0, 8.0)

    np.testing.assert_allclose(
        np.linalg.norm(raised[2] - raised[0]),
        np.linalg.norm(start[2] - start[0]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.linalg.norm(raised[4] - raised[2]),
        np.linalg.norm(start[4] - start[2]),
        atol=1e-6,
    )
    np.testing.assert_allclose(raised[[1, 3, 5, 6, 7]], start[[1, 3, 5, 6, 7]])
    assert raised[4, 1] < start[4, 1] - 1.2
    assert abs(float(raised[4, 1] - raised[0, 1])) < 0.2
    np.testing.assert_allclose(returned, start, atol=1e-6)


def test_all_articulated_arm_guides_preserve_captured_segment_lengths() -> None:
    start = neutral_dummy_positions()
    movement_times = {
        "left_arm_raise": 4.0,
        "right_arm_raise": 4.0,
        "left_elbow_bend": 4.0,
        "right_elbow_bend": 4.0,
        "left_forward_reach": 4.0,
        "right_forward_reach": 4.0,
        "both_arms_raise": 4.0,
        "arms_open_close": 2.0,
        "elbow_bends": 4.0,
        "wrist_forearm": 1.0,
    }
    expected_lengths = [
        np.linalg.norm(start[end] - start[begin])
        for begin, end in ((0, 2), (2, 4), (1, 3), (3, 5))
    ]

    for movement, elapsed_s in movement_times.items():
        guide = guide_positions_for_block(movement, start, elapsed_s, 8.0)
        actual_lengths = [
            np.linalg.norm(guide[end] - guide[begin])
            for begin, end in ((0, 2), (2, 4), (1, 3), (3, 5))
        ]
        np.testing.assert_allclose(actual_lengths, expected_lengths, atol=1e-6)
        assert np.max(np.linalg.norm(guide - start, axis=1)) > 0.25


def test_simultaneous_outward_arm_guide_never_crosses_the_torso() -> None:
    start = neutral_dummy_positions()

    for elapsed_s in np.linspace(0.0, 8.0, 33):
        guide = guide_positions_for_block(
            "arms_open_close",
            start,
            float(elapsed_s),
            8.0,
        )
        assert guide[4, 0] < 0.0
        assert guide[5, 0] > 0.0

    first_extension = guide_positions_for_block("arms_open_close", start, 2.0, 8.0)
    returned = guide_positions_for_block("arms_open_close", start, 4.0, 8.0)
    second_extension = guide_positions_for_block("arms_open_close", start, 6.0, 8.0)
    assert first_extension[4, 0] < start[4, 0] - 0.5
    np.testing.assert_allclose(returned, start, atol=1e-6)
    np.testing.assert_allclose(second_extension, first_extension, atol=1e-6)


def test_large_centered_movement_title_uses_camera_relative_wording() -> None:
    overlay = CalibrationOverlayState()
    positions = neutral_dummy_positions()
    block = CalibrationMovementBlock("left_forward_reach", "left_arm", 8.0)
    overlay.update_truth(_feature_vector(positions))
    overlay.update_dummy(block, 0.0)
    overlay.update_movement_title(block, alpha=0.5)
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    cv2 = RecordingCv2()

    overlay.render(frame, cv2, pose_result=None, mirror_x=False)

    assert PROFILE_MOVEMENT_TITLES[block.name] == "LEFT ARM TOWARD CAMERA"
    assert cv2.text[-1][0][1] == "LEFT ARM TOWARD CAMERA"
    assert len(cv2.rectangles) == 2
    assert cv2.weights[-1] == 0.5


def test_forward_reach_is_a_full_depth_movement_from_captured_pose() -> None:
    start = neutral_dummy_positions()
    reached = guide_positions_for_block("left_forward_reach", start, 4.0, 8.0)

    assert reached[4, 2] < start[4, 2] - 1.0
    assert reached[4, 1] < start[4, 1] - 1.0
    np.testing.assert_allclose(reached[[1, 3, 5, 6, 7]], start[[1, 3, 5, 6, 7]])


def test_torso_lean_rotates_upper_body_without_moving_hips() -> None:
    start = neutral_dummy_positions()
    leaned = guide_positions_for_block("torso_side_lean", start, 2.0, 8.0)

    np.testing.assert_allclose(leaned[[6, 7]], start[[6, 7]])
    assert leaned[[0, 1], 0].mean() > start[[0, 1], 0].mean() + 0.25
    for first, second in ((0, 1), (0, 2), (2, 4), (1, 3), (3, 5)):
        np.testing.assert_allclose(
            np.linalg.norm(leaned[second] - leaned[first]),
            np.linalg.norm(start[second] - start[first]),
            atol=1e-6,
        )


def test_neutral_rest_gate_grows_then_shrinks_over_two_seconds() -> None:
    gate = NeutralRestGate(
        hold_duration_s=2.0,
        velocity_threshold=0.08,
        wrist_tolerance=0.3,
    )
    neutral = neutral_dummy_positions()
    still = _rest_feature_vector(neutral)

    assert gate.update(still, 10.0) is False
    assert gate.status.phase == "calibrating"
    assert gate.status.circle_scale == 0.0
    assert gate.update(still, 11.0) is False
    assert gate.status.circle_scale == 0.5
    assert gate.update(still, 12.0) is True
    assert gate.status.circle_scale == 1.0
    np.testing.assert_allclose(gate.neutral_wrists, neutral[[4, 5]])

    gate.begin_return_to_rest()
    assert gate.status.circle_scale == 1.0
    assert gate.update(still, 20.0) is False
    assert gate.update(still, 21.0) is False
    assert gate.status.circle_scale == 0.5
    assert gate.update(still, 22.0) is True
    assert gate.status.circle_scale == 0.0


def test_neutral_rest_gate_resets_on_motion_or_leaving_wrist_targets() -> None:
    gate = NeutralRestGate()
    neutral = neutral_dummy_positions()
    still = _rest_feature_vector(neutral)
    fast = _rest_feature_vector(neutral, velocity=0.8)

    gate.update(still, 0.0)
    gate.update(still, 1.0)
    gate.update(fast, 1.5)
    gate.update(fast, 1.8)
    assert gate.status.circle_scale == 0.0
    assert gate.status.wrist_positions is None

    gate.update(still, 2.0)
    gate.update(still, 2.2)
    moved = neutral.copy()
    moved[4, 0] += 0.5
    gate.update(_rest_feature_vector(moved), 2.5)
    gate.update(_rest_feature_vector(moved), 2.8)
    assert gate.status.circle_scale == 0.0
    assert gate.update(still, 3.0) is False
    assert gate.update(still, 5.0) is True

    gate.begin_return_to_rest()
    gate.update(still, 6.0)
    gate.update(fast, 7.0)
    gate.update(fast, 7.3)
    assert gate.status.circle_scale == 1.0
    assert gate.status.condition_satisfied is False
    gate.update(still, 8.0)
    gate.update(_rest_feature_vector(moved), 9.0)
    gate.update(_rest_feature_vector(moved), 9.3)
    assert gate.status.circle_scale == 1.0


def test_neutral_rest_gate_ignores_brief_velocity_jitter() -> None:
    gate = NeutralRestGate()
    neutral = neutral_dummy_positions()
    still = _rest_feature_vector(neutral)
    one_frame_spike = _rest_feature_vector(neutral, velocity=1.0)

    assert gate.update(still, 10.0) is False
    assert gate.update(still, 11.0) is False
    gate.update(one_frame_spike, 11.1)
    assert gate.status.circle_scale > 0.5
    assert gate.update(still, 11.2) is False
    assert gate.update(still, 12.0) is True


def test_rest_feedback_circle_radius_uses_interpolated_scale() -> None:
    frame = np.zeros((200, 240, 3), dtype=np.uint8)
    cv2 = RecordingCv2()
    wrists = neutral_dummy_positions()[[4, 5]]
    status = NeutralRestStatus(
        phase="waiting",
        wrist_positions=wrists,
        circle_scale=0.5,
        condition_satisfied=True,
        wrist_tolerance=0.3,
    )

    _draw_neutral_rest_circles(
        frame,
        cv2,
        status,
        (120.0, 100.0, 100.0),
        mirror_x=False,
    )

    assert len(cv2.circles) == 2
    assert cv2.text[0][0][1] == "Hold still for the next movement"
    assert [circle[1]["radius"] for circle in cv2.circles] == [15, 15]
    assert all(circle[1]["color"] == (40, 230, 40) for circle in cv2.circles)


def test_initial_rest_prompt_appears_before_wrist_targets_exist() -> None:
    frame = np.zeros((200, 240, 3), dtype=np.uint8)
    cv2 = RecordingCv2()
    status = NeutralRestStatus(
        phase="calibrating",
        wrist_positions=None,
        circle_scale=0.0,
        condition_satisfied=False,
        wrist_tolerance=0.3,
    )

    _draw_neutral_rest_circles(
        frame,
        cv2,
        status,
        (120.0, 100.0, 100.0),
        mirror_x=False,
    )

    assert cv2.circles == []
    assert cv2.text[0][0][1] == "Stand naturally with your hands at rest"
