from types import SimpleNamespace

import pytest

from streaming.pose import AsyncPoseEstimator, PoseLandmark, PoseResult


def make_landmark(value: float) -> SimpleNamespace:
    return SimpleNamespace(
        x=value,
        y=value + 1,
        z=value + 2,
        visibility=value + 3,
        presence=value + 4,
    )


def test_missing_model_is_rejected() -> None:
    estimator = AsyncPoseEstimator("model-that-does-not-exist.task")

    with pytest.raises(FileNotFoundError, match="pose model not found"):
        estimator.start()


def test_callback_emits_empty_result_when_no_pose() -> None:
    estimator = AsyncPoseEstimator("unused.task")
    image = SimpleNamespace(width=640, height=480)
    result = SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])

    estimator._on_pose_result(result, image, 123)

    emitted = estimator.get_nowait()
    assert emitted is not None
    assert emitted.timestamp_ms == 123
    assert emitted.image_width == 640
    assert emitted.image_height == 480
    assert emitted.pose_detected is False
    assert emitted.landmarks == []
    assert emitted.world_landmarks == []
    assert estimator.get_latest() is emitted


def test_callback_preserves_pose_and_world_landmarks() -> None:
    estimator = AsyncPoseEstimator("unused.task")
    image = SimpleNamespace(width=320, height=240)
    result = SimpleNamespace(
        pose_landmarks=[[make_landmark(1.0), make_landmark(2.0)]],
        pose_world_landmarks=[[make_landmark(3.0)]],
    )

    estimator._on_pose_result(result, image, 456)

    emitted = estimator.get_nowait()
    assert emitted is not None
    assert emitted.pose_detected is True
    assert len(emitted.landmarks) == 2
    assert emitted.landmarks[0].x == 1.0
    assert emitted.landmarks[0].presence == 5.0
    assert len(emitted.world_landmarks) == 1
    assert emitted.world_landmarks[0].x == 3.0


def test_full_queue_drops_oldest_result() -> None:
    estimator = AsyncPoseEstimator("unused.task", result_queue_size=2)
    image = SimpleNamespace(width=1, height=1)
    no_pose = SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])

    estimator._on_pose_result(no_pose, image, 1)
    estimator._on_pose_result(no_pose, image, 2)
    estimator._on_pose_result(no_pose, image, 3)

    assert estimator.get_nowait().timestamp_ms == 2
    assert estimator.get_nowait().timestamp_ms == 3
    assert estimator.get_nowait() is None


def test_preview_draws_landmarks_and_connections() -> None:
    class FakeCv2:
        LINE_AA = 16

        def __init__(self) -> None:
            self.lines = []
            self.circles = []

        def line(self, *args, **kwargs) -> None:
            self.lines.append((args, kwargs))

        def circle(self, *args, **kwargs) -> None:
            self.circles.append((args, kwargs))

    estimator = AsyncPoseEstimator("unused.task")
    estimator._cv2 = FakeCv2()
    estimator._pose_connections = ((0, 1),)
    frame = SimpleNamespace(shape=(100, 200, 3))
    landmarks = [
        PoseLandmark(0.25, 0.5, 0.0, 1.0, 1.0),
        PoseLandmark(0.75, 0.25, 0.0, 1.0, 1.0),
    ]
    result = PoseResult(1, 1.0, 200, 100, landmarks, [], True)

    estimator._draw_pose_overlay(frame, result)

    assert estimator._cv2.lines[0][0][1:3] == ((50, 50), (150, 25))
    assert len(estimator._cv2.circles) == 2


def test_preview_draws_mirrored_landmarks_without_mutating_result() -> None:
    class FakeCv2:
        LINE_AA = 16

        def __init__(self) -> None:
            self.lines = []
            self.circles = []

        def line(self, *args, **kwargs) -> None:
            self.lines.append((args, kwargs))

        def circle(self, *args, **kwargs) -> None:
            self.circles.append((args, kwargs))

    estimator = AsyncPoseEstimator("unused.task")
    estimator._cv2 = FakeCv2()
    estimator._pose_connections = ((0, 1),)
    frame = SimpleNamespace(shape=(100, 200, 3))
    landmarks = [
        PoseLandmark(0.25, 0.5, 0.0, 1.0, 1.0),
        PoseLandmark(0.75, 0.25, 0.0, 1.0, 1.0),
    ]
    result = PoseResult(1, 1.0, 200, 100, landmarks, [], True)

    estimator._draw_pose_overlay(frame, result, mirror_x=True)

    assert estimator._cv2.lines[0][0][1:3] == ((150, 50), (50, 25))
    assert result.landmarks[0].x == 0.25


def test_preview_frame_mirrors_image_when_enabled() -> None:
    class FakeCv2:
        def __init__(self) -> None:
            self.flipped = False

        def flip(self, frame, axis):
            self.flipped = True
            assert axis == 1
            return "mirrored"

    estimator = AsyncPoseEstimator("unused.task", mirror_frame=True)
    estimator._cv2 = FakeCv2()

    assert estimator._preview_frame("raw") == "mirrored"
    assert estimator._cv2.flipped is True


def test_preview_frame_copies_image_when_not_mirrored() -> None:
    class FakeFrame:
        def __init__(self) -> None:
            self.copied = False

        def copy(self):
            self.copied = True
            return "copy"

    estimator = AsyncPoseEstimator("unused.task", mirror_frame=False)
    estimator._cv2 = object()
    frame = FakeFrame()

    assert estimator._preview_frame(frame) == "copy"
    assert frame.copied is True


@pytest.mark.parametrize(
    ("argument", "value"),
    [("target_fps", 0), ("result_queue_size", 0)],
)
def test_positive_configuration_is_required(argument, value) -> None:
    with pytest.raises(ValueError):
        AsyncPoseEstimator("unused.task", **{argument: value})
