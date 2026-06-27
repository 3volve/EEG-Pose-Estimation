"""Asynchronous webcam pose estimation with MediaPipe Tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread, current_thread
import time
from types import ModuleType
from typing import Any, Callable, Iterator
import warnings

from config import POSE_CAMERA_INDEX, POSE_RESULT_QUEUE_SIZE, POSE_TARGET_FPS


@dataclass(frozen=True, slots=True)
class PoseLandmark:
    x: float | None
    y: float | None
    z: float | None
    visibility: float | None
    presence: float | None


@dataclass(frozen=True, slots=True)
class PoseResult:
    timestamp_ms: int
    received_time_s: float
    image_width: int
    image_height: int
    landmarks: list[PoseLandmark]
    world_landmarks: list[PoseLandmark]
    pose_detected: bool


class AsyncPoseEstimator:
    """Capture webcam frames and publish fresh pose results asynchronously."""

    _PREVIEW_WINDOW = "Async Pose Estimator"

    def __init__(
        self,
        model_path: str,
        camera_index: int = POSE_CAMERA_INDEX,
        target_fps: float = POSE_TARGET_FPS,
        result_queue_size: int = POSE_RESULT_QUEUE_SIZE,
        mirror_frame: bool = False,
        draw_preview: bool = False,
        draw_builtin_pose_overlay: bool = True,
        preview_renderer: Callable[[Any, ModuleType, PoseResult | None, bool], None] | None = None,
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")
        if result_queue_size <= 0:
            raise ValueError("result_queue_size must be greater than zero")

        self.model_path = Path(model_path)
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.mirror_frame = mirror_frame
        self.draw_preview = draw_preview
        self.draw_builtin_pose_overlay = draw_builtin_pose_overlay
        self.preview_renderer = preview_renderer

        self._results: Queue[PoseResult] = Queue(maxsize=result_queue_size)
        self._latest: PoseResult | None = None
        self._latest_lock = Lock()
        self._state_lock = Lock()
        self._stop_event = Event()
        self._capture_thread: Thread | None = None
        self._camera: Any = None
        self._landmarker: Any = None
        self._cv2: ModuleType | None = None
        self._mp: ModuleType | None = None
        self._pose_connections: tuple[tuple[int, int], ...] = ()
        self._preview_created = False
        self._error: RuntimeError | None = None

    @property
    def is_running(self) -> bool:
        thread = self._capture_thread
        return thread is not None and thread.is_alive()

    @property
    def error(self) -> RuntimeError | None:
        """Return a capture-thread failure, if one occurred."""
        return self._error

    def start(self) -> None:
        with self._state_lock:
            if self.is_running:
                return
            if self._capture_thread is not None:
                self._release_resources()
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"MediaPipe pose model not found: {self.model_path}"
                )

            cv2, mp = self._import_runtime()
            camera = cv2.VideoCapture(self.camera_index)
            if not camera.isOpened():
                camera.release()
                raise RuntimeError(
                    f"Could not open camera at index {self.camera_index}"
                )

            try:
                options = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=str(self.model_path)
                    ),
                    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                    num_poses=1,
                    output_segmentation_masks=False,
                    result_callback=self._on_pose_result,
                )
                landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
                    options
                )
            except Exception as exc:
                camera.release()
                raise RuntimeError(
                    f"Could not initialize MediaPipe Pose Landmarker: {exc}"
                ) from exc

            self._clear_results()
            self._latest = None
            self._error = None
            self._stop_event.clear()
            self._cv2 = cv2
            self._mp = mp
            self._pose_connections = tuple(
                (connection.start, connection.end)
                for connection in (
                    mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
                )
            )
            self._camera = camera
            self._landmarker = landmarker
            self._capture_thread = Thread(
                target=self._capture_loop,
                name="pose-capture",
                daemon=True,
            )
            self._capture_thread.start()

    def stop(self) -> None:
        with self._state_lock:
            self._stop_event.set()
            thread = self._capture_thread
            if thread is not None and thread is not current_thread():
                thread.join()
            self._release_resources()

    def _release_resources(self) -> None:
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._preview_created and self._cv2 is not None:
            self._cv2.destroyWindow(self._PREVIEW_WINDOW)
            self._preview_created = False
        self._capture_thread = None

    def get_latest(self) -> PoseResult | None:
        with self._latest_lock:
            return self._latest

    def get_nowait(self) -> PoseResult | None:
        try:
            return self._results.get_nowait()
        except Empty:
            return None

    def results(self) -> Iterator[PoseResult]:
        """Yield queued results until capture has stopped and the queue is empty."""
        while self.is_running or not self._results.empty():
            try:
                yield self._results.get(timeout=0.1)
            except Empty:
                continue

    def __enter__(self) -> AsyncPoseEstimator:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    @staticmethod
    def _import_runtime() -> tuple[ModuleType, ModuleType]:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe and OpenCV are required. Install them in the active "
                "environment with: "
                "python -m pip install mediapipe opencv-contrib-python"
            ) from exc
        return cv2, mp

    def _capture_loop(self) -> None:
        assert self._camera is not None
        assert self._landmarker is not None
        assert self._cv2 is not None
        assert self._mp is not None

        frame_period_s = 1.0 / self.target_fps
        next_capture_time = time.monotonic()
        last_timestamp_ms = -1

        while not self._stop_event.is_set():
            wait_s = next_capture_time - time.monotonic()
            if wait_s > 0 and self._stop_event.wait(wait_s):
                break

            ok, frame = self._camera.read()
            if not ok:
                self._error = RuntimeError(
                    f"Frame capture failed for camera index {self.camera_index}"
                )
                warnings.warn(str(self._error), RuntimeWarning, stacklevel=2)
                self._stop_event.set()
                break

            rgb_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            timestamp_ms = max(
                time.monotonic_ns() // 1_000_000,
                last_timestamp_ms + 1,
            )
            last_timestamp_ms = timestamp_ms
            self._landmarker.detect_async(mp_image, timestamp_ms)

            if self.draw_preview:
                preview_frame = self._preview_frame(frame)
                result = self.get_latest()
                if self.draw_builtin_pose_overlay:
                    self._draw_pose_overlay(
                        preview_frame,
                        result,
                        mirror_x=self.mirror_frame,
                    )
                if self.preview_renderer is not None:
                    self.preview_renderer(
                        preview_frame,
                        self._cv2,
                        result,
                        self.mirror_frame,
                    )
                self._cv2.imshow(self._PREVIEW_WINDOW, preview_frame)
                self._preview_created = True
                if self._cv2.waitKey(1) & 0xFF == ord("q"):
                    self._stop_event.set()

            next_capture_time = max(
                next_capture_time + frame_period_s,
                time.monotonic(),
            )

    def _preview_frame(self, frame: Any) -> Any:
        assert self._cv2 is not None
        if self.mirror_frame:
            return self._cv2.flip(frame, 1)
        return frame.copy()

    def _draw_pose_overlay(
        self,
        frame: Any,
        result: PoseResult | None,
        *,
        mirror_x: bool = False,
    ) -> None:
        if result is None or not result.pose_detected:
            return

        assert self._cv2 is not None
        height, width = frame.shape[:2]
        points = [
            (
                round((1.0 - landmark.x if mirror_x else landmark.x) * width),
                round(landmark.y * height),
            )
            for landmark in result.landmarks
        ]

        for start, end in self._pose_connections:
            self._cv2.line(
                frame,
                points[start],
                points[end],
                color=(0, 220, 0),
                thickness=2,
                lineType=self._cv2.LINE_AA,
            )

        for point in points:
            self._cv2.circle(
                frame,
                point,
                radius=3,
                color=(0, 80, 255),
                thickness=-1,
                lineType=self._cv2.LINE_AA,
            )

    def _on_pose_result(
        self,
        result: Any,
        output_image: Any,
        timestamp_ms: int,
    ) -> None:
        pose_detected = bool(result.pose_landmarks)
        landmarks = (
            self._convert_landmarks(result.pose_landmarks[0])
            if pose_detected
            else []
        )
        world_landmarks = (
            self._convert_landmarks(result.pose_world_landmarks[0])
            if result.pose_world_landmarks
            else []
        )
        pose_result = PoseResult(
            timestamp_ms=timestamp_ms,
            received_time_s=time.monotonic(),
            image_width=output_image.width,
            image_height=output_image.height,
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            pose_detected=pose_detected,
        )

        with self._latest_lock:
            self._latest = pose_result

        while True:
            try:
                self._results.put_nowait(pose_result)
                break
            except Full:
                try:
                    self._results.get_nowait()
                except Empty:
                    pass

    @staticmethod
    def _convert_landmarks(landmarks: list[Any]) -> list[PoseLandmark]:
        return [
            PoseLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=landmark.visibility,
                presence=landmark.presence,
            )
            for landmark in landmarks
        ]

    def _clear_results(self) -> None:
        while True:
            try:
                self._results.get_nowait()
            except Empty:
                return
