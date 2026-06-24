# Pose Encoding

This package trains and runs the pose autoencoder that turns normalized pose
features into latent training targets. Webcam capture and MediaPipe landmark
streaming live in `streaming.pose`.

The configured `synaptech-arm` conda environment uses Python 3.12. Required
packages are listed in the local `requirements.txt`.

## Model

Download the MediaPipe Pose Landmarker Lite model and keep it as a local file
named `pose_landmarker_lite.task`. Pass its path explicitly when starting the
demo.

Official model and documentation:
https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker

## Run

```powershell
conda activate synaptech-arm
python -m pose_encoding.collect_pose_features --model C:\path\to\pose_landmarker_lite.task --mirror
```

The preview draws the latest predicted pose as green connections with red
landmark points. Because inference is asynchronous, the overlay can trail the
displayed frame slightly. Press `q` in the preview window or `Ctrl+C` in the
terminal to stop.

Use the module directly:

```python
from streaming.pose import AsyncPoseEstimator

estimator = AsyncPoseEstimator("pose_landmarker_lite.task")
estimator.start()

result = estimator.get_latest()

estimator.stop()
```

`get_latest()` polls the newest result without consuming it. `get_nowait()`
consumes one queued result, and `results()` yields queued results until capture
stops. When the queue is full, the oldest result is discarded.
