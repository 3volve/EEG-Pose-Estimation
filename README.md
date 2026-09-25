# EEG-Pose-Estimation

I'm exploring whether EEG signals can be used to estimate upper-body pose andmovement. I use webcam-based pose tracking to provide training targets, then train models to predict those representations from EEG.

This is an active exploratory research prototype, bringing together signal processing, machine learning, real-time data collection, and experimental evaluation.

## What I've built

- **Synchronized collection:** EEG streaming through Lab Streaming Layer (LSL),
  MediaPipe pose tracking, filtering, and timestamp-based pairing.
- **Modeling:** a pose autoencoder and a wavelet-based EEG model that uses
  temporal context to predict pose representations.
- **Personalization:** guided calibration, participant profiles, and adaptation
  across recording sessions.
- **Evaluation infrastructure:** held-out data management, detailed recording
  diagnostics, and regression tests for timing, preprocessing, and data splits.

## Current focus

The main question is whether the models capture useful movement information
that holds up across recordings and days. Much of my current work is separating
that information from timing errors, movement artifacts, noisy camera labels,
and repeated movement patterns.

The pipeline is implemented, but reliable EEG-based pose estimation has not yet
been established. Experiments and architecture are still evolving; this
repository is a snapshot of that ongoing work.

## A quick look at the code

[`streaming/`](streaming/) contains acquisition and alignment,
[`pose_encoding/`](pose_encoding/) contains the pose representation, and
[`eeg_encoding/`](eeg_encoding/) contains EEG modeling and personalization.
[`tests/`](tests/) covers the main pipeline contracts.

**Built with:** Python, PyTorch, NumPy/SciPy, PyWavelets, MediaPipe, OpenCV, and LSL.
