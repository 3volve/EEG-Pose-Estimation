from pathlib import Path


ROOT_DIR = Path(__file__).parent
BASE_OUTPUT_DIR = ROOT_DIR / "output"


# EEG packet stream.
EEG_CHANNELS: int = 4
EEG_SAMPLE_RATE: int = 250
EEG_PACKET_SIZE: int = 200
EEG_PACKET_STRIDE: int = 50
EEG_STREAM_TIMEOUT_MARGIN_S: float = 0.01


# EEG encoding defaults.
EEG_MODELS_ROOT: Path = ROOT_DIR / "eeg_encoding" / "models"
EEG_ENCODING_MODEL: str = str(EEG_MODELS_ROOT / "<placeholder>.pt")


# Pose stream and pose encoding defaults.
POSE_CAMERA_INDEX: int = 0
POSE_TARGET_FPS: float = 30.0
POSE_RESULT_QUEUE_SIZE: int = 128
POSE_CAPTURE_MIN_CONFIDENCE: float = 0.5
POSE_INCLUDE_VELOCITY: bool = True
POSE_USE_WORLD_LANDMARKS: bool = True
POSE_MODELS_ROOT: Path = ROOT_DIR / "pose_encoding" / "models"
POSE_MODEL: str = str(POSE_MODELS_ROOT / "pose_landmarker_full.task")
POSE_ENCODING_MODEL: str = str(POSE_MODELS_ROOT / "pose_autoencoder_velocity_l24.pt")


# EEG/pose pairing.
PAIRING_MAX_POSE_GAP_S: float = 0.12
PAIRING_POLL_DELAY_S: float = 0.002


# Pose feature preparation and diagnostics.
POSE_PREP_DROP_START_SECONDS: float = 3.0
POSE_PREP_MIN_CONFIDENCE: float = 0.7
POSE_PREP_GAP_MS: float = 75.0
POSE_PREP_MEDIAN_WINDOW: int = 5
POSE_PREP_MEAN_WINDOW: int = 5
POSE_PREP_MIN_SEGMENT_SAMPLES: int = 5
POSE_EVAL_LOW_CONFIDENCE: float = 0.7
POSE_EVAL_OUTLIER_PERCENTILE: float = 99.5
POSE_DATA_GLOB: str = str(ROOT_DIR / "pose_encoding" / "data" / "*.npz")


# Live pose labels use causal smoothing before pose-latent encoding.
POSE_LIVE_SMOOTHING: bool = True
POSE_LIVE_MEDIAN_WINDOW: int = POSE_PREP_MEDIAN_WINDOW
POSE_LIVE_MEAN_WINDOW: int = POSE_PREP_MEAN_WINDOW


# Pose autoencoder training.
POSE_AUTOENCODER_LATENT_DIM: int = 24
POSE_AUTOENCODER_HIDDEN_DIMS: tuple[int, int] = (64, 32)
POSE_AUTOENCODER_EPOCHS: int = 250
POSE_AUTOENCODER_BATCH_SIZE: int = 128
POSE_AUTOENCODER_LR: float = 1e-4
POSE_AUTOENCODER_VAL_SPLIT: float = 0.1


# EEG-to-pose-latent model training.
EEG_MODEL_EPOCHS: int = 200
EEG_MODEL_BATCH_SIZE: int = 64
EEG_MODEL_LR: float = 1e-3
EEG_MODEL_LATENT_DIM: int = 24
EEG_MODEL_HIDDEN_DIM: int = 128
EEG_MODEL_BETA: float = 2e-3
EEG_MODEL_RECONSTRUCTION_WEIGHT: float = 1e-1
EEG_MODEL_SEED: int = 0
EEG_MODEL_VAL_SPLIT: float = 0.2
EEG_CONTEXT_PACKET_COUNT: int = 2
EEG_STANDARDIZE_POSE_LATENTS: bool = True
EEG_VALIDATE_BY_RUN: bool = True
EEG_DATA_GLOB: str = str(ROOT_DIR / "eeg_encoding" / "data" / "paired_run_??.npz")
EEG_WAVELET: str = "db4"
EEG_WAVELET_LEVEL: int = 4
EEG_WAVELET_MODE: str = "periodization"
EEG_WAVELET_STANDARDIZE_INPUT: bool = True
EEG_DECODED_POSITION_LOSS_WEIGHT: float = 0.5
EEG_DECODED_VELOCITY_LOSS_WEIGHT: float = 1.5
EEG_STILLNESS_LOSS_WEIGHT: float = 0.2
EEG_STILLNESS_TARGET_VELOCITY_THRESHOLD: float = 0.08
EEG_STILLNESS_ALLOWED_PREDICTED_VELOCITY: float = 0.12
EEG_POSITION_LANDMARK_WEIGHTS: tuple[float, ...] = (
    1.5,  # left shoulder
    1.5,  # right shoulder
    1.0,  # left elbow
    1.0,  # right elbow
    0.75,  # left wrist
    0.75,  # right wrist
    1.5,  # left hip
    1.5,  # right hip
)
EEG_VELOCITY_LANDMARK_WEIGHTS: tuple[float, ...] = (
    0.5,  # left shoulder
    0.5,  # right shoulder
    0.9,  # left elbow
    0.9,  # right elbow
    1.0,  # left wrist
    1.0,  # right wrist
    0.25, # left hip
    0.25, # right hip
)


# Runtime defaults.
DEFAULT_DEVICE: str = "cpu"
LIVE_PREDICT_DURATION_S: float = 0.0
COLLECT_DURATION_S: float = 60.0
