from pathlib import Path


ROOT_DIR = Path(__file__).parent
BASE_OUTPUT_DIR = ROOT_DIR / "output"


# EEG packet stream.
EEG_SOURCE_CHANNEL_INDICES: tuple[int, ...] = (1, 2, 3, 4)
EEG_CHANNELS: int = len(EEG_SOURCE_CHANNEL_INDICES)
EEG_SAMPLE_RATE: int = 250
EEG_PACKET_SIZE: int = 200
EEG_PACKET_STRIDE: int = 50
EEG_ACQUISITION_SAMPLES: int = 5
EEG_ACQUISITION_TIMEOUT_S: float = 0.02
EEG_BANDSTOP_HZ: tuple[float, float] = (55.0, 65.0)
EEG_BANDSTOP_ORDER: int = 4
EEG_PREPROCESSING_VERSION: str = "lsl-columns-1-4-bandstop-55-65-v1"


# EEG encoding defaults.
EEG_MODELS_ROOT: Path = ROOT_DIR / "eeg_encoding" / "models"
EEG_ENCODING_MODEL: str = str(EEG_MODELS_ROOT / "eeg_base_corrected_v1.pt")
EEG_PROFILES_ROOT: Path = ROOT_DIR / "profiles" / EEG_PREPROCESSING_VERSION


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
POSE_LIVE_MEDIAN_WINDOW: int = 3
POSE_LIVE_MEAN_WINDOW: int = 2


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
EEG_CONTEXT_PACKET_COUNT: int = 4
EEG_STANDARDIZE_POSE_LATENTS: bool = True
EEG_VALIDATE_BY_RUN: bool = True
EEG_DATA_GLOB: str = str(ROOT_DIR / "eeg_encoding" / "data" / "*.npz")
EEG_WAVELET: str = "db4"
EEG_WAVELET_LEVEL: int = 4
EEG_WAVELET_MODE: str = "periodization"
EEG_WAVELET_STANDARDIZE_INPUT: bool = True
EEG_USE_BAND_ADAPTER: bool = True
EEG_DECODED_POSITION_LOSS_WEIGHT: float = 0.65
EEG_DECODED_VELOCITY_LOSS_WEIGHT: float = 1.5
EEG_STILLNESS_LOSS_WEIGHT: float = 0.4
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


# EEG personalization and calibration defaults.
EEG_ADAPTATION_MODE_ADAPTER_ONLY: str = "adapter_only"
EEG_ADAPTATION_MODE_ADAPTER_HEAD: str = "adapter_head"
EEG_ADAPTATION_MODE_SESSION: str = "session"
EEG_ADAPTATION_MODE_SESSION_DEEP: str = "session_deep"
EEG_ADAPTATION_MODE_PROFILE_HEAD: str = "profile_head"
EEG_ADAPTATION_MODE_PROFILE_CORE: str = "profile_core"
EEG_ADAPTATION_MODE_PROFILE_ENCODER: str = "profile_encoder"
EEG_ADAPTATION_MODE_PROFILE_FULL: str = "profile_full"
EEG_PROFILE_BUILD_INNER_EPOCHS: int = 8
EEG_PROFILE_BUILD_QUERY_EPOCHS: int = 1
EEG_PROFILE_BUILD_QUERY_FRACTION: float = 0.25
EEG_PROFILE_BUILD_REGRESSION_TOLERANCE: float = 0.02
EEG_PROFILE_BUILD_MOVEMENT_TOLERANCE: float = 0.05
EEG_CALIBRATION_MIN_DURATION_S: float = 30.0
EEG_CALIBRATION_MAX_SHORT_DURATION_S: float = 180.0
EEG_CALIBRATION_MIN_TRUSTED_SAMPLES: int = 50
EEG_CALIBRATION_MIN_POSE_CONFIDENCE: float = 0.7
EEG_CALIBRATION_MIN_INTERPOLATION_CONFIDENCE: float = 0.5
EEG_CALIBRATION_MAX_POSE_RECONSTRUCTION_ERROR: float = 0.12
EEG_CALIBRATION_MAX_DECODED_POSE_ERROR: float = 0.25
EEG_CALIBRATION_MAX_STATIONARY_FALSE_POSITIVE_SCORE: float = 0.2
EEG_CALIBRATION_MIN_MOVEMENT_RESPONSE_SCORE: float = 0.5
EEG_CALIBRATION_READY_SCORE: float = 0.9
EEG_CALIBRATION_PLATEAU_PATIENCE: int = 5
EEG_ONLINE_CALIBRATION_BATCH_SIZE: int = 16
EEG_ONLINE_CALIBRATION_MIN_BATCH_SIZE: int = 8
EEG_ONLINE_CALIBRATION_UPDATE_EVERY: int = 4
EEG_ONLINE_CALIBRATION_STEPS_PER_UPDATE: int = 2
EEG_ONLINE_CALIBRATION_MAX_SAMPLES: int = 512
EEG_PROFILE_CREATION_DURATION_S: float = 600.0


# Runtime defaults.
DEFAULT_DEVICE: str = "cpu"
LIVE_PREDICT_DURATION_S: float = 0.0
COLLECT_DURATION_S: float = 60.0
