from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PairedTrainingFrame:
    packet_id: int
    eeg: NDArray[np.float32]
    target_time_s: float
    pose_latent: NDArray[np.float32]
    pose_confidence: float
    pose_reconstruction_error: float
    interpolation_confidence: float

