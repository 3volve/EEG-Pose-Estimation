from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PredictedPoseLatentFrame:
    packet_id: int
    target_time_s: float
    predicted_latent: NDArray[np.float32]
    uncertainty: float | None = None

