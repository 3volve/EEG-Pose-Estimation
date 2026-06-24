from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class EegPacket:
    packet_id: int
    samples: NDArray[np.float32]
    start_time_s: float
    end_time_s: float
    sample_rate: int

