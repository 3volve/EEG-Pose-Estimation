from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .records import PairedTrainingFrame


def save_paired_frames(
    path: str | Path,
    frames: list[PairedTrainingFrame],
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        packet_id=np.asarray([frame.packet_id for frame in frames], dtype=np.int64),
        eeg=np.stack([frame.eeg for frame in frames]).astype(np.float32, copy=False),
        target_time_s=np.asarray(
            [frame.target_time_s for frame in frames],
            dtype=np.float64,
        ),
        pose_latent=np.stack([frame.pose_latent for frame in frames]).astype(
            np.float32,
            copy=False,
        ),
        pose_confidence=np.asarray(
            [frame.pose_confidence for frame in frames],
            dtype=np.float32,
        ),
        pose_reconstruction_error=np.asarray(
            [frame.pose_reconstruction_error for frame in frames],
            dtype=np.float32,
        ),
        interpolation_confidence=np.asarray(
            [frame.interpolation_confidence for frame in frames],
            dtype=np.float32,
        ),
        **{
            key: np.asarray(value)
            for key, value in (metadata or {}).items()
        },
    )


def load_paired_arrays(path: str | Path) -> dict[str, NDArray]:
    with np.load(path) as dataset:
        return {key: dataset[key] for key in dataset.files}

