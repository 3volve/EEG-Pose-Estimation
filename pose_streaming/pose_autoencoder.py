"""Small autoencoder for normalized pose feature vectors."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import torch
from torch import Tensor, nn


class PoseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 48,
        latent_dim: int = 8,
        hidden_dims: tuple[int, int] = (64, 32),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        first_hidden, second_hidden = hidden_dims

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, input_dim),
        )

    def encode(self, features: Tensor) -> Tensor:
        return self.encoder(features)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(features)
        return self.decode(latent), latent


def save_checkpoint(
    path: str | Path,
    model: PoseAutoencoder,
    config: dict[str, Any],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
            "hidden_dims": model.hidden_dims,
            "config": dict(config),
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[PoseAutoencoder, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Pose autoencoder checkpoint not found: {checkpoint_path}"
        )

    try:
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        model = PoseAutoencoder(
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
            hidden_dims=tuple(checkpoint["hidden_dims"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        config = dict(checkpoint.get("config", {}))
    except (
        EOFError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"Invalid pose autoencoder checkpoint: {checkpoint_path}"
        ) from exc

    config.setdefault("input_dim", model.input_dim)
    config.setdefault("latent_dim", model.latent_dim)
    config.setdefault("hidden_dims", model.hidden_dims)
    model.to(map_location)
    model.eval()
    return model, config
