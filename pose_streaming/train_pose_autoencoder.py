"""Offline training for the pose autoencoder."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from pose_autoencoder import PoseAutoencoder, save_checkpoint


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _validation_split(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return parsed


def load_feature_matrix(path: str | Path) -> NDArray[np.float32]:
    """Load and validate an external pose feature archive."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Pose feature dataset not found: {data_path}")
    if data_path.suffix.lower() != ".npz":
        raise ValueError(f"Pose feature dataset must be a .npz file: {data_path}")

    try:
        with np.load(data_path) as dataset:
            if "features" not in dataset:
                raise ValueError("archive is missing required 'features' array")
            features = np.asarray(dataset["features"], dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not load pose feature dataset {data_path}: {exc}"
        ) from exc

    if features.ndim != 2:
        raise ValueError(
            f"Pose feature dataset must be 2D; got shape {features.shape}"
        )
    if len(features) == 0:
        raise ValueError("Pose feature dataset is empty")
    if not np.isfinite(features).all():
        raise ValueError("Pose feature dataset contains non-finite values")
    return features


def train_autoencoder(
    features: NDArray[np.float32],
    *,
    latent_dim: int = 8,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_split: float = 0.1,
    device: str | torch.device = "cpu",
    seed: int = 0,
    verbose: bool = False,
) -> tuple[PoseAutoencoder, dict[str, list[float]]]:
    torch.manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(features))
    validation_size = (
        min(max(1, round(len(dataset) * val_split)), len(dataset) - 1)
        if val_split > 0 and len(dataset) > 1
        else 0
    )
    training_size = len(dataset) - validation_size
    training_data, validation_data = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    training_loader = DataLoader(
        training_data,
        batch_size=min(batch_size, training_size),
        shuffle=True,
    )
    validation_loader = (
        DataLoader(
            validation_data,
            batch_size=min(batch_size, validation_size),
        )
        if validation_size
        else training_loader
    )

    model = PoseAutoencoder(
        input_dim=features.shape[1],
        latent_dim=latent_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    history = {"train_loss": [], "val_loss": []}
    best_validation_loss = float("inf")
    best_state = deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = _run_epoch(
            model,
            training_loader,
            loss_fn,
            device,
            optimizer,
        )
        model.eval()
        with torch.no_grad():
            validation_loss = _run_epoch(
                model,
                validation_loader,
                loss_fn,
                device,
            )

        history["train_loss"].append(training_loss)
        history["val_loss"].append(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = deepcopy(model.state_dict())
        if verbose:
            print(
                f"epoch {epoch:03d}/{epochs}: "
                f"train={training_loss:.6f} val={validation_loss:.6f}"
            )

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def _run_epoch(
    model: PoseAutoencoder,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str | torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    total_loss = 0.0
    sample_count = 0
    for (batch,) in loader:
        batch = batch.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        reconstruction, _ = model(batch)
        loss = loss_fn(reconstruction, batch)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(batch)
        sample_count += len(batch)
    return total_loss / sample_count


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an autoencoder on saved pose feature vectors."
    )
    parser.add_argument("--data", required=True, help="Input .npz feature archive")
    parser.add_argument("--out", required=True, help="Output checkpoint path")
    parser.add_argument("--latent-dim", type=_positive_int, default=8)
    parser.add_argument("--epochs", type=_positive_int, default=100)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--lr", type=_positive_float, default=1e-3)
    parser.add_argument("--val-split", type=_validation_split, default=0.1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    features = load_feature_matrix(args.data)
    model, history = train_autoencoder(
        features,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device=device,
        verbose=True,
    )
    config = {
        "data": str(Path(args.data)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_split": args.val_split,
        "device": device,
        "best_validation_loss": min(history["val_loss"]),
    }
    save_checkpoint(args.out, model, config)
    print(
        f"Saved best checkpoint to {args.out}; "
        f"validation SmoothL1={config['best_validation_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
