"""
Helper functions for saving and loading PaDiM checkpoints,
and for random channel subsampling (replaces PCA, no sklearn required).

Adapted from SterilVision's anomaly_detection/pipeline.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from backend.models.padim import PaDiM


@dataclass
class ScoreStats:
    """Training-set score statistics used for cross-image normalization."""
    mean: float
    std: float
    p95: float   # 95th percentile max-score on training images


@dataclass
class PaDiMFitResult:
    padim: PaDiM
    channel_indices: torch.Tensor
    num_train_images: int
    feature_shape: tuple
    score_stats: ScoreStats | None = None


def choose_channel_indices(
    num_channels: int,
    target: int | None,
    seed: int = 42,
) -> torch.Tensor:
    """Random channel subsampling — returns indices of selected channels."""
    if target is None or target <= 0 or target >= num_channels:
        return torch.arange(num_channels, dtype=torch.long)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    selected = torch.randperm(num_channels, generator=gen)[:target]
    return torch.sort(selected).values.to(dtype=torch.long)


def reduce_channels(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return features[:, indices, :, :]


def save_padim_model(
    path: str | Path,
    fit_result: PaDiMFitResult,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    score_stats_dict = None
    if fit_result.score_stats is not None:
        ss = fit_result.score_stats
        score_stats_dict = {"mean": ss.mean, "std": ss.std, "p95": ss.p95}

    torch.save({
        "padim": {
            "mean": fit_result.padim.mean.cpu(),
            "covariance": fit_result.padim.covariance.cpu(),
            "spatial_shape": fit_result.padim.spatial_shape,
            "epsilon": fit_result.padim.epsilon,
        },
        "channel_indices": fit_result.channel_indices.cpu(),
        "num_train_images": fit_result.num_train_images,
        "feature_shape": fit_result.feature_shape,
        "score_stats": score_stats_dict,
        "metadata": metadata or {},
    }, path)
    print(f"[Checkpoint] Saved: {path}")


def load_padim_model(path: str | Path) -> tuple[PaDiM, torch.Tensor, dict[str, Any]]:
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)

    ps = state["padim"]
    padim = PaDiM(epsilon=float(ps.get("epsilon", 0.01)))
    padim.mean = ps["mean"]
    padim.covariance = ps["covariance"]
    padim.spatial_shape = tuple(ps["spatial_shape"])
    padim._is_fitted = True

    indices = state.get("channel_indices")
    if indices is None:
        indices = torch.arange(padim.mean.shape[1], dtype=torch.long)
    else:
        indices = indices.to(dtype=torch.long, device="cpu")

    metadata = state.get("metadata", {})
    ss_dict = state.get("score_stats")
    score_stats = None
    if ss_dict:
        score_stats = ScoreStats(
            mean=float(ss_dict["mean"]),
            std=float(ss_dict["std"]),
            p95=float(ss_dict["p95"]),
        )
    return padim, indices, metadata, score_stats
