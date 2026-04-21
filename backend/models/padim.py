import torch
import numpy as np
from typing import Optional, Tuple


class PaDiM:
    """
    PaDiM anomaly detector.

    Fits a multivariate Gaussian per spatial patch position on normal training features.
    At inference time computes the Mahalanobis distance, which is the anomaly score map.

    Usage:
        padim = PaDiM()
        padim.fit(train_features)   # [N, C, H, W]
        padim.save("model.pt")

        padim = PaDiM.load("model.pt")
        score_map = padim.predict(test_features)  # [B, H, W]
    """

    def __init__(self, epsilon: float = 0.01):
        self.epsilon = epsilon
        self.mean: Optional[torch.Tensor] = None
        self.covariance: Optional[torch.Tensor] = None
        self.spatial_shape: Optional[Tuple[int, int]] = None
        self._is_fitted = False

    def fit(self, features: torch.Tensor) -> "PaDiM":
        N, C, H, W = features.shape
        self.spatial_shape = (H, W)
        num_patches = H * W

        print(f"[PaDiM] Fitting on {N} images, {C} channels, {H}x{W} patches...")

        features_flat = features.reshape(N, C, num_patches).permute(2, 0, 1)  # [H*W, N, C]
        self.mean = features_flat.mean(dim=1)  # [H*W, C]

        centered = features_flat - self.mean.unsqueeze(1)  # [H*W, N, C]
        self.covariance = torch.bmm(centered.transpose(1, 2), centered) / N  # [H*W, C, C]

        identity = torch.eye(C, device=features.device).unsqueeze(0)
        self.covariance += self.epsilon * identity

        self._is_fitted = True
        print(f"[PaDiM] Done. mean={self.mean.shape}, cov={self.covariance.shape}")
        return self

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if not self._is_fitted:
            raise RuntimeError("PaDiM not fitted. Call .fit() first.")

        B, C, H, W = features.shape
        num_patches = H * W

        feat_flat = features.reshape(B, C, num_patches).permute(2, 0, 1)  # [H*W, B, C]
        diff = feat_flat - self.mean.unsqueeze(1).to(features.device)  # [H*W, B, C]

        cov_inv = torch.linalg.inv(self.covariance.to(features.device))  # [H*W, C, C]
        left = torch.bmm(diff, cov_inv)  # [H*W, B, C]
        mahal_sq = (left * diff).sum(dim=2)  # [H*W, B]
        mahal = torch.sqrt(torch.clamp(mahal_sq, min=0))
        return mahal.permute(1, 0).reshape(B, H, W)  # [B, H, W]

    def save(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError("Nothing to save — model not fitted.")
        torch.save({
            "mean": self.mean.cpu(),
            "covariance": self.covariance.cpu(),
            "spatial_shape": self.spatial_shape,
            "epsilon": self.epsilon,
        }, path)
        print(f"[PaDiM] Saved: {path}")

    @classmethod
    def load(cls, path: str) -> "PaDiM":
        state = torch.load(path, weights_only=False)
        padim = cls(epsilon=state["epsilon"])
        padim.mean = state["mean"]
        padim.covariance = state["covariance"]
        padim.spatial_shape = state["spatial_shape"]
        padim._is_fitted = True
        return padim
