import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from backend.models.backbone import FeatureExtractor
from backend.models.base import AnomalyResult, InferenceOutput
from backend.models.padim import PaDiM
from backend.inference.heatmap import process_score_map
from backend.inference.overlay import build_heatmap_overlay
from backend.inference.pipeline_utils import ScoreStats, load_padim_model, reduce_channels

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class InferenceEngine:
    """
    Lazy-loads and caches one (FeatureExtractor, PaDiM, channel_indices) triple per category.
    Thread-safety is not guaranteed — intended for single-process FastAPI use.
    """

    def __init__(self, checkpoint_dir: str | Path, device: str = "auto"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._cache: dict[str, tuple[FeatureExtractor, PaDiM, torch.Tensor, dict, ScoreStats | None]] = {}

    def is_trained(self, category: str) -> bool:
        return (self.checkpoint_dir / f"{category}.pt").exists()

    def trained_categories(self) -> list[str]:
        return [p.stem for p in self.checkpoint_dir.glob("*.pt")]

    def _load_model(
        self, category: str
    ) -> tuple[FeatureExtractor, PaDiM, torch.Tensor, dict, ScoreStats | None]:
        if category in self._cache:
            return self._cache[category]

        ckpt_path = self.checkpoint_dir / f"{category}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint for '{category}'. Run training first: "
                f"python -m backend.training.train --categories {category}"
            )

        padim, indices, metadata, score_stats = load_padim_model(ckpt_path)
        backbone_cfg = metadata.get("backbone_config", {})
        extractor = FeatureExtractor(
            backbone=backbone_cfg.get("backbone", "efficientnet_b0"),
            layers=backbone_cfg.get("layers", [1, 2, 3]),
            device=self.device,
            pretrained=True,
        )
        self._cache[category] = (extractor, padim, indices, metadata, score_stats)
        print(f"[Engine] Loaded model for '{category}'")
        return self._cache[category]

    def predict(
        self,
        category: str,
        image_path: str,
        threshold: float = 0.5,
        overlay_alpha: float = 0.5,
    ) -> InferenceOutput:
        t0 = time.perf_counter()
        extractor, padim, indices, metadata, score_stats = self._load_model(category)

        image_size = metadata.get("backbone_config", {}).get("image_size", 224)
        transform = _build_transform(image_size)

        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]

        with torch.no_grad():
            features = extractor.extract(tensor)                  # [1, C, h, w]
            features = reduce_channels(features.cpu(), indices)   # [1, target_C, h, w]
            raw_scores = padim.predict(features)                  # [1, h, w]

        # Raw score (max Mahalanobis distance) normalized against training distribution.
        # Normalized score > 1.0 means the image is more anomalous than 95% of training images.
        raw_max = float(raw_scores.max().item())
        if score_stats is not None and score_stats.p95 > 0:
            # Scores > 1.0 mean more anomalous than 95% of normal reference images.
            normalized_score = float(np.clip(raw_max / score_stats.p95, 0.0, 5.0))
        else:
            normalized_score = float(raw_max)

        # Heatmap: per-image normalization for clear visualization of anomaly regions
        score_map = process_score_map(
            raw_scores,
            target_size=(image_size, image_size),
            sigma=4.0,
            normalize=True,
        )  # [1, H, W] in [0, 1]

        heatmap_np = score_map[0].cpu().numpy().astype(np.float32)

        heatmap_b64, overlay_b64 = build_heatmap_overlay(
            image_path, heatmap_np, image_size=image_size, alpha=overlay_alpha
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return InferenceOutput(
            category=category,
            inference_time_ms=round(elapsed_ms, 1),
            anomaly=AnomalyResult(
                score=round(normalized_score, 4),
                is_anomaly=normalized_score >= threshold,
                threshold=threshold,
            ),
            heatmap_b64=heatmap_b64,
            overlay_b64=overlay_b64,
        )

    def unload(self, category: str) -> None:
        if category in self._cache:
            extractor, _, _, _, _ = self._cache.pop(category)
            extractor.cleanup()
            print(f"[Engine] Unloaded '{category}'")
