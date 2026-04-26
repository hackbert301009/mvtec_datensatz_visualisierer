"""
Training script for PaDiM anomaly detection models.

Usage:
    python -m backend.training.train --categories bottle capsule
    python -m backend.training.train --dataset mvtec_ad2 --categories can fabric
    python -m backend.training.train --all
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.data.discovery import BaseDiscovery, create_discovery
from backend.models.backbone import FeatureExtractor
from backend.models.padim import PaDiM
from backend.inference.pipeline_utils import (
    PaDiMFitResult,
    ScoreStats,
    choose_channel_indices,
    reduce_channels,
    save_padim_model,
)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _make_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _extract_features_batched(
    image_paths: list[str],
    extractor: FeatureExtractor,
    transform: transforms.Compose,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    all_features = []
    total = len(image_paths)

    for start in range(0, total, batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(tensors)
        feats = extractor.extract(batch).detach().cpu()
        all_features.append(feats)
        print(f"  [{min(start + batch_size, total)}/{total}] features extracted", end="\r")

    print()
    return torch.cat(all_features, dim=0)


def train_category_with_discovery(
    category: str,
    discovery: BaseDiscovery,
    checkpoint_dir: str | Path,
    backbone: str = "efficientnet_b0",
    layers: list[int] | None = None,
    image_size: int = 224,
    target_channels: int = 100,
    batch_size: int = 32,
    epsilon: float = 0.01,
    device: str = "auto",
) -> Path:
    if layers is None:
        layers = [1, 2, 3]

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out_path = checkpoint_dir / f"{category}.pt"

    print(f"\n{'='*60}")
    print(f"[train] Category: {category}")
    print(f"{'='*60}")

    train_paths = discovery.get_train_paths(category)
    print(f"[train] {len(train_paths)} training images found")

    if device == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    extractor = FeatureExtractor(backbone=backbone, layers=layers, device=str(resolved_device))
    transform = _make_transform(image_size)

    t0 = time.perf_counter()
    print(f"[train] Extracting features (batch_size={batch_size})...")
    all_features = _extract_features_batched(train_paths, extractor, transform, batch_size, resolved_device)

    N, C, H, W = all_features.shape
    print(f"[train] Features: {N} images, {C} channels, {H}x{W} spatial")

    indices = choose_channel_indices(C, target_channels, seed=42)
    reduced = reduce_channels(all_features, indices)
    print(f"[train] Using {len(indices)}/{C} channels")

    padim = PaDiM(epsilon=epsilon)
    padim.fit(reduced)

    import numpy as np
    calib_paths = discovery.get_good_test_images(category)

    if calib_paths:
        print(f"[train] Calibrating on {len(calib_paths)} test/good images...")
        calib_features = _extract_features_batched(calib_paths, extractor, transform, batch_size, resolved_device)
        calib_features = reduce_channels(calib_features, indices)
        with torch.no_grad():
            calib_scores = padim.predict(calib_features)
        calib_max = calib_scores.flatten(1).max(dim=1).values.cpu().numpy()
        score_stats = ScoreStats(
            mean=float(np.mean(calib_max)),
            std=float(np.std(calib_max)),
            p95=float(np.percentile(calib_max, 95)),
        )
    else:
        print("[train] No test/good images found, calibrating on training scores")
        with torch.no_grad():
            raw_train_scores = padim.predict(reduced)
            max_scores = raw_train_scores.flatten(1).max(dim=1).values.cpu().numpy()
        score_stats = ScoreStats(
            mean=float(np.mean(max_scores)),
            std=float(np.std(max_scores)),
            p95=float(np.percentile(max_scores, 95)),
        )

    print(f"[train] Score stats — mean: {score_stats.mean:.1f}, p95: {score_stats.p95:.1f}")

    fit_result = PaDiMFitResult(
        padim=padim,
        channel_indices=indices,
        num_train_images=N,
        feature_shape=tuple(reduced.shape),
        score_stats=score_stats,
    )
    save_padim_model(
        out_path,
        fit_result,
        metadata={
            "category": category,
            "trained_at": datetime.now().isoformat(),
            "backbone_config": {
                "backbone": backbone,
                "layers": layers,
                "image_size": image_size,
            },
        },
    )

    elapsed = time.perf_counter() - t0
    print(f"[train] Done in {elapsed:.1f}s → {out_path}")
    extractor.cleanup()
    return out_path


def train_category(
    category: str,
    dataset_root: str | Path,
    checkpoint_dir: str | Path,
    dataset_type: str = "mvtec_ad",
    **kwargs,
) -> Path:
    discovery = create_discovery(dataset_type, dataset_root)
    return train_category_with_discovery(category, discovery, checkpoint_dir, **kwargs)


def train_all(categories: list[str], **kwargs) -> dict[str, Path]:
    results = {}
    for cat in categories:
        try:
            results[cat] = train_category(cat, **kwargs)
        except Exception as e:
            print(f"[train] ERROR for '{cat}': {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PaDiM models for MVTec categories")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", default="mvtec_ad", help="Dataset ID (mvtec_ad or mvtec_ad2)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    datasets_cfg = cfg.get("datasets", {})

    if args.dataset not in datasets_cfg:
        print(f"[train] Unknown dataset '{args.dataset}'. Available: {list(datasets_cfg.keys())}")
        sys.exit(1)

    ds_cfg = datasets_cfg[args.dataset]
    dataset_root = ds_cfg["root"]
    dataset_type = ds_cfg["type"]
    checkpoint_dir = Path(cfg["paths"]["checkpoints"]) / args.dataset
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    discovery = create_discovery(dataset_type, dataset_root)

    if args.all:
        categories = discovery.list_categories()
    elif args.categories:
        categories = args.categories
    else:
        categories = train_cfg.get("default_categories", ["bottle", "capsule"])
        print(f"[train] No categories specified, using defaults: {categories}")

    for cat in categories:
        try:
            train_category_with_discovery(
                category=cat,
                discovery=discovery,
                checkpoint_dir=checkpoint_dir,
                backbone=model_cfg.get("backbone", "efficientnet_b0"),
                layers=model_cfg.get("layers", [1, 2, 3]),
                image_size=model_cfg.get("image_size", 224),
                target_channels=model_cfg.get("target_channels", 100),
                batch_size=train_cfg.get("batch_size", 32),
                epsilon=model_cfg.get("epsilon", 0.01),
                device=model_cfg.get("device", "auto"),
            )
        except Exception as e:
            print(f"[train] ERROR for '{cat}': {e}")
