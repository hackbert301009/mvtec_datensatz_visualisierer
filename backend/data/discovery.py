from __future__ import annotations
from pathlib import Path

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

MVTEC_AD_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

MVTEC_AD2_CATEGORIES = [
    "can", "fabric", "fruit_jelly", "rice",
    "sheet_metal", "vial", "wallplugs", "walnuts",
]

MVTEC_AD3_CATEGORIES = [
    "bagel", "cable_gland", "carrot", "cookie",
    "dowel", "foam", "peach", "potato", "rope", "tire",
]

# AD2 defect-type keys use "__" as path separator to avoid URL issues
_AD2_SPLITS = [
    ("test_public__good",    "test · good",    False),
    ("test_public__bad",     "test · bad",     True),
    ("validation__good",     "val · good",     False),
    ("validation__bad",      "val · bad",      True),
    ("train__good",          "train · good",   False),
    ("test_private",         "private",        None),
    ("test_private_mixed",   "private·mixed",  None),
]


class BaseDiscovery:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def list_categories(self) -> list[str]:
        raise NotImplementedError

    def list_defect_types(self, category: str) -> list[dict]:
        """Return list of {id, label, is_anomaly} dicts."""
        raise NotImplementedError

    def list_images(self, category: str, defect_id: str) -> list[dict]:
        """Return list of {path, filename, is_anomaly} dicts."""
        raise NotImplementedError

    def get_train_paths(self, category: str) -> list[str]:
        raise NotImplementedError

    def get_good_test_images(self, category: str) -> list[str]:
        """Return paths of good/normal test images for score calibration."""
        raise NotImplementedError


class MVTecADDiscovery(BaseDiscovery):
    """MVTec Anomaly Detection v1 — category/test/{defect_type}/"""

    def list_categories(self) -> list[str]:
        if not self.root.exists():
            return []
        found = sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        ordered = [c for c in MVTEC_AD_CATEGORIES if c in found]
        extras = [c for c in found if c not in MVTEC_AD_CATEGORIES]
        return ordered + extras

    def list_defect_types(self, category: str) -> list[dict]:
        test_dir = self.root / category / "test"
        if not test_dir.exists():
            return []
        names = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
        if "good" in names:
            names.remove("good")
            names.append("good")
        return [
            {"id": name, "label": name, "is_anomaly": name != "good"}
            for name in names
        ]

    def list_images(self, category: str, defect_id: str) -> list[dict]:
        defect_dir = self.root / category / "test" / defect_id
        if not defect_dir.exists():
            return []
        images = sorted(
            p for p in defect_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        is_anomaly = defect_id != "good"
        return [{"path": str(p), "filename": p.name, "is_anomaly": is_anomaly} for p in images]

    def get_train_paths(self, category: str) -> list[str]:
        train_dir = self.root / category / "train" / "good"
        if not train_dir.exists():
            raise FileNotFoundError(f"Training dir not found: {train_dir}")
        paths = sorted(
            str(p) for p in train_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not paths:
            raise RuntimeError(f"No training images found in {train_dir}")
        return paths

    def get_good_test_images(self, category: str) -> list[str]:
        return [img["path"] for img in self.list_images(category, "good")]


class MVTecAD2Discovery(BaseDiscovery):
    """MVTec Anomaly Detection 2 — category/{split}/{good|bad}/"""

    def list_categories(self) -> list[str]:
        if not self.root.exists():
            return []
        found = sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        ordered = [c for c in MVTEC_AD2_CATEGORIES if c in found]
        extras = [c for c in found if c not in MVTEC_AD2_CATEGORIES]
        return ordered + extras

    def list_defect_types(self, category: str) -> list[dict]:
        cat_dir = self.root / category
        if not cat_dir.exists():
            return []
        result = []
        for key, label, is_anomaly in _AD2_SPLITS:
            parts = key.replace("__", "/").split("/")
            p = cat_dir
            for part in parts:
                p = p / part
            if p.exists():
                result.append({"id": key, "label": label, "is_anomaly": is_anomaly})
        return result

    def list_images(self, category: str, defect_id: str) -> list[dict]:
        parts = defect_id.replace("__", "/").split("/")
        image_dir = self.root / category
        for part in parts:
            image_dir = image_dir / part
        if not image_dir.exists():
            return []
        is_anomaly = next(
            (a for k, _, a in _AD2_SPLITS if k == defect_id),
            None,
        )
        images = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        return [{"path": str(p), "filename": p.name, "is_anomaly": is_anomaly} for p in images]

    def get_train_paths(self, category: str) -> list[str]:
        train_dir = self.root / category / "train" / "good"
        if not train_dir.exists():
            raise FileNotFoundError(f"Training dir not found: {train_dir}")
        paths = sorted(
            str(p) for p in train_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not paths:
            raise RuntimeError(f"No training images found in {train_dir}")
        return paths

    def get_good_test_images(self, category: str) -> list[str]:
        return [img["path"] for img in self.list_images(category, "test_public__good")]


class MVTecAD3Discovery(BaseDiscovery):
    """MVTec 3D Anomaly Detection — category/test/{defect}/rgb/ for images.
    Training uses category/train/good/rgb/.
    The archive extracts without a top-level folder, so root may be a shared
    data directory; we only return known AD3 category names to avoid collisions.
    """

    def list_categories(self) -> list[str]:
        if not self.root.exists():
            return []
        found = {d.name for d in self.root.iterdir() if d.is_dir()}
        return [c for c in MVTEC_AD3_CATEGORIES if c in found]

    def list_defect_types(self, category: str) -> list[dict]:
        test_dir = self.root / category / "test"
        if not test_dir.exists():
            return []
        names = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
        if "good" in names:
            names.remove("good")
            names.append("good")
        return [
            {"id": name, "label": name, "is_anomaly": name != "good"}
            for name in names
        ]

    def list_images(self, category: str, defect_id: str) -> list[dict]:
        rgb_dir = self.root / category / "test" / defect_id / "rgb"
        if not rgb_dir.exists():
            # Fallback: try direct folder (no rgb subfolder)
            rgb_dir = self.root / category / "test" / defect_id
        if not rgb_dir.exists():
            return []
        images = sorted(
            p for p in rgb_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        is_anomaly = defect_id != "good"
        return [{"path": str(p), "filename": p.name, "is_anomaly": is_anomaly} for p in images]

    def get_train_paths(self, category: str) -> list[str]:
        train_dir = self.root / category / "train" / "good" / "rgb"
        if not train_dir.exists():
            raise FileNotFoundError(f"Training dir not found: {train_dir}")
        paths = sorted(
            str(p) for p in train_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not paths:
            raise RuntimeError(f"No training images found in {train_dir}")
        return paths

    def get_good_test_images(self, category: str) -> list[str]:
        return [img["path"] for img in self.list_images(category, "good")]


def create_discovery(dataset_type: str, root: str | Path) -> BaseDiscovery:
    if dataset_type == "mvtec_ad":
        return MVTecADDiscovery(root)
    if dataset_type == "mvtec_ad2":
        return MVTecAD2Discovery(root)
    if dataset_type == "mvtec_ad3":
        return MVTecAD3Discovery(root)
    raise ValueError(f"Unknown dataset type: {dataset_type!r}")


# Backward compat alias
DatasetDiscovery = MVTecADDiscovery
