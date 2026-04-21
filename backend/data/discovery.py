from pathlib import Path


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


class DatasetDiscovery:
    """Scans a local MVTec AD directory and exposes categories, defect types, and image paths."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def list_categories(self) -> list[str]:
        if not self.root.exists():
            return []
        found = sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        # Preserve canonical MVTec order for categories that exist
        ordered = [c for c in MVTEC_CATEGORIES if c in found]
        extras = [c for c in found if c not in MVTEC_CATEGORIES]
        return ordered + extras

    def list_defect_types(self, category: str) -> list[str]:
        test_dir = self.root / category / "test"
        if not test_dir.exists():
            return []
        types = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
        # Put "good" last so defects appear first in the UI
        if "good" in types:
            types.remove("good")
            types.append("good")
        return types

    def list_images(self, category: str, defect_type: str) -> list[dict]:
        defect_dir = self.root / category / "test" / defect_type
        if not defect_dir.exists():
            return []
        images = sorted(
            p for p in defect_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        is_anomaly = defect_type != "good"
        return [
            {"path": str(p), "filename": p.name, "is_anomaly": is_anomaly}
            for p in images
        ]

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
