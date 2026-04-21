import base64
import cv2
import numpy as np
from PIL import Image


def apply_colormap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """float32 [H,W] in [0,1] → uint8 [H,W,3] RGB"""
    uint8 = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(uint8, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def blend_overlay(
    original_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Alpha-blend original image and colorized heatmap."""
    orig = original_rgb.astype(np.float32)
    heat = heatmap_rgb.astype(np.float32)
    blended = cv2.addWeighted(orig, 1.0 - alpha, heat, alpha, 0)
    return blended.astype(np.uint8)


def image_to_base64(image_rgb: np.ndarray) -> str:
    """Encode RGB uint8 numpy array as base64 PNG string."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf).decode("utf-8")


def load_image_rgb(image_path: str, size: int) -> np.ndarray:
    """Load image from path, resize to size×size, return uint8 RGB [H,W,3]."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def build_heatmap_overlay(
    image_path: str,
    heatmap: np.ndarray,
    image_size: int = 224,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> tuple[str, str]:
    """
    Given an image path and a float32 [H,W] heatmap in [0,1],
    returns (heatmap_b64, overlay_b64) as base64 PNG strings.
    """
    original = load_image_rgb(image_path, image_size)

    h_resized = cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    colored = apply_colormap(h_resized, colormap)
    overlay = blend_overlay(original, colored, alpha=alpha)

    heatmap_b64 = image_to_base64(colored)
    overlay_b64 = image_to_base64(overlay)
    return heatmap_b64, overlay_b64
