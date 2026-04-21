import torch
import torch.nn.functional as F
from typing import Tuple


def smooth_score_map(score_map: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    single = score_map.dim() == 2
    if single:
        score_map = score_map.unsqueeze(0)

    if sigma <= 0:
        return score_map.squeeze(0) if single else score_map

    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, device=score_map.device, dtype=score_map.dtype)
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()

    data = score_map.unsqueeze(1)
    data = F.pad(data, (radius, radius, 0, 0), mode="reflect")
    data = F.conv2d(data, kernel_1d.view(1, 1, 1, -1))
    data = F.pad(data, (0, 0, radius, radius), mode="reflect")
    data = F.conv2d(data, kernel_1d.view(1, 1, -1, 1))

    result = data.squeeze(1)
    return result.squeeze(0) if single else result


def upscale_score_map(score_map: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    single = score_map.dim() == 2
    if single:
        score_map = score_map.unsqueeze(0)

    upscaled = F.interpolate(
        score_map.unsqueeze(1),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    return upscaled.squeeze(0) if single else upscaled


def normalize_score_map(score_map: torch.Tensor) -> torch.Tensor:
    vmin = score_map.min()
    vmax = score_map.max()
    if vmax - vmin < 1e-8:
        return torch.zeros_like(score_map)
    return (score_map - vmin) / (vmax - vmin)


def process_score_map(
    raw_scores: torch.Tensor,
    target_size: Tuple[int, int] = (224, 224),
    sigma: float = 4.0,
    normalize: bool = True,
) -> torch.Tensor:
    smoothed = smooth_score_map(raw_scores, sigma=sigma)
    upscaled = upscale_score_map(smoothed, target_size=target_size)
    if normalize:
        upscaled = normalize_score_map(upscaled)
    return upscaled
