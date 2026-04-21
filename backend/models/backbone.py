import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Dict, Optional


class FeatureExtractor:
    """
    Extracts multi-layer features from a pretrained backbone via forward hooks.

    Supported backbones: efficientnet_b0, mobilenet_v2.
    Features from all requested layers are interpolated to a common spatial resolution
    and concatenated along the channel dimension.

    Usage:
        extractor = FeatureExtractor(backbone="efficientnet_b0", layers=[1, 2, 3])
        features = extractor.extract(image_batch)  # [B, C_total, H, W]
        extractor.cleanup()
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        layers: List[int] = [1, 2, 3],
        device: str = "auto",
        pretrained: bool = True,
        backbone_state: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone_name = backbone
        self.layer_indices = layers
        self._features: Dict[int, torch.Tensor] = {}
        self._hooks = []

        self.model = self._load_backbone(backbone, pretrained=pretrained)
        if backbone_state is not None:
            self.model.load_state_dict(backbone_state)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._register_hooks()

        print(f"[FeatureExtractor] {backbone} on {self.device}, hooks on layers {layers}")

    def _load_backbone(self, name: str, pretrained: bool = True) -> nn.Module:
        if name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            return models.efficientnet_b0(weights=weights)
        elif name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            return models.mobilenet_v2(weights=weights)
        raise ValueError(f"Unsupported backbone '{name}'. Use 'efficientnet_b0' or 'mobilenet_v2'.")

    def _get_hook_layers(self) -> List[nn.Module]:
        if self.backbone_name == "efficientnet_b0":
            return [self.model.features[i] for i in self.layer_indices]
        elif self.backbone_name == "mobilenet_v2":
            mapping = {1: 3, 2: 6, 3: 13}
            return [self.model.features[mapping.get(i, i)] for i in self.layer_indices]
        raise ValueError(f"No layer mapping for '{self.backbone_name}'")

    def _register_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._features = {}
        for idx, layer in zip(self.layer_indices, self._get_hook_layers()):
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            self._features[layer_idx] = output
        return hook_fn

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        self._features = {}
        images = images.to(self.device)
        _ = self.model(images)

        feature_maps = []
        target_size = None

        for idx in sorted(self._features.keys()):
            feat = self._features[idx]
            if target_size is None:
                target_size = feat.shape[2:]
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            feature_maps.append(feat)

        return torch.cat(feature_maps, dim=1)

    def get_feature_dim(self) -> int:
        dummy = torch.randn(1, 3, 224, 224)
        return self.extract(dummy).shape[1]

    def cleanup(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
