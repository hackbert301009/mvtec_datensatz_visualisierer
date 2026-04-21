from dataclasses import dataclass


@dataclass
class AnomalyResult:
    score: float      # max normalized Mahalanobis distance in [0, 1]
    is_anomaly: bool
    threshold: float


@dataclass
class InferenceOutput:
    category: str
    inference_time_ms: float
    anomaly: AnomalyResult
    heatmap_b64: str   # base64 PNG, JET colormap (blue=normal, red=anomaly)
    overlay_b64: str   # base64 PNG, original image + heatmap alpha-blended
