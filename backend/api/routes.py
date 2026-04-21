import threading
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()

# Background training state: category → "training" | "done" | "error:<msg>"
_train_status: dict[str, str] = {}
_train_lock = threading.Lock()


# ── Request / Response models ────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    category: str
    image_path: str


class InferenceResponse(BaseModel):
    score: float          # normalized against training p95; >1.0 = more anomalous than 95% of training
    is_anomaly: bool
    threshold: float
    heatmap_b64: str
    overlay_b64: str
    inference_time_ms: float


class TrainStatusResponse(BaseModel):
    category: str
    status: str         # "training" | "done" | "error:<msg>" | "not_started"
    trained: bool


# ── Dataset endpoints ────────────────────────────────────────────────────────

@router.get("/datasets")
async def list_datasets() -> list[str]:
    return ["mvtec_anomaly_detection"]


@router.get("/categories")
async def list_categories(request: Request) -> list[str]:
    return request.app.state.discovery.list_categories()


@router.get("/defect-types")
async def list_defect_types(category: str, request: Request) -> list[str]:
    types = request.app.state.discovery.list_defect_types(category)
    if not types:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
    return types


@router.get("/images")
async def list_images(category: str, defect: str, request: Request) -> list[dict]:
    return request.app.state.discovery.list_images(category, defect)


# ── Image serving ─────────────────────────────────────────────────────────────

@router.get("/image")
async def serve_image(path: str, request: Request) -> FileResponse:
    dataset_root = Path(request.app.state.config["dataset"]["root"])
    image_path = Path(path)

    # Security: only serve files inside the configured dataset root
    try:
        image_path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Path outside dataset root")

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(str(image_path))


# ── Inference ─────────────────────────────────────────────────────────────────

@router.post("/inference", response_model=InferenceResponse)
async def run_inference(body: InferenceRequest, request: Request) -> InferenceResponse:
    engine = request.app.state.engine
    cfg = request.app.state.config
    threshold = cfg.get("inference", {}).get("threshold", 0.5)
    alpha = cfg.get("inference", {}).get("overlay_alpha", 0.5)

    if not engine.is_trained(body.category):
        raise HTTPException(
            status_code=400,
            detail=f"Model for '{body.category}' not trained yet. "
                   f"Call GET /api/train?categories={body.category} first.",
        )

    try:
        result = engine.predict(
            category=body.category,
            image_path=body.image_path,
            threshold=threshold,
            overlay_alpha=alpha,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return InferenceResponse(
        score=result.anomaly.score,
        is_anomaly=result.anomaly.is_anomaly,
        threshold=result.anomaly.threshold,
        heatmap_b64=result.heatmap_b64,
        overlay_b64=result.overlay_b64,
        inference_time_ms=result.inference_time_ms,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def _run_training_bg(category: str, cfg: dict) -> None:
    """Background task — trains a single category and updates _train_status."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from backend.training.train import train_category

    with _train_lock:
        _train_status[category] = "training"
    try:
        train_category(
            category=category,
            dataset_root=cfg["dataset"]["root"],
            checkpoint_dir=cfg["paths"]["checkpoints"],
            backbone=cfg.get("model", {}).get("backbone", "efficientnet_b0"),
            layers=cfg.get("model", {}).get("layers", [1, 2, 3]),
            image_size=cfg.get("model", {}).get("image_size", 224),
            target_channels=cfg.get("model", {}).get("target_channels", 100),
            batch_size=cfg.get("training", {}).get("batch_size", 32),
            epsilon=cfg.get("model", {}).get("epsilon", 0.01),
            device=cfg.get("model", {}).get("device", "auto"),
        )
        with _train_lock:
            _train_status[category] = "done"
    except Exception as e:
        with _train_lock:
            _train_status[category] = f"error:{e}"


@router.get("/train")
async def trigger_training(
    categories: str,
    background_tasks: BackgroundTasks,
    request: Request,
) -> list[TrainStatusResponse]:
    engine = request.app.state.engine
    cfg = request.app.state.config
    cat_list = [c.strip() for c in categories.split(",") if c.strip()]

    responses = []
    for cat in cat_list:
        with _train_lock:
            status = _train_status.get(cat)

        if status == "training":
            responses.append(TrainStatusResponse(category=cat, status="training", trained=False))
        elif engine.is_trained(cat) and status != "training":
            with _train_lock:
                _train_status[cat] = "done"
            responses.append(TrainStatusResponse(category=cat, status="done", trained=True))
        else:
            background_tasks.add_task(_run_training_bg, cat, cfg)
            with _train_lock:
                _train_status[cat] = "training"
            responses.append(TrainStatusResponse(category=cat, status="training", trained=False))

    return responses


@router.get("/train/status")
async def training_status(category: str, request: Request) -> TrainStatusResponse:
    engine = request.app.state.engine
    with _train_lock:
        status = _train_status.get(category, "not_started")

    trained = engine.is_trained(category)
    if trained and status not in ("training",):
        status = "done"

    return TrainStatusResponse(category=category, status=status, trained=trained)


@router.get("/models")
async def list_trained_models(request: Request) -> list[str]:
    return request.app.state.engine.trained_categories()
