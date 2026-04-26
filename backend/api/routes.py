import threading
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()

_train_status: dict[str, str] = {}
_train_lock = threading.Lock()


# ── Models ───────────────────────────────────────────────────────────────────

class DefectType(BaseModel):
    id: str
    label: str
    is_anomaly: bool | None


class DatasetInfo(BaseModel):
    id: str
    name: str
    available: bool
    category_count: int


class InferenceRequest(BaseModel):
    dataset: str = "mvtec_ad"
    category: str
    image_path: str


class InferenceResponse(BaseModel):
    score: float
    is_anomaly: bool
    threshold: float
    heatmap_b64: str
    overlay_b64: str
    inference_time_ms: float


class TrainStatusResponse(BaseModel):
    category: str
    status: str
    trained: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_discovery(request: Request, dataset: str):
    disc = request.app.state.discoveries.get(dataset)
    if disc is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found")
    return disc


def _get_engine(request: Request, dataset: str):
    engine = request.app.state.engines.get(dataset)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' has no inference engine")
    return engine


def _train_key(dataset: str, category: str) -> str:
    return f"{dataset}::{category}"


# ── Dataset endpoints ─────────────────────────────────────────────────────────

@router.get("/datasets")
async def list_datasets(request: Request) -> list[DatasetInfo]:
    cfg = request.app.state.datasets_cfg
    discoveries = request.app.state.discoveries
    result = []
    for ds_id, ds_cfg in cfg.items():
        disc = discoveries.get(ds_id)
        cats = disc.list_categories() if disc else []
        result.append(DatasetInfo(
            id=ds_id,
            name=ds_cfg["name"],
            available=disc is not None and disc.root.exists(),
            category_count=len(cats),
        ))
    return result


@router.get("/categories")
async def list_categories(request: Request, dataset: str = "mvtec_ad") -> list[str]:
    return _get_discovery(request, dataset).list_categories()


@router.get("/defect-types")
async def list_defect_types(
    category: str, request: Request, dataset: str = "mvtec_ad"
) -> list[DefectType]:
    types = _get_discovery(request, dataset).list_defect_types(category)
    if not types:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
    return [DefectType(**t) for t in types]


@router.get("/images")
async def list_images(
    category: str, defect: str, request: Request, dataset: str = "mvtec_ad"
) -> list[dict]:
    return _get_discovery(request, dataset).list_images(category, defect)


# ── Image serving ─────────────────────────────────────────────────────────────

@router.get("/image")
async def serve_image(path: str, request: Request) -> FileResponse:
    image_path = Path(path)

    # Security: only serve files inside a configured dataset root
    allowed_roots = [
        Path(ds_cfg["root"]).resolve()
        for ds_cfg in request.app.state.datasets_cfg.values()
    ]
    resolved = image_path.resolve()
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(status_code=403, detail="Path outside dataset roots")

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(str(image_path))


# ── Inference ─────────────────────────────────────────────────────────────────

@router.post("/inference", response_model=InferenceResponse)
async def run_inference(body: InferenceRequest, request: Request) -> InferenceResponse:
    engine = _get_engine(request, body.dataset)
    cfg = request.app.state.config
    threshold = cfg.get("inference", {}).get("threshold", 1.0)
    alpha = cfg.get("inference", {}).get("overlay_alpha", 0.5)

    if not engine.is_trained(body.category):
        raise HTTPException(
            status_code=400,
            detail=f"Model for '{body.category}' not trained yet.",
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

def _run_training_bg(dataset: str, category: str, cfg: dict, dataset_type: str, dataset_root: str, checkpoint_dir: str) -> None:
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).resolve().parents[2]))

    from backend.data.discovery import create_discovery
    from backend.training.train import train_category_with_discovery

    key = _train_key(dataset, category)
    with _train_lock:
        _train_status[key] = "training"
    try:
        discovery = create_discovery(dataset_type, dataset_root)
        train_category_with_discovery(
            category=category,
            discovery=discovery,
            checkpoint_dir=checkpoint_dir,
            backbone=cfg.get("model", {}).get("backbone", "efficientnet_b0"),
            layers=cfg.get("model", {}).get("layers", [1, 2, 3]),
            image_size=cfg.get("model", {}).get("image_size", 224),
            target_channels=cfg.get("model", {}).get("target_channels", 100),
            batch_size=cfg.get("training", {}).get("batch_size", 32),
            epsilon=cfg.get("model", {}).get("epsilon", 0.01),
            device=cfg.get("model", {}).get("device", "auto"),
        )
        with _train_lock:
            _train_status[key] = "done"
    except Exception as e:
        with _train_lock:
            _train_status[key] = f"error:{e}"


@router.get("/train")
async def trigger_training(
    categories: str,
    background_tasks: BackgroundTasks,
    request: Request,
    dataset: str = "mvtec_ad",
) -> list[TrainStatusResponse]:
    engine = _get_engine(request, dataset)
    cfg = request.app.state.config
    ds_cfg = request.app.state.datasets_cfg.get(dataset, {})

    import os
    project_root = str(request.app.state.config.get("_project_root", ""))
    base_ckpt_dir = os.path.join(
        str(Path(__file__).resolve().parents[2]),
        cfg["paths"]["checkpoints"],
        dataset,
    )

    cat_list = [c.strip() for c in categories.split(",") if c.strip()]
    responses = []
    for cat in cat_list:
        key = _train_key(dataset, cat)
        with _train_lock:
            status = _train_status.get(key)

        if status == "training":
            responses.append(TrainStatusResponse(category=cat, status="training", trained=False))
        elif engine.is_trained(cat) and status != "training":
            with _train_lock:
                _train_status[key] = "done"
            responses.append(TrainStatusResponse(category=cat, status="done", trained=True))
        else:
            background_tasks.add_task(
                _run_training_bg,
                dataset, cat, cfg,
                ds_cfg.get("type", "mvtec_ad"),
                ds_cfg.get("root", ""),
                base_ckpt_dir,
            )
            with _train_lock:
                _train_status[key] = "training"
            responses.append(TrainStatusResponse(category=cat, status="training", trained=False))

    return responses


@router.get("/train/status")
async def training_status(
    category: str, request: Request, dataset: str = "mvtec_ad"
) -> TrainStatusResponse:
    engine = _get_engine(request, dataset)
    key = _train_key(dataset, category)
    with _train_lock:
        status = _train_status.get(key, "not_started")

    trained = engine.is_trained(category)
    if trained and status not in ("training",):
        status = "done"

    return TrainStatusResponse(category=category, status=status, trained=trained)


@router.get("/models")
async def list_trained_models(
    request: Request, dataset: str = "mvtec_ad"
) -> list[str]:
    return _get_engine(request, dataset).trained_categories()
