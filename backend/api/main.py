import sys
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.api.routes import router
from backend.data.discovery import create_discovery
from backend.inference.engine import InferenceEngine


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. "
            "Run the server from the project root directory."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = _PROJECT_ROOT / "config.yaml"
    cfg = _load_config(config_path)

    app.state.config = cfg

    datasets_cfg = cfg.get("datasets", {})
    app.state.datasets_cfg = datasets_cfg

    app.state.discoveries = {
        ds_id: create_discovery(ds_cfg["type"], ds_cfg["root"])
        for ds_id, ds_cfg in datasets_cfg.items()
    }

    base_ckpt = _PROJECT_ROOT / cfg["paths"]["checkpoints"]
    app.state.engines = {
        ds_id: InferenceEngine(
            checkpoint_dir=base_ckpt / ds_id,
            device=cfg.get("model", {}).get("device", "auto"),
        )
        for ds_id in datasets_cfg
    }

    print("[API] Started. Datasets:", list(datasets_cfg.keys()))
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="MVTec Vision Lab",
        description="Browse MVTec datasets, train PaDiM models, and inspect XAI heatmaps.",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")

    frontend_dir = _PROJECT_ROOT / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


app = create_app()
