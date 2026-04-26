# MVTec Vision Lab

An open-source browser tool for exploring **MVTec AD** and **MVTec AD 2** datasets, training **PaDiM** anomaly detection models, and inspecting per-image **XAI heatmaps** — all without leaving your browser.

---

## Features

- **Two datasets** — MVTec AD (15 categories) and MVTec AD 2 (8 categories) with one-click switching
- **PaDiM training** — fits multivariate Gaussians on normal training images; no GPU required
- **XAI heatmaps** — per-pixel Mahalanobis distance maps (blue = normal, red = anomaly)
- **Overlay view** — original image + heatmap alpha-blended
- **Animated score bar** — visual anomaly score gauge with threshold marker
- **REST API** (FastAPI) with OpenAPI docs at `/docs`
- **Beautiful dark UI** — 3-column layout with glassmorphic modal and smooth animations
- **In-browser training** — click "Train Selected" to start without leaving the UI

---

## Project Structure

```
mvtec_datensatz_visualisierer/
├── backend/
│   ├── data/discovery.py       # MVTecADDiscovery + MVTecAD2Discovery
│   ├── models/
│   │   ├── padim.py            # PaDiM (Mahalanobis distance)
│   │   └── backbone.py         # EfficientNet-B0 feature extractor
│   ├── inference/
│   │   ├── engine.py           # lazy model cache + predict()
│   │   ├── heatmap.py          # score map → smooth → normalize
│   │   ├── overlay.py          # JET colormap + alpha blend → base64
│   │   └── pipeline_utils.py   # checkpoint save/load, channel subsampling
│   ├── training/train.py       # CLI + in-process training
│   └── api/
│       ├── main.py             # FastAPI app factory
│       └── routes.py           # all endpoints
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── checkpoints/
│   ├── mvtec_ad/               # .pt files for AD v1
│   └── mvtec_ad2/              # .pt files for AD 2
├── config.yaml
└── README.md
```

---

## Setup

### 1. Prerequisites

- Python ≥ 3.10
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset (15 categories)
- [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) dataset (8 categories, optional)

### 2. Install dependencies

```bash
cd mvtec_datensatz_visualisierer
pip install -r backend/requirements.txt
```

### 3. Configure dataset paths

Edit `config.yaml`:

```yaml
datasets:
  mvtec_ad:
    name: "MVTec AD"
    root: /path/to/mvtec_anomaly_detection   # ← change
    type: mvtec_ad
  mvtec_ad2:
    name: "MVTec AD 2"
    root: /path/to/mvtec_ad_2                # ← change (optional)
    type: mvtec_ad2
```

**MVTec AD structure** (`category/train/good/`, `category/test/{defect}/`):
```
mvtec_anomaly_detection/
├── bottle/
│   ├── train/good/
│   └── test/{broken_large,broken_small,contamination,good}/
└── ...
```

**MVTec AD 2 structure** (`category/train/good/`, `category/test_public/{good,bad}/`):
```
mvtec_ad_2/
├── can/
│   ├── train/good/
│   ├── validation/{good,bad}/
│   ├── test_public/{good,bad}/
│   ├── test_private/
│   └── test_private_mixed/
└── ...
```

---

## Start the Server

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** — API docs at **http://localhost:8000/docs**.

---

## Training

```bash
# MVTec AD — specific categories
python -m backend.training.train --categories bottle capsule

# MVTec AD 2
python -m backend.training.train --dataset mvtec_ad2 --categories can fabric

# Train all categories of a dataset
python -m backend.training.train --dataset mvtec_ad --all

# Training via the UI: select a category → click "Train Selected"
```

Checkpoints are saved to `checkpoints/{dataset_id}/{category}.pt`.  
Training takes **3–8 minutes per category** on CPU.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/datasets` | List all configured datasets |
| GET | `/api/categories?dataset=mvtec_ad` | Categories for a dataset |
| GET | `/api/defect-types?category=bottle&dataset=mvtec_ad` | Defect types / splits |
| GET | `/api/images?category=bottle&defect=good&dataset=mvtec_ad` | Image list |
| GET | `/api/image?path=...` | Serve an image file |
| POST | `/api/inference` | Run PaDiM + generate heatmaps |
| GET | `/api/train?categories=bottle&dataset=mvtec_ad` | Start background training |
| GET | `/api/train/status?category=bottle&dataset=mvtec_ad` | Training status |
| GET | `/api/models?dataset=mvtec_ad` | List trained categories |

---
test
## How It Works

**PaDiM** (Patch Distribution Modeling) is a training-free anomaly detection method:

1. **Feature extraction** — pretrained EfficientNet-B0 processes each normal training image; features from three intermediate layers are concatenated per spatial position.
2. **Distribution fitting** — a multivariate Gaussian is fitted at each patch location over all training features.
3. **Inference** — Mahalanobis distance from each patch to its Gaussian; high distance = anomaly.
4. **Score normalization** — the 95th percentile of max-distances on `test/good` images becomes the threshold. Score > 1.0 means more anomalous than 95% of known-good images.
5. **Heatmap** — 14×14 distance map → Gaussian-smoothed → upscaled to 224×224 → JET colormap.

---

## Requirements

```
fastapi  uvicorn  pydantic
torch  torchvision
opencv-python-headless  Pillow  numpy
pyyaml
```

---

## License

MIT
