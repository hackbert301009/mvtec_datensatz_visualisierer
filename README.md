# MVTec Anomaly Detection Viewer

An open-source tool for exploring the [MVTec Anomaly Detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad), training a **PaDiM** anomaly detection model, and inspecting per-image **XAI heatmaps** directly in the browser.

![Layout: categories → defect types → image grid → heatmap modal]()

---

## Features

- **Auto-discovery** of all 15 MVTec categories from a local dataset directory
- **PaDiM training** — fits multivariate Gaussians on normal training images; no GPU required
- **XAI heatmaps** — per-pixel Mahalanobis distance maps (JET colormap: blue = normal, red = anomaly)
- **Overlay view** — original image + heatmap alpha-blended
- **REST API** (FastAPI) with automatic OpenAPI docs at `/docs`
- **Browser UI** — 3-column layout: categories · defect types · image grid + detail modal
- **In-browser training trigger** — click "Train Selected" to start training without leaving the UI
- Extensible: add new backbones or model types by implementing the `InferenceEngine` interface

---

## Project Structure

```
mvtec_datensatz_visualisierer/
├── backend/
│   ├── data/discovery.py       # scans dataset directory
│   ├── models/
│   │   ├── padim.py            # PaDiM core (Mahalanobis distance)
│   │   └── backbone.py         # EfficientNet-B0 / MobileNetV2 feature extractor
│   ├── inference/
│   │   ├── engine.py           # lazy model cache + predict()
│   │   ├── heatmap.py          # smooth → upscale → normalize score maps
│   │   ├── overlay.py          # colormap + alpha blend → base64
│   │   └── pipeline_utils.py   # save/load checkpoints, channel subsampling
│   ├── training/train.py       # CLI training script
│   └── api/
│       ├── main.py             # FastAPI app factory
│       └── routes.py           # all endpoints
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── checkpoints/                # .pt files saved here after training
├── config.yaml                 # dataset path, model settings
└── README.md
```

---

## Setup

### 1. Prerequisites

- Python ≥ 3.10
- The [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) downloaded locally

### 2. Install dependencies

```bash
cd mvtec_datensatz_visualisierer
pip install -r backend/requirements.txt
```

### 3. Configure dataset path

Edit `config.yaml`:

```yaml
dataset:
  root: /path/to/mvtec_anomaly_detection   # ← change this
```

Expected dataset structure:

```
mvtec_anomaly_detection/
├── bottle/
│   ├── train/
│   │   └── good/          ← normal training images
│   ├── test/
│   │   ├── broken_large/  ← defect images
│   │   ├── broken_small/
│   │   ├── contamination/
│   │   └── good/
│   └── ground_truth/
│       ├── broken_large/
│       └── ...
├── capsule/
│   └── ...
└── ...
```

---

## Training

Train a model for specific categories:

```bash
# Two categories
python -m backend.training.train --categories bottle capsule

# Single category
python -m backend.training.train --categories hazelnut

# All available categories
python -m backend.training.train --all

# Custom config path
python -m backend.training.train --categories bottle --config /path/to/config.yaml
```

Training output is saved to `checkpoints/{category}.pt`. On CPU, expect roughly **3–8 minutes per category** (varies by image count and hardware).

---

## Start the Server

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** in your browser.

API documentation is available at **http://localhost:8000/docs**.

---

## Using the UI

1. **Select a category** from the left panel (e.g. `bottle`)
2. If not trained, click **Train Selected** — a badge will update to "trained" when done
3. **Select a defect type** from the middle panel (e.g. `broken_large`)
4. **Click an image thumbnail** in the right panel
5. The detail modal opens with three tabs:
   - **Original** — source image
   - **Heatmap** — anomaly score map (blue = normal, red = anomaly)
   - **Overlay** — original + heatmap blended at 50% opacity
6. The score and anomaly verdict are shown in the modal header

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/datasets` | List dataset names |
| GET | `/api/categories` | List available MVTec categories |
| GET | `/api/defect-types?category=bottle` | Defect types for a category |
| GET | `/api/images?category=bottle&defect=broken_large` | Images for a category + defect |
| GET | `/api/image?path=...` | Serve a single image file |
| POST | `/api/inference` | Run PaDiM + generate heatmaps |
| GET | `/api/train?categories=bottle,capsule` | Trigger background training |
| GET | `/api/train/status?category=bottle` | Check training status |
| GET | `/api/models` | List trained categories |

---

## Configuration Reference (`config.yaml`)

```yaml
dataset:
  root: /path/to/mvtec_anomaly_detection

model:
  backbone: efficientnet_b0   # or mobilenet_v2
  layers: [1, 2, 3]           # which backbone layers to hook
  image_size: 224
  target_channels: 100        # random channel subsampling (reduces memory)
  epsilon: 0.01               # covariance regularization
  sigma: 4.0                  # Gaussian smoothing of score maps
  device: auto                # auto | cpu | cuda

training:
  batch_size: 32
  default_categories:
    - bottle
    - capsule

inference:
  threshold: 1.0              # score ≥ threshold → anomaly
                              # (score is p95-normalized: >1.0 means more anomalous
                              #  than 95% of normal reference images)
  overlay_alpha: 0.5          # heatmap opacity in overlay

paths:
  checkpoints: checkpoints
  results: results
```

---

## How It Works

PaDiM (Patch Distribution Modeling) is a training-free anomaly detection method:

1. **Feature extraction** — a pretrained EfficientNet-B0 processes each training image; features from three intermediate layers are extracted via forward hooks and concatenated.
2. **Distribution fitting** — for each spatial patch position, a multivariate Gaussian is fitted over all training features (only normal images).
3. **Inference** — at test time, the Mahalanobis distance from each patch to its fitted distribution is computed; high distance = anomaly.
4. **Heatmap** — the 14×14 distance map is Gaussian-smoothed, upscaled to 224×224, normalized per-image to [0, 1] for visualization, and colorized with JET colormap (blue = normal region, red = anomalous region).

No gradient-based training — fitting takes minutes instead of hours.

**Score normalization**: after fitting, the model computes max Mahalanobis scores on the `test/good` images and stores the 95th percentile (`p95`). At inference, `score = raw_max / p95`. A score > 1.0 means the image is more anomalous than 95% of known-good reference images. The default threshold is 1.0.

---

## Roadmap

- [ ] Support all 15 MVTec categories simultaneously
- [ ] AUROC / pixel-AUROC evaluation metrics
- [ ] Side-by-side comparison of multiple images
- [ ] Export heatmaps as PNG/PDF report
- [ ] Additional models: FastFlow, EfficientAD, SimpleNet
- [ ] Threshold auto-calibration from validation set
- [ ] Docker container for easy deployment

---

## Requirements

```
fastapi, uvicorn, pydantic
torch, torchvision
opencv-python-headless, Pillow, numpy
pyyaml
```

See [backend/requirements.txt](backend/requirements.txt) for pinned versions.

---

## License

MIT
