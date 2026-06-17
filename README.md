# Hatched Area Violation Detection

End-to-end computer vision pipeline that detects vehicles entering hatched road markings using a fine-tuned YOLOv8 model, ByteTrack, polygon-based zone checks, and reads Turkish license plates on confirmed violations.

> **Graduation Thesis** — Sertaç Akalın, Istanbul Arel University, Computer Engineering (2026)
> **Advisor:** Prof. Dr. Haluk Gümüşkaya

---

## What This System Does

```
Traffic Video  →  YOLOv8 (best_v4.pt)  →  ByteTrack  →  Zone Check (Shapely)
                                                              ↓
                          State Machine + Trajectory + Severity Scoring
                                                              ↓
                            Plate Detection + PaddleOCR + TR Format Validation
                                                              ↓
                                      Violation Report (SQLite + video + plates)
```

1. **Detects** cars, trucks, buses, motorcycles in every frame (YOLOv8m fine-tune, `best_v4.pt`)
2. **Tracks** them across frames with persistent IDs (ByteTrack)
3. **Checks** whether the bottom-center point of each vehicle enters a manually defined hatched area polygon (Shapely)
4. **Confirms** the violation only after the vehicle stays inside for at least N frames, with a per-track lock + cooldown and spatial deduplication — eliminates false positives from bounding-box jitter, brief edge contact, and track-ID churn
5. **Classifies** each violation (KAYNAK / SEYİR / KENAR_TEMASI / DİĞER) and assigns a 0–100 severity score
6. **Reads the license plate** of confirmed violators (YOLOv8n plate detector + PaddleOCR + Turkish 81-province format validation)
7. **Logs** the event to SQLite with vehicle crop, plate crop, frame snapshot, and timestamp

## Original Contribution

The detection (YOLOv8) and tracking (ByteTrack) components are off-the-shelf. The original work in this thesis is:

1. **Fine-tuning YOLOv8m on a custom Istanbul traffic dataset** — the production model `best_v4.pt` is a warm-start from `best_v3.pt` (the `final_v3` 100-epoch fine-tune) with 365 additional manually verified drone frames, for a 6,948-image training set. It reaches **mAP@50 = 0.896** on the held-out test set (Roboflow community sub-set 0.875, auto-labeled CCTV 0.918, drone 0.822). Evaluated against the COCO baseline (`yolov8s.pt`), an earlier yolov8s fine-tune (`best.pt`), and the v3 model — see `weights/README.md`, `scripts/compare_v3_v4_metrics.py`, and `docs/thesis/chapter7_experiments_evaluation.md`.
2. **Temporal filtering with a four-state state machine** (OUTSIDE → ENTERING → INSIDE → VIOLATION) that requires N consecutive frames inside the zone plus a per-track lock + cooldown window, hardened with **spatial deduplication** (suppresses a new violation if the same x/y region fired within a recent window) to absorb track-ID churn — eliminates bounding-box jitter false positives that a simple inside/outside check produces.
3. **Trajectory + severity scoring pipeline** that classifies each confirmed violation as `KAYNAK` (lane change / diagonal crossing) / `SEYİR` (cruising inside the zone) / `KENAR_TEMASI` (edge contact, likely false positive) / `DİĞER` and assigns a 0–100 score, enabling threshold-based false-positive filtering downstream.
4. **Two-stage plate recognition** with a fine-tuned YOLOv8n plate detector (`plate.pt`, dataset `TR-PLAKA-1`), best-frame voting from a per-track ring buffer, PaddleOCR (EasyOCR optional) with upscale + unsharp pre-processing, and Turkish 81-province plate format validation (`src/plate/tr_plate.py`).
5. **A hand-annotated Istanbul hatched area video dataset** with ground-truth violation timestamps, used for an empirical pipeline-level field test of **P = 0.889, R = 0.800, F1 = 0.842** (`docs/thesis/chapter8_error_analysis.md`, `scripts/empirical_pipeline_eval.py`).

## Architecture

```
src/
├── core/            # Config, frame I/O, data models, visualization
├── detection/       # YOLOv8 vehicle detector wrapper
├── tracking/        # ByteTrack wrapper (single tracker)
├── zones/           # Polygon zone manager, ROI selector
├── violation/       # State machine, trajectory, severity
├── plate/           # Plate detector + PaddleOCR/EasyOCR + TR format validation
├── storage/         # SQLite schema, migrations, logger
└── pipeline/        # Main orchestrator + factory

scripts/             # CLI tools (run pipeline, select ROI, evaluate, compare v3/v4, report)
scripts/notebooks/   # Colab notebooks (frame extraction → fine-tuning → evaluation)
configs/             # YAML config + ByteTrack tuning + zone polygons
app.py               # Gradio web demo (with optional plate recognition)
archive/             # Removed prototype code kept only for reference
```

### State Machine

```
OUTSIDE --[enters zone]--> ENTERING --[>= min_frames_in_zone]--> INSIDE --[confirmed]--> VIOLATION
   ^                                                                                        |
   +---------------------------[exits zone for >= exit_frames]----------------------------+
```

### Severity Scoring Formula

```
score = 0.30 * duration_norm + 0.25 * distance_norm + 0.30 * depth_norm + 0.15 * angle_norm

Level:    0–25 DÜŞÜK  |  25–50 ORTA  |  50–75 YÜKSEK  |  75–100 KRİTİK
Type:     KAYNAK       (short diagonal crossing / lane change)
          SEYİR        (extended travel inside zone)
          KENAR_TEMASI (shallow penetration, likely false positive)
          DİĞER        (uncategorized)
```

## Tech Stack

| Component          | Technology                              |
|--------------------|-----------------------------------------|
| Vehicle Detection  | YOLOv8m fine-tune `best_v4.pt` (Ultralytics) |
| Plate Detection    | YOLOv8n fine-tune `plate.pt` (Ultralytics) |
| Plate OCR          | PaddleOCR (default) / EasyOCR (optional)|
| Plate Validation   | Custom TR format regex + 81 province codes |
| Fine-tuning        | Roboflow + Google Colab (Tesla T4)      |
| Tracking           | ByteTrack (Ultralytics integration)     |
| Geometry           | Shapely                                 |
| Storage            | SQLite (WAL mode, idempotent migrations)|
| Web Demo           | Gradio + Plotly                         |
| Plotting           | Matplotlib + Seaborn + Plotly           |

## Installation

```bash
git clone <repo-url>
cd hatched-area-violation-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Define the hatched area polygon by clicking on the first frame of the video
python scripts/select_roi.py \
  --video data/videos/test/cam1.mp4 \
  --output configs/zones/cam1.json

# 2. Run the detection pipeline on the video
python scripts/run_pipeline.py \
  --config configs/config.yaml \
  --video data/videos/test/cam1.mp4 \
  --zone  configs/zones/cam1.json

# 3. Evaluate detected violations against manual ground truth
python scripts/evaluate_with_ground_truth.py \
  --video data/videos/test/cam1.mp4 \
  --ground-truth data/ground_truth/cam1.json

# 4. Launch the Gradio web demo
python app.py  # http://localhost:7860
```

### Google Colab (for GPU fine-tuning)

All notebooks live under `scripts/notebooks/`:

```
00_frame_extraction.ipynb            → extract frames from raw videos
02_vehicle_detection_baseline.ipynb  → pretrained YOLOv8 baseline on your videos
03_vehicle_detection_finetuning.ipynb→ fine-tune YOLOv8 on Roboflow dataset
04_tracking_comparison.ipynb         → ByteTrack sanity check
05_train_mobese_v3.ipynb             → v3 fine-tune (yolov8m, final_v3)
06_train_plate_detector.ipynb        → plate detector (yolov8n, TR-PLAKA-1)
07_violation_detection_eval.ipynb    → end-to-end pipeline on a test video
08_train_mobese_v4.ipynb             → v4 warm-start (best_v3 + drone frames)
09_train_mobese_v5.ipynb             → v5 experiment (work in progress)
```

## Data Layout

| Item              | Location                        |
|-------------------|---------------------------------|
| Raw videos        | `data/videos/raw/`              |
| Test video cuts   | `data/videos/test/`             |
| Annotated frames  | `data/datasets/vehicle_detection/` (Roboflow export) |
| Ground truth      | `data/ground_truth/*.json`      |
| Pretrained weights | `weights/yolov8s.pt`           |
| Fine-tuned weights (v1, legacy) | `weights/best.pt`     |
| Fine-tuned weights (v3) | `weights/best_v3.pt`      |
| Fine-tuned weights (v4, production) | `weights/best_v4.pt` |
| Plate detector weights | `weights/plate.pt`         |
| Zone polygons     | `configs/zones/*.json`          |

## Dataset and Citation

Large data artifacts are intentionally not tracked in this repository. Raw videos,
Roboflow exports, model weights, and generated results live outside git because
they are either external or reproducible.

Dataset inventory used in the thesis:

| Source | Images | Annotation type |
|--------|--------|-----------------|
| Roboflow community source set | 2,919 | community/manual |
| Pseudo-labeled day/night/evening frames | 3,664 | YOLOv8 auto-label |
| Hand-verified cam10/cam11 frames | 365 | Label Studio manual |
| **Total thesis inventory** | **6,948** | mixed |

The Roboflow upload package prepared from the recovered local data contains
4,029 images and 74,279 bounding boxes with classes `bus`, `car`, `motorcycle`,
and `truck`. After publishing the dataset on Roboflow Universe, use the
Universe project's **Cite This Project** BibTeX for the dataset citation. Use
[`CITATION.cff`](CITATION.cff) to cite this software repository.

## Thesis Scope

This project is deliberately scoped to a **working end-to-end system**. The following items are **in scope** and shipped:

- Manual hatched area definition (click-to-polygon ROI tool)
- YOLOv8m fine-tuning on a custom Istanbul traffic dataset (`best_v4.pt`)
- ByteTrack vehicle tracking
- Four-state temporal state machine + per-track lock + cooldown + spatial deduplication
- Severity scoring (0–100) + violation-type classification
- Two-stage Turkish license plate recognition with 81-province format validation
- SQLite persistence + idempotent schema migrations
- Gradio web demo (fixed-camera deployment, optional plate recognition)
- Quantitative evaluation against hand-annotated ground truth

The following items are **explicitly out of scope**:

- Automatic hatched area detection (no segmentation model)
- Homography-based speed estimation
- Alternative trackers other than ByteTrack (BoT-SORT/DeepSORT scaffolding exists in `configs/` but is not evaluated)
- Multi-camera fusion / re-identification across cameras
- Dynamic zone tracking for moving cameras (the drone/PTZ feature-matching prototype was removed; deployment assumes a fixed camera)

Some earlier prototype files are retained under [`archive/`](archive/) only as historical reference and are not part of the evaluated system.

## Project Status

- [x] Core pipeline (detection → tracking → zone → violation → storage)
- [x] ByteTrack integration
- [x] Manual ROI selection (click-to-define polygon)
- [x] Severity scoring + violation type classification
- [x] Spatial deduplication + tuned ByteTrack (fixes same-vehicle multi-count)
- [x] Two-stage plate recognition (detect → OCR → TR validation)
- [x] Gradio web demo (fixed camera, plate toggle)
- [x] YOLOv8 fine-tuning notebooks (v1 `best.pt`, v3 `best_v3.pt`, v4 `best_v4.pt`)
- [x] Plate detector training notebook (`06_train_plate_detector.ipynb`)
- [x] DB schema with idempotent migrations
- [x] Ground-truth annotation (`data/ground_truth/*.json`)
- [x] Detection mAP evaluation + v3-vs-v4 comparison (Ch7, `compare_v3_v4_metrics.py`)
- [x] Pipeline-level P / R / F1 field test (Ch8: P=0.889, R=0.800, F1=0.842)
- [x] Thesis write-up (chapters under `docs/thesis/`)
- [ ] Roboflow dataset export documented in `configs/dataset_info.yaml` (TODOs remain — see `docs/PROJECT_FINAL_CHECKLIST.md`)
- [ ] Plate OCR accuracy evaluation (separate metric from violation P/R)
- [~] Unit test suite (`tests/` — currently TR plate validation only)

## License

Part of a graduation thesis at Istanbul Arel University, Department of Computer Engineering.
