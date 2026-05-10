# Hatched Area Violation Detection

End-to-end computer vision pipeline that detects vehicles entering hatched road markings using YOLOv8, ByteTrack, polygon-based zone checks, and reads Turkish license plates on confirmed violations.

> **Graduation Thesis** — Sertaç Akalın, Istanbul Arel University, Computer Engineering (2026)
> **Advisor:** Prof. Dr. Haluk Gümüşkaya

---

## What This System Does

```
Traffic Video  →  YOLOv8 (best_v3.pt)  →  ByteTrack  →  Zone Check (Shapely)
                                                              ↓
                          State Machine + Trajectory + Severity Scoring
                                                              ↓
                              Plate Detection + EasyOCR + TR Format Validation
                                                              ↓
                                      Violation Report (SQLite + video + plates)
```

1. **Detects** cars, trucks, buses, motorcycles in every frame (YOLOv8m fine-tune)
2. **Tracks** them across frames with persistent IDs (ByteTrack)
3. **Checks** whether the bottom-center point of each vehicle enters a manually defined hatched area polygon (Shapely)
4. **Confirms** the violation only after the vehicle stays inside for at least N frames — reduces false positives from bounding box jitter and brief edge contact
5. **Classifies** each violation (LANE_CHANGE / CRUISING / EDGE_CONTACT) and assigns a 0–100 severity score
6. **Reads the license plate** of confirmed violators (YOLOv8n plate detector + EasyOCR + Turkish 81-province format validation)
7. **Logs** the event to SQLite with vehicle crop, plate crop, frame snapshot, and timestamp

## Original Contribution

The detection (YOLOv8) and tracking (ByteTrack) components are off-the-shelf. The original work in this thesis is:

1. **Fine-tuning YOLOv8m on a custom Istanbul traffic dataset (`final_v3`, 100 epochs)** — domain adaptation for local vehicle appearances and overpass camera angles. Evaluated against both the COCO baseline (`yolov8s.pt`) and an earlier yolov8s fine-tune (`best.pt`) — see `weights/README.md` and `configs/dataset_info.yaml`.
2. **Temporal filtering with a four-state state machine** (OUTSIDE → ENTERING → INSIDE → VIOLATION) that requires N consecutive frames inside the zone plus a per-track lock + cooldown window — eliminates bounding-box jitter false positives that a simple inside/outside check produces.
3. **Trajectory + severity scoring pipeline** that classifies each confirmed violation as `LANE_CHANGE` / `CRUISING` / `EDGE_CONTACT` and assigns a 0–100 score, enabling threshold-based false-positive filtering downstream.
4. **Two-stage plate recognition** with a fine-tuned YOLOv8n plate detector (`plate.pt`, dataset `TR-PLAKA-1`), best-frame voting from a per-track ring buffer, EasyOCR upscale + unsharp pre-processing, and Turkish 81-province plate format validation (`src/plate/tr_plate.py`).
5. **A hand-annotated Istanbul hatched area video dataset** with ground-truth violation timestamps for quantitative evaluation (`data/ground_truth/`).

## Architecture

```
src/
├── core/            # Config, frame I/O, data models, visualization, heatmap
├── detection/       # YOLOv8 vehicle detector wrapper
├── tracking/        # ByteTrack wrapper (single tracker)
├── zones/           # Polygon zone manager, ROI selector
├── violation/       # State machine, trajectory, severity
├── plate/           # Plate detector + EasyOCR + TR format validation
├── storage/         # SQLite schema, migrations, logger
└── pipeline/        # Main orchestrator + factory

scripts/             # CLI tools (run pipeline, select ROI, evaluate, report)
notebooks/           # Colab notebooks (frame extraction → fine-tuning → evaluation)
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

Level:    0–25 LOW  |  25–50 MEDIUM  |  50–75 HIGH  |  75–100 CRITICAL
Type:     LANE_CHANGE (short diagonal crossing)
          CRUISING    (extended travel inside zone)
          EDGE_CONTACT (shallow penetration, likely false positive)
```

## Tech Stack

| Component          | Technology                              |
|--------------------|-----------------------------------------|
| Vehicle Detection  | YOLOv8m fine-tune `best_v3.pt` (Ultralytics) |
| Plate Detection    | YOLOv8n fine-tune `plate.pt` (Ultralytics) |
| Plate OCR          | EasyOCR (PyTorch backend)               |
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

```
00_frame_extraction.ipynb           → extract frames from raw videos
02_vehicle_detection_baseline.ipynb → pretrained YOLOv8 baseline on your videos
03_vehicle_detection_finetuning.ipynb → fine-tune YOLOv8s on Roboflow dataset
04_tracking_comparison.ipynb        → ByteTrack sanity check
07_violation_detection_eval.ipynb   → end-to-end pipeline on a test video
11_master_pipeline.ipynb            → fine-tune → pipeline → comparison (all-in-one)
```

## Data Layout

| Item              | Location                        |
|-------------------|---------------------------------|
| Raw videos        | `data/videos/raw/`              |
| Test video cuts   | `data/videos/test/`             |
| Annotated frames  | `data/datasets/vehicle_detection/` (Roboflow export) |
| Ground truth      | `data/ground_truth/*.json`      |
| Pretrained weights | `weights/yolov8s.pt`           |
| Fine-tuned weights (legacy) | `weights/best.pt`     |
| Fine-tuned weights (production) | `weights/best_v3.pt` |
| Plate detector weights | `weights/plate.pt`         |
| Zone polygons     | `configs/zones/*.json`          |

## Thesis Scope

This project is deliberately scoped to a **working end-to-end system**. The following items are **in scope** and shipped:

- Manual hatched area definition (click-to-polygon ROI tool)
- YOLOv8m fine-tuning on a custom Istanbul traffic dataset
- ByteTrack vehicle tracking
- Four-state temporal state machine + per-track lock + cooldown
- Severity scoring (0–100) + violation-type classification
- Two-stage Turkish license plate recognition with 81-province format validation
- SQLite persistence + idempotent schema migrations
- Gradio web demo with dynamic zone tracking for moving cameras
- Quantitative evaluation against hand-annotated ground truth

The following items are **explicitly out of scope**:

- Automatic hatched area detection (no segmentation model)
- Homography-based speed estimation
- Alternative trackers other than ByteTrack (BoT-SORT/DeepSORT scaffolding exists in `configs/` but is not evaluated)
- Multi-camera fusion / re-identification across cameras

Some earlier prototype files are retained under [`archive/`](archive/) only as historical reference and are not part of the evaluated system.

## Project Status

- [x] Core pipeline (detection → tracking → zone → violation → storage)
- [x] ByteTrack integration
- [x] Manual ROI selection (click-to-define polygon)
- [x] Severity scoring + violation type classification
- [x] Two-stage plate recognition (detect → OCR → TR validation)
- [x] Gradio web demo (with plate toggle + dynamic zone for moving cameras)
- [x] YOLOv8 fine-tuning notebooks (v1 `best.pt` + v3 `best_v3.pt`)
- [x] Plate detector training notebook (`06_train_plate_detector.ipynb`)
- [x] DB schema with idempotent migrations
- [ ] Roboflow dataset export documented in `configs/dataset_info.yaml` (TODOs remain — see `docs/PROJECT_FINAL_CHECKLIST.md`)
- [ ] Ground-truth annotation (target: 20–30 violations × 2 videos)
- [ ] Precision / Recall / F1 evaluation on the test set
- [ ] Plate OCR accuracy evaluation (separate metric from violation P/R)
- [ ] Unit test suite (`tests/` — see checklist)
- [ ] Thesis write-up (template-conformant, 12 sections)

## License

Part of a graduation thesis at Istanbul Arel University, Department of Computer Engineering.
