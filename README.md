# Hatched Area Violation Detection

End-to-end computer vision pipeline that detects vehicles entering hatched road markings using YOLOv8, ByteTrack, and polygon-based zone checks.

> **Graduation Thesis** — Sertaç Akalın, Istanbul Arel University, Computer Engineering (2026)
> **Advisor:** Prof. Dr. Haluk Gümüşkaya

---

## What This System Does

```
Traffic Video  →  YOLOv8 Detection  →  ByteTrack Tracking  →  Zone Check (Shapely)
                                                                      ↓
                                    State Machine + Trajectory + Severity Scoring
                                                                      ↓
                                                  Violation Report (SQLite + video)
```

1. **Detects** cars, trucks, buses, motorcycles in every frame (YOLOv8)
2. **Tracks** them across frames with persistent IDs (ByteTrack)
3. **Checks** whether the bottom-center point of each vehicle enters a manually defined hatched area polygon (Shapely)
4. **Confirms** the violation only after the vehicle stays inside for at least N frames — reduces false positives from bounding box jitter and brief edge contact
5. **Classifies** each violation (LANE_CHANGE / CRUISING / EDGE_CONTACT) and assigns a 0–100 severity score
6. **Logs** the event to SQLite with a frame crop and timestamp

## Original Contribution

The detection (YOLOv8) and tracking (ByteTrack) components are off-the-shelf. The original work in this thesis is:

1. **Fine-tuning YOLOv8s on a custom Istanbul traffic dataset** — domain adaptation for local vehicle appearances, evaluated against the pretrained COCO baseline.
2. **Temporal filtering with a four-state state machine** (OUTSIDE → ENTERING → INSIDE → VIOLATION) that requires N consecutive frames inside the zone plus a cooldown window — eliminates bounding-box jitter false positives that a simple inside/outside check produces.
3. **A small hand-annotated Istanbul hatched area video dataset** with ground-truth violation timestamps for quantitative evaluation.

Trajectory analysis and severity scoring (`severity.py`, `trajectory.py`) are included as supporting features.

## Architecture

```
src/
├── core/            # Config, frame I/O, data models, visualization
├── detection/       # YOLOv8 vehicle detector wrapper
├── tracking/        # ByteTrack wrapper (single tracker)
├── zones/           # Polygon zone manager, ROI selector
├── violation/       # State machine, trajectory, severity
├── storage/         # SQLite schema and logger
└── pipeline/        # Main orchestrator

scripts/             # CLI tools (run pipeline, select ROI, evaluate)
notebooks/           # Colab notebooks (frame extraction → fine-tuning → evaluation)
configs/             # YAML config + ByteTrack tuning + zone polygons
app.py               # Gradio web demo
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

| Component          | Technology              |
|--------------------|-------------------------|
| Vehicle Detection  | YOLOv8s (Ultralytics)   |
| Fine-tuning        | Roboflow + Colab (T4)   |
| Tracking           | ByteTrack (supervision) |
| Geometry           | Shapely                 |
| Storage            | SQLite                  |
| Web Demo           | Gradio                  |
| Plotting           | Matplotlib + Seaborn    |

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
| Fine-tuned weights | `weights/best.pt`              |
| Zone polygons     | `configs/zones/*.json`          |

## Thesis Scope

This project is deliberately scoped to a **working end-to-end system** rather than a maximal feature set. The following items are **not part of the project scope**:

- Automatic hatched area detection
- License plate recognition
- Homography-based speed estimation
- Alternative trackers other than ByteTrack
- Dynamic zone tracking for moving cameras
- Streamlit monitoring dashboard

Some earlier prototype files are retained under [`archive/`](archive/) only as historical reference and are not part of the evaluated system.

## Project Status

- [x] Core pipeline (detection → tracking → zone → violation → storage)
- [x] ByteTrack integration
- [x] Manual ROI selection (click-to-define polygon)
- [x] Severity scoring + violation type classification
- [x] Gradio web demo
- [x] YOLOv8 fine-tuning notebook
- [ ] Roboflow dataset export (target: 300–500 frames)
- [ ] Ground-truth annotation (target: 20–30 violations across 2 videos)
- [ ] Precision / Recall / F1 evaluation
- [ ] Thesis write-up (template-conformant, 12 sections)

## License

Part of a graduation thesis at Istanbul Arel University, Department of Computer Engineering.
