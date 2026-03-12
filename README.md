# Hatched Area Violation Detection

Automated detection of vehicles illegally crossing hatched road markings using YOLOv8, ByteTrack, and trajectory-based severity scoring.

> **Graduation Thesis** — Sertaç Akalın, Istanbul Arel University (2026)

## Overview

This system processes traffic camera footage to detect vehicles that illegally enter hatched (no-go) zones on roads. It goes beyond simple zone-presence checks by analyzing vehicle trajectories and computing a multi-dimensional severity score for each violation.

## Pipeline

```
Video Input
    │
    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  YOLOv8      │────▶│  ByteTrack   │────▶│  Zone Check  │
│  Detection   │     │  Tracking    │     │  (Shapely)   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌────────────────────────────┘
                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  State       │────▶│  Trajectory  │────▶│  Severity    │
│  Machine     │     │  Analysis    │     │  Scoring     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌────────────────────────────┘
                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Plate       │────▶│  SQLite      │────▶│  Output      │
│  OCR         │     │  Storage     │     │  Video       │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Key Features

- **Vehicle Detection** — YOLOv8s with optional fine-tuning on Istanbul traffic data
- **Multi-Object Tracking** — ByteTrack for persistent vehicle identity across frames
- **Polygon-Based Zone Control** — Manual ROI definition with Shapely point-in-polygon checks
- **State Machine** — 4-state temporal filtering (OUTSIDE → ENTERING → INSIDE → VIOLATION)
- **Trajectory Analysis** — Entry/exit points, crossing angle, penetration depth
- **Severity Scoring** — Multi-dimensional score (0-100) based on duration, distance, depth, angle
- **Violation Classification** — Categorizes violations as: lane-change, through-travel, edge-contact
- **License Plate Recognition** — YOLOv8n detection + PaddleOCR with Turkish plate validation

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/download_weights.py --all
```

## Usage

```bash
# 1. Define hatched area polygon (interactive GUI)
python scripts/select_roi.py \
  --video data/videos/test/test_01.mp4 \
  --output configs/zones/test_01.json

# 2. Run the detection pipeline
python scripts/run_pipeline.py \
  --config configs/config.yaml \
  --video data/videos/test/test_01.mp4 \
  --show

# 3. Run evaluation against ground truth
python scripts/run_evaluation.py \
  --videos data/videos/test/*.mp4
```

## Project Structure

```
├── src/
│   ├── core/           # Config, data models, frame provider, visualizer
│   ├── detection/      # YOLOv8 vehicle detector
│   ├── tracking/       # ByteTrack, BoT-SORT, DeepSORT wrappers
│   ├── zones/          # Polygon zone manager, ROI selector
│   ├── violation/      # State machine, violation detector, trajectory, severity
│   ├── alpr/           # Plate detection, OCR, preprocessing, validation
│   ├── storage/        # SQLite database, violation logger
│   ├── dashboard/      # Streamlit web UI
│   └── pipeline/       # Main orchestrator
├── configs/            # YAML configs + zone polygon JSONs
├── scripts/            # CLI tools (run, evaluate, select ROI, download weights)
├── notebooks/          # Experiment notebooks (Colab)
├── data/               # Videos, datasets, ground truth (not tracked)
├── weights/            # Model weights (not tracked)
└── results/            # Experiment outputs (not tracked)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection | YOLOv8 (Ultralytics) |
| Tracking | ByteTrack |
| Geometry | Shapely |
| Plate OCR | PaddleOCR |
| Storage | SQLite |
| Dashboard | Streamlit |
| Training | Google Colab |

## License

This project is part of a graduation thesis at Istanbul Arel University.
