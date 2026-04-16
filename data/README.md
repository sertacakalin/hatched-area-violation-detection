# Data Directory

**This project runs fully locally.** There is no live sync with Google
Drive, Roboflow, or any cloud service. Data is imported **once** at
setup time and then the project operates against the local copies.

Git only tracks tiny files: this README, `ground_truth/*.json`, and
`ground_truth/README.md`. Everything else (videos, annotated frame
exports, extracted JPEG dumps) is gitignored and lives on disk only.

## Directory Layout

```
data/
├── README.md                       ✓ tracked
├── videos/
│   ├── raw/                        ✗ gitignored — full-length recordings
│   │   ├── cam1.mov
│   │   ├── cam2.MOV
│   │   ├── cam3.MOV
│   │   ├── cam4.mov
│   │   └── cam5.mov
│   └── test/                       ✗ gitignored — short clips for iteration
│       └── cam{1..5}.mp4           (optional, cut from raw/ with ffmpeg)
├── datasets/
│   └── vehicle_detection/          ✗ gitignored — Roboflow YOLOv8 export
│       ├── data.yaml
│       ├── train/ {images,labels}/
│       └── valid/ {images,labels}/
├── frames/                         ✗ gitignored — extracted JPEGs used
│   ├── cam1/                         for Roboflow annotation
│   ├── cam2/
│   ├── cam3/
│   ├── cam4/
│   └── cam5/
└── ground_truth/
    ├── README.md                   ✓ tracked
    └── cam{1..5}.json              ✓ tracked (small, human-authored)
```

## One-Time Import (first-run setup)

Everything below happens **once**. After this, you work entirely
against the local filesystem — no Drive, no Roboflow runtime calls.

### 1. Raw videos
Source: `Drive'ım/istanbul_trafik_kayit/` (Google Drive).
Drop them into `data/videos/raw/` however you prefer — Finder drag,
`rclone copy`, or `cp` from a mounted Drive Desktop.

### 2. Annotated frames (Roboflow export)
Source: Roboflow project `istanbul-traffic-vehicles` **version 2**.
Export once as YOLOv8 format and unzip into
`data/datasets/vehicle_detection/`. The exact Roboflow workspace and
project slugs live in [`configs/dataset_info.yaml`](../configs/dataset_info.yaml).

### 3. Extracted JPEG frames (optional)
Source: `Drive'ım/tez_frames/cam{1..5}/`. 1332 JPEGs used as the
annotation source. Copy into `data/frames/cam{1..5}/` if you plan to
re-annotate or visualize; not required for inference.

### 4. Ground truth (JSON)
Already in `data/ground_truth/cam{1..5}.json` (or will be, once you
import them from `Drive'ım/tez_sonuclari/ground_truth/`). These are
**version controlled** and the only evaluation input the project
needs once the other three are on disk.

## After Import

The pipeline only touches local paths:

```bash
python scripts/run_pipeline.py \
  --config configs/config.yaml \
  --video data/videos/test/cam1.mp4 \
  --zone  configs/zones/cam1.json

python scripts/evaluate_with_ground_truth.py \
  --video data/videos/test/cam1.mp4 \
  --ground-truth data/ground_truth/cam1.json
```

No network calls, no Drive mounts, no Colab. If the machine is
offline, nothing breaks.
