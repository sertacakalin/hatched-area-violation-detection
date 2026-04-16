# Model Weights

`*.pt` files are **not tracked by git** — they live in Google Drive and
are pulled into this directory on demand. Only this README is tracked.

## Files Expected Here

| File | Size | Provenance |
|---|---|---|
| `yolov8s.pt` | ~22 MB | COCO-pretrained YOLOv8s, downloaded from Ultralytics on first use. Baseline for the comparison experiment. |
| `best.pt` | ~21 MB | Fine-tuned on Roboflow dataset `istanbul-traffic-vehicles` v2. Primary model used for all pipeline evaluation. |

## `best.pt` Provenance

Metadata extracted directly from the checkpoint:

| Field | Value |
|---|---|
| Base model | `yolov8s.pt` (COCO pretrained) |
| Training data | `istanbul-traffic-vehicles` version 2 (Roboflow) |
| Classes | `bus`, `car`, `motorcycle`, `truck` (4) |
| Image size | 640×640 |
| Batch size | 16 |
| Max epochs | 50 (early-stopping patience 10) |
| Optimizer | Auto → SGD + cosine annealing |
| Initial LR | 0.01 |
| Momentum | 0.937 |
| Weight decay | 5e-4 |
| Device | Google Colab (Tesla T4) |
| Training date | 2026-04-05 |
| Drive run directory | `MyDrive/tez_models/istanbul_finetune4/` |

Full human-readable provenance (plus the final validation metrics you
still need to fill in) lives in
[`configs/dataset_info.yaml`](../configs/dataset_info.yaml).

The Ultralytics run directory on Drive also contains:

```
istanbul_finetune4/
├── weights/
│   ├── best.pt              ← this file
│   └── last.pt
├── results.csv              ← per-epoch metrics (Precision, Recall, mAP50, mAP50-95)
├── results.png              ← loss curves
├── confusion_matrix.png
├── PR_curve.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
├── labels.jpg
└── val_batch*_pred.jpg      ← sample predictions
```

**Copy these figures into `docs/figures/`** for the thesis — they are
Section 7 (Experiments and Evaluation) material.

## How to Pull `best.pt` Locally

### Option A — Manual (recommended, one-shot)

1. Open Google Drive → `MyDrive/tez_models/istanbul_finetune4/weights/best.pt`
2. Download and drop the file into `weights/best.pt`
3. Done.

### Option B — Programmatic (from Colab or a synced Drive)

```bash
# On Colab (Drive mounted):
cp /content/drive/MyDrive/tez_models/istanbul_finetune4/weights/best.pt weights/best.pt

# On a Mac with Google Drive Desktop:
cp ~/Google\ Drive/My\ Drive/tez_models/istanbul_finetune4/weights/best.pt weights/best.pt
```

## How to Pull `yolov8s.pt` (baseline)

Ultralytics auto-downloads it on first use, or run:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
# → downloads to ~/.cache/ultralytics or current directory
```

## Reproducing `best.pt` From Scratch

If you ever lose `best.pt` on Drive, you can retrain it from the same
Roboflow dataset version using
`notebooks/03_vehicle_detection_finetuning.ipynb` on Colab. Make sure:

- `ROBOFLOW_API_KEY` is in Colab Secrets (**never commit it**)
- Drive is mounted
- Runtime is set to **T4 GPU**
- Hyperparameters match those in `configs/dataset_info.yaml → training`

Expect ~1.5 hours of training time on a Tesla T4.
