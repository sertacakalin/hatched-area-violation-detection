# Model Weights

`*.pt` files are **not tracked by git** — they live in Google Drive and
are pulled into this directory on demand. Only this README is tracked.

## Files Expected Here

| File | Size | Provenance |
|---|---|---|
| `yolov8s.pt` | ~22 MB | COCO-pretrained YOLOv8s, downloaded from Ultralytics on first use. Baseline for the comparison experiment. |
| `best.pt` | ~21 MB | v1 fine-tune (yolov8s, 50 epoch, dataset v2). Legacy baseline. |
| `best_v3.pt` | ~50 MB | **v3 fine-tune (yolov8m, 100 epoch, lr=1e-3, dataset `final_v3`). Production model — `configs/config.yaml` defaultu.** Sınıflar: `{0:bus, 1:car, 2:motorcycle, 3:truck}`. mAP50=0.808, mAP50-95=0.653, P=0.765, R=0.787. |
| `plate.pt` | ~6 MB | Türk plaka tespit modeli (yolov8n, 80 epoch, dataset `TR-PLAKA-1`). Tek sınıf: `license_plate`. mAP50=0.982, mAP50-95=0.817. Eğitim: `notebooks/06_train_plate_detector.ipynb`. |

## `best_v3.pt` Provenance (Production)

| Field | Value |
|---|---|
| Base model | `yolov8m.pt` (COCO pretrained) |
| Training data | `final_v3` (Roboflow / Drive) |
| Classes | `bus`, `car`, `motorcycle`, `truck` (4) |
| Image size | 640×640 |
| Batch size | 16 |
| Max epochs | 100 |
| Initial LR | 0.001 (fine-tune, best.pt'den 10× küçük) |
| Final precision | 0.7647 |
| Final recall | 0.7873 |
| Final mAP50 | 0.8075 |
| Final mAP50-95 | 0.6530 |

## `plate.pt` Provenance

| Field | Value |
|---|---|
| Base model | `yolov8n.pt` (COCO pretrained) |
| Training data | `TR-PLAKA-1` (Roboflow) |
| Class | `license_plate` (1) |
| Image size | 640×640 |
| Batch size | 32 |
| Max epochs | 80 |
| Initial LR | 0.001 |
| Final precision | 0.9560 |
| Final recall | 0.9447 |
| Final mAP50 | 0.9815 |
| Final mAP50-95 | 0.8171 |
| Training notebook | `notebooks/06_train_plate_detector.ipynb` |

## `best.pt` Provenance (Legacy)

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
| Final mAP50 | 0.5969 |
| Final mAP50-95 | 0.4496 |

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

## How to Pull `best_v3.pt` Locally (Production)

### Option A — Manual

1. Open Google Drive → `MyDrive/tez_models/final_v3/weights/best.pt`
2. Download and rename to `weights/best_v3.pt`
3. Done.

### Option B — Programmatic

```bash
# Colab (Drive mounted):
cp /content/drive/MyDrive/tez_models/final_v3/weights/best.pt weights/best_v3.pt
# Mac (Drive Desktop):
cp ~/Google\ Drive/My\ Drive/tez_models/final_v3/weights/best.pt weights/best_v3.pt
```

## How to Pull `plate.pt` Locally

```bash
cp ~/Google\ Drive/My\ Drive/tez_models/plate/weights/best.pt weights/plate.pt
```

Veya `notebooks/06_train_plate_detector.ipynb` ile retrain (TR-PLAKA-1 dataset, ~30 dk T4).

## How to Pull `best.pt` Locally (legacy)

```bash
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
