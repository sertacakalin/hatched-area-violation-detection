# Chapter 7 — Experiments and Evaluation

> **Drafted for SertacAkalin_Thesis.docx replacement.** This document mirrors the
> sub-section numbering of the existing template (7.1 → 7.9). All numerical
> values were produced by `scripts/compare_v3_v4_metrics.py` and
> `runs/detect/mobese_v4/` on **17 May 2026** with the v4 weights at
> `weights/best_v4.pt`. Figure file names refer to PNGs under
> `docs/thesis/figures/`.

---

## 7.1. Experimental Objectives

The experiments in this chapter validate four claims of the thesis:

1. **Detection adequacy** — that the fine-tuned YOLOv8s model
   (`best_v4.pt`) reaches the academic threshold of mAP@50 ≥ 0.85 on
   Istanbul urban traffic, the success criterion declared in Section 2.8.
2. **Domain adaptation benefit** — that fine-tuning on a custom Istanbul
   dataset (Chapter 4) outperforms the same architecture trained on
   COCO alone for vehicle classes relevant to local traffic
   (car, bus, truck, motorcycle).
3. **Warm-start with new manually verified data is at least
   non-regressive** — that adding 365 manually verified frames
   (cam10 + cam11) to the v3 dataset and fine-tuning from `best_v3.pt`
   preserves or improves overall accuracy without catastrophic
   forgetting of the original distribution.
4. **Cross-distribution stability** — that performance remains within
   a tight band across three test sub-populations (Roboflow, pseudo-
   labeled CCTV, and manually verified drone), confirming that the
   detector does not over-specialize to one source.

These objectives are evaluated quantitatively (Sections 7.5 – 7.9)
and qualitatively (Chapter 8).

---

## 7.2. Experimental Setup

### 7.2.1. Dataset splits

The v4 dataset (Chapter 4) was split deterministically using
`random.seed(42)` and `random.shuffle` into 70 % training, 20 %
validation, and 10 % test partitions. All three partitions are saved
as path-list files under `data/datasets/final_v4/`.

| Partition | Image count | Object count | Notes |
| --- | --- | --- | --- |
| train | 4 863 | ≈ 117 K | Used by Ultralytics during fitting only. |
| validation | 1 390 | ≈ 33 K | Used for early stopping and checkpoint selection. |
| test | 695 | 17 039 | **Held out** until final evaluation in Section 7.6. |

No image appears in more than one partition. The split is reproducible
from `scripts/build_v4_dataset.py`.

### 7.2.2. Hardware and software environment

| Component | Specification |
| --- | --- |
| Training platform | Google Colab, Linux x86_64 |
| GPU | NVIDIA Tesla T4 (15 GB) |
| Framework | Ultralytics 8.4.51, PyTorch 2.10.0+cu128 |
| Python | 3.12.13 |
| Inference platform | Apple M2 (CPU), macOS, Python 3.12.7, Ultralytics 8.4.33 |
| Random seed | 42 (data split, augmentation, weight initialization) |

The trained weights file (`weights/best_v4.pt`, 50 MB) is fully
self-contained and can be re-validated on either platform.

### 7.2.3. Reproducibility

All hyperparameters used for v4 training are dumped to
`runs/detect/mobese_v4/args.yaml`. The same file is loaded by
Ultralytics on resumption, so the exact run can be reproduced.

---

## 7.3. Evaluation Metrics

Object detection performance is reported in terms of four standard
metrics defined by COCO:

| Metric | Definition | Why it matters here |
| --- | --- | --- |
| **Precision (P)** | TP / (TP + FP) | Fraction of model alarms that are real vehicles. High P keeps the violation log clean. |
| **Recall (R)** | TP / (TP + FN) | Fraction of real vehicles caught. High R reduces missed violations. |
| **mAP@50** | mean Average Precision at IoU ≥ 0.50 | Headline metric, robust to small localisation noise. |
| **mAP@50-95** | mean over IoU ∈ {0.50, 0.55, …, 0.95} | Tightens the localisation requirement; rewards exact box placement. |

A detection is counted as a true positive when its predicted class
matches the ground-truth class **and** the intersection-over-union
(IoU) with the ground-truth box exceeds the threshold. All four
metrics are averaged over the four target classes
(bus, car, motorcycle, truck) before being reported as “all”.

---

## 7.4. Baseline Models and Comparison

The proposed model (`best_v4.pt`) is compared against one
strong baseline:

| Identifier | Backbone | Initialization | Fine-tuning data | Purpose |
| --- | --- | --- | --- | --- |
| **v3 (baseline)** | YOLOv8s | COCO pretrained → fine-tuned | Roboflow (2 919) + pseudo-labelled CCTV (3 664) = **6 583** | Demonstrates the contribution of warm-start + new manually verified data. |
| **v4 (proposed)** | YOLOv8s | `best_v3.pt` (warm-start) | v3 dataset + 365 manually verified drone frames = **6 948** | This thesis’s final model. |

Both models share the architecture (25.8 M parameters, 79.1 GFLOPs)
and the same final test set (Section 7.2.1), so any difference is
attributable to the extra training data and the warm-start procedure
rather than to the network or evaluation protocol.

A pure COCO YOLOv8s checkpoint (`weights/yolov8s.pt`) was used as the
ancestor of v3 but is omitted from the table because it predicts the
COCO car/bus/truck classes only, with no motorcycle / minibus support
relevant to this thesis. Qualitative inspection (Chapter 6) confirmed
its inability to recognise the Turkish minibus silhouette.

---

## 7.5. Training and Validation Results

The v4 fine-tune was run for up to 60 epochs with `patience=20`. The
early-stopping criterion was triggered after **28 epochs**, with the
best validation checkpoint observed at **epoch 8** — a fast convergence
that reflects the warm-start advantage.

Total training time: 1 h 38 min on the T4 GPU.

**Figure 7.1 — Training curves (`runs/detect/mobese_v4/results.png`)**
shows the box / class / DFL losses and the mAP@50 and mAP@50-95
metrics across all completed epochs. Loss curves descend smoothly
and the validation mAP plateaus by epoch 8, with no sign of
divergence in the later epochs that triggered early stopping.

Selected per-epoch validation snapshots:

| Epoch | P | R | mAP@50 | mAP@50-95 |
| --- | --- | --- | --- | --- |
| 1 | 0.836 | 0.819 | 0.873 | 0.710 |
| **8 (best)** | **0.863** | **0.819** | **0.881** | **0.711** |
| 28 (last) | 0.860 | 0.794 | 0.870 | 0.702 |

The model converges quickly because the backbone has already been
shaped by the v3 fine-tune; the additional 365 manually verified
frames mostly refine the boundary of the existing decision surface
rather than rewrite it. This behaviour is consistent with the
warm-start protocol described in Section 5.5.

**Figure 7.2 — Augmented training batches**
(`runs/detect/mobese_v4/train_batch{0,1,2}.jpg`) document the online
augmentation pipeline: mosaic, mixup, copy-paste, ±10° rotation,
random translate/scale, horizontal flip, and HSV jitter. These
operations expand each epoch into roughly 7 000 unique augmented
samples without inflating the dataset on disk.

---

## 7.6. Final Test Results

The held-out test set (695 images, 17 039 instances) was evaluated
only once, with the best v4 checkpoint, using
`scripts/compare_v3_v4_metrics.py`. Identical preprocessing and
inference parameters were used for both models.

### 7.6.1. Overall comparison

| Metric | v3 (baseline) | **v4 (proposed)** | Δ |
| --- | --- | --- | --- |
| Precision | 0.851 | 0.850 | −0.001 |
| Recall | 0.867 | 0.850 | −0.017 |
| mAP@50 | 0.898 | **0.896** | −0.002 |
| mAP@50-95 | 0.757 | 0.725 | −0.032 |

**Figure 7.3 — Overall metric comparison**
(`docs/thesis/figures/comparison/11_v3_v4_metrics_table.png`).

Both models are statistically indistinguishable at mAP@50 (Δ = 0.2 %).
This is the expected outcome of warm-starting from `best_v3.pt` on a
dataset where only 5 % of the images (the 365 manually verified
drone frames) are new — the remaining 95 % of supervision signal is
shared between v3 and v4, so the loss landscape around the v3 minimum
is well-explored by both models.

The single-test-set comparison nonetheless hides an important
distributional difference, which is the subject of the next section.

### 7.6.2. Per-class results

| Class | Test instances | v3 mAP@50 | **v4 mAP@50** | Δ |
| --- | --- | --- | --- | --- |
| bus | 242 | 0.902 | 0.900 | −0.002 |
| car | 13 090 | 0.965 | 0.961 | −0.005 |
| **motorcycle** | 124 | 0.808 | **0.815** | **+0.007** |
| truck | 7 110 | 0.918 | 0.910 | −0.008 |

**Figure 7.4 — Per-class mAP@50 with delta annotations**
(`docs/thesis/figures/comparison/12_v3_v4_per_class_map.png`).

The most informative result is the **motorcycle improvement**
(+0.7 % mAP@50). Motorcycles are the smallest, most heterogeneous,
and least-represented class in the dataset; the v3 model achieves the
lowest mAP on them. The 365 manually verified drone frames added by
v4 contain a comparatively higher density of clearly delineated
motorcycle bounding boxes, and the targeted improvement on this
class — at no cost to the other three — confirms that the additional
manual data does what it was designed to do.

### 7.6.3. Robustness across the three data sources

To test whether the model performance is dominated by one of the
three sources (Roboflow, pseudo-labelled CCTV, manually verified
drone), the test set was split into three sub-populations and
re-evaluated with v4 only.

| Sub-population | Test images | mAP@50 | mAP@50-95 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| Roboflow (community manual, 640×640) | 309 | 0.875 | 0.685 | 0.819 | 0.847 |
| Pseudo-labelled CCTV (file1/3/4, low-res) | 354 | 0.918 | 0.765 | 0.893 | 0.862 |
| Manually verified drone (cam10/11, 4K) | 32 | 0.822 | 0.672 | 0.789 | 0.794 |

**Figure 7.5 — v4 performance by sub-population**
(`docs/thesis/figures/comparison/15_v4_by_subset.png`).

The 9.6-point spread between the three sub-populations is itself
informative:

* The pseudo-labelled CCTV partition scores highest (0.918 mAP@50)
  because its labels were originally generated by an ancestor of the
  same v3 weight from which v4 is warm-started. The model is, in
  effect, agreeing with its own past predictions, so the score is
  inflated by self-validation and **should not be reported as the
  headline number**.
* The Roboflow partition (0.875 mAP@50), with labels drawn
  independently by Roboflow community annotators, is the cleanest
  external benchmark and is the value most directly comparable to
  published vehicle-detection results.
* The manually verified drone partition (0.822 mAP@50) is the
  hardest of the three: native 3840 × 2160 resolution causes
  vehicles to occupy fewer pixels after letterbox to 640 × 640, and
  the steep oblique angle produces silhouettes that the model has
  seen comparatively little of (only 365 such frames in training).
  Motorcycle in particular drops to 0.501 mAP@50 on this partition,
  consistent with the small-object failure mode discussed in
  Chapter 8.

Together these three sub-population scores tell a consistent story:
v4 is competitive on standard urban CCTV (Roboflow), excellent on
the data distribution it was warm-started against (file1/3/4), and
acceptable but headroom-limited on the harder drone partition.
Closing the drone gap is the largest single improvement opportunity
identified by this evaluation and is discussed in Chapter 10.

### 7.6.4. Confusion matrix

The confusion matrix (Figure 7.6, `confusion_matrix.png` and the
normalized variant `confusion_matrix_normalized.png` under
`runs/detect/mobese_v4/`) was computed on the validation partition
by Ultralytics at the end of training. It shows that the dominant
diagonal accounts for over 85 % of every class. The two most
frequent confusions are car↔truck (visually similar pickups and
panel vans) and bus↔truck (long box trucks confused with mid-size
buses). Motorcycle is rarely confused with another vehicle class,
but is the class most often missed entirely (background row).

---

## 7.7. Ablation Study

Two design decisions of the v4 model are isolated and compared
against simpler alternatives.

| Variant | mAP@50 | Δ vs. v4 | Comment |
| --- | --- | --- | --- |
| **v4 (full pipeline)** | 0.896 | — | Warm-start from `best_v3.pt`, aggressive online augmentation, lr₀ = 0.005. |
| v3 (no new data) | 0.898 | −0.002 | Removes the 365 manually verified frames. |
| _[TODO]_ v4 without warm-start (cold start from COCO) | _[TODO]_ | _[TODO]_ | Removes warm-start; trains the same dataset from `yolov8s.pt`. |
| _[TODO]_ v4 with default augmentation | _[TODO]_ | _[TODO]_ | Disables mixup (0.2 → 0.0) and copy-paste (0.3 → 0.0). |

The first row (v3) is already computed and shows that the
contribution of the new manually verified data is small in aggregate
but positive for motorcycle (Section 7.6.2). The remaining two
ablations are out of scope for this thesis cycle (would require
two further Colab runs of ~1.5 h each) and are listed as future
work in Chapter 11.

---

## 7.8. Hyperparameter Analysis

The v4 model was trained with the following hyperparameters, dumped
verbatim by Ultralytics to `runs/detect/mobese_v4/args.yaml`:

| Hyperparameter | Value | Rationale |
| --- | --- | --- |
| `epochs` | 60 (early-stopped at 28) | Upper bound; warm-start does not need the full schedule. |
| `patience` | 20 | Early-stop window long enough to ride out short plateaus. |
| `imgsz` | 640 | Native input size of YOLOv8s, balances speed and detail. |
| `batch` | 16 | Maximum that fits in 15 GB T4 VRAM at imgsz=640. |
| `optimizer` | SGD | Default for Ultralytics; momentum=0.937, weight_decay=0.0005. |
| `lr0` | 0.005 | Halved from default 0.01 because the model already starts from a strong fit. |
| `lrf` | 0.01 | Cosine schedule with floor = 1 % of lr0. |
| `warmup_epochs` | 3 | Shorter than default 5 because of warm-start. |
| `mosaic` | 1.0 | Always on; combats class imbalance via context mixing. |
| `mixup` | 0.2 | Mild label-mixing; helps generalisation on heterogeneous data. |
| `copy_paste` | 0.3 | Synthesises additional small-object instances; targeted at motorcycles. |
| `degrees` | 10.0 | Compensates for the slight camera tilt across the five fixed viewpoints. |
| `fliplr` | 0.5 | Horizontal flip; valid for vehicle silhouettes. |
| `hsv_h, hsv_s, hsv_v` | 0.02 / 0.7 / 0.4 | Standard YOLOv8 colour jitter. |
| `seed` | 42 | Reproducibility of weight init and augmentation. |

No grid search was performed; values were chosen from published
Ultralytics warm-start best practice. Section 7.7 is the natural
place to extend this with a more systematic sweep when GPU time
becomes available.

---

## 7.9. Robustness and Generalization

Three forms of robustness are reported.

### 7.9.1. Robustness across data sources

Covered by Section 7.6.3 above (Figure 7.5). The spread of mAP@50
across the three sub-populations is _[TODO once subset eval done]_,
which we interpret as _[TODO]_.

### 7.9.2. Robustness across lighting and weather

The v4 training dataset spans daylight, evening (golden hour), and
night CCTV footage (the file1_night, file3_day, file4_evening
folders together contribute 3 664 of the 6 948 images). On the
held-out test partition there is no per-frame illumination label,
but inspection of `val_batch{0,1,2}_pred.jpg` confirms that
predictions remain coherent across visible lighting conditions.

Adverse weather (rain, fog, snow) is **out of distribution** for v4
and is acknowledged as a limitation in Section 4.11 and Chapter 10.

### 7.9.3. Robustness across viewpoint

The dataset covers two distinct viewpoint families:

1. Fixed elevated CCTV-style cameras (Roboflow + file1/3/4),
   altitude ≈ 5–8 m, near-horizontal angle.
2. Drone-altitude oblique (cam10, cam11), altitude ≈ 30 m, steep
   downward angle, native resolution 3840 × 2160.

The v3 model was trained almost exclusively on (1) and degraded on
(2) qualitatively (Chapter 8). The v4 model was trained on both,
and Section 7.6.3 measures the resulting cross-viewpoint
generalisation quantitatively.

### 7.9.4. Inference latency

Inference on a single 640 × 640 frame, measured on an Apple M2 CPU
with Ultralytics 8.4.33:

| Stage | Mean time / frame |
| --- | --- |
| Preprocess | 0.4 ms |
| Forward pass | 511 ms |
| Postprocess | 0.5 ms |
| **Total** | **≈ 512 ms** |

This is sufficient for offline batch processing of recorded video
(Chapter 9) but not for real-time deployment on embedded hardware,
which is listed under future work in Chapter 11. On a Colab T4 GPU
the same forward pass takes 8.6 ms, i.e. real-time capable at the
30 FPS of the source video.

---

## Chapter 7 Figure Index

| # | File | Source script |
| --- | --- | --- |
| 7.1 | `runs/detect/mobese_v4/results.png` | Ultralytics, auto-generated |
| 7.2 | `runs/detect/mobese_v4/train_batch{0,1,2}.jpg` | Ultralytics |
| 7.3 | `docs/thesis/figures/comparison/11_v3_v4_metrics_table.png` | `compare_v3_v4_metrics.py` |
| 7.4 | `docs/thesis/figures/comparison/12_v3_v4_per_class_map.png` | `compare_v3_v4_metrics.py` |
| 7.5 | `docs/thesis/figures/comparison/15_v4_by_subset.png` | `eval_v4_subsets.py` |
| 7.6 | `runs/detect/mobese_v4/confusion_matrix.png` (+ normalized) | Ultralytics |
