# 4. Dataset And Preprocessing

This chapter describes the data used in the project. Three datasets are involved:

1. **IHTD (`final_v3`)** — the main vehicle dataset, built and labeled by the author. It combines two sources: (i) original videos recorded by the author at two distinct overpass locations in Istanbul (Mall of Istanbul and Güneşli), and (ii) still frames captured from İBB MOBESE public traffic cameras at several other points around the city. Used to fine-tune the YOLOv8m vehicle detector (`weights/best_v3.pt`).
2. **`TR-PLAKA-1`** — a public Turkish license-plate dataset, used to fine-tune the YOLOv8n plate detector (`weights/plate.pt`).
3. **`ground_truth`** — a small set of hand-annotated JSON files used to evaluate the full pipeline end-to-end.

All numbers in this chapter come from `configs/dataset_info.yaml` and from the path-list files under `data/datasets/final_v3/`.

---

## 4.1. Dataset Description

**Vehicle dataset — IHTD (`final_v3`).** The Istanbul Hatched-Area Traffic Dataset (IHTD) is a custom-built dataset of Istanbul traffic images. The production export is called `final_v3` and contains **6 631 labeled frames** across **four vehicle classes**: `bus`, `car`, `motorcycle`, `truck`. Each frame is a JPEG image with a sibling YOLO-format `.txt` label file listing bounding boxes and class IDs. The dataset was assembled in two phases:

- **Phase 1 — Author's own recordings, manually labeled:** 2 919 frames (44 % of the dataset) extracted from videos that the author recorded at two distinct Istanbul overpass locations. Three camera viewpoints were captured at **Mall of Istanbul** (`cam1`, `cam2`, `cam3`) and two at **Güneşli** (`cam4`, `cam5`). Every bounding box in this phase was drawn by hand on Roboflow.
- **Phase 2 — İBB MOBESE frames, semi-supervised:** 3 715 frames (56 % of the dataset) captured as stills from İstanbul Büyükşehir Belediyesi MOBESE public traffic cameras at 4–10 distinct points around the city. These were pre-labeled by an earlier YOLOv8s detector and then manually reviewed and corrected before being merged into the production export.

The detailed per-source breakdown is given in Table 4.2 (Section 4.10).

**Plate dataset — `TR-PLAKA-1`.** A public Roboflow dataset of Turkish license plates. Single class (`license_plate`). `[TODO: paste Roboflow URL and total / train / valid / test counts]`.

**Ground-truth evaluation set.** A small set of author-produced JSON files under `data/ground_truth/`, one per evaluated video. Each file lists violation events with start and end frame numbers, vehicle class, and violation type (`LANE_CHANGE`, `CRUISING`, `EDGE_CONTACT`). This set is used only for measuring Precision, Recall, and F1 of the full pipeline — it is never seen by the trained models.

---

## 4.2. Data Collection

The IHTD vehicle dataset was assembled and annotated entirely by the author, so that it would reflect local Istanbul traffic conditions.

**Source of the video footage.** The dataset combines two independent sources of footage:

1. **Author's own video recordings (Phase 1, `cam1`–`cam5`).** Continuous MP4 videos recorded by the author at two distinct overpass locations in Istanbul: **Mall of Istanbul** (yielding viewpoints `cam1`, `cam2`, `cam3`) and **Güneşli** (yielding viewpoints `cam4`, `cam5`). All five viewpoints were filmed during daytime in fair-weather conditions.

2. **İBB MOBESE camera stills (Phase 2, `file1_night` / `file3_day` / `file4_evening`).** Still frames captured from the publicly accessible İstanbul Büyükşehir Belediyesi (İBB) MOBESE traffic-camera network at **4–10 distinct points** around the city. These frames cover daytime, evening, and night-time illumination, which is why they extend the lighting range beyond Phase 1. The "mobese" provenance is preserved inside the project in the training notebook filename `05_train_mobese_v3.ipynb`.

Rain, fog, and snow conditions remain out of scope in both phases.

**Annotation.** All bounding boxes were drawn on the Roboflow web platform. Each visible vehicle received one bounding box and one class label (`bus`, `car`, `motorcycle`, or `truck`).

**Quality control.** The dataset has a single annotator (the author), so an inter-annotator agreement metric (Cohen's κ) is not applicable. Instead, a second visual inspection pass was performed in Roboflow to correct mislabeled classes (for example, a bus annotated as a truck) and to remove boxes drawn around heavily occluded or motion-blurred vehicles. One auto-labeled batch (`_REMOVED_file2_dense`) was discarded entirely because its preliminary labels were too noisy.

`[TODO: number of source videos, recording locations, recording date range, and total raw footage duration]`.

---

## 4.3. Data Structure and Features

The dataset uses the standard YOLO format. Each image has a sibling `.txt` file in which every line describes one bounding box:

```
<class_id> <x_center> <y_center> <width> <height>
```

with `class_id ∈ {0=bus, 1=car, 2=motorcycle, 3=truck}` and all coordinates normalized to `[0, 1]` of the image size.

Images are RGB JPEGs, 8 bits per channel. The native recording resolution is `[TODO: confirm — typically 1920×1080 or 3840×2160]`; every frame is resized to 640 × 640 with letterbox padding before training.

The ground-truth JSON schema is documented in `data/ground_truth/README.md` and is not repeated here.

---

## 4.4. Data Splitting

The vehicle dataset is split into three parts:

| Split | Images | Share |
|---|---:|---:|
| Training | 4 642 | 70 % |
| Validation | 1 326 | 20 % |
| Test | 663 | 10 % |
| **Total** | **6 631** | **100 %** |

The split was produced randomly by Roboflow with seed `42`, at the image level. Because frames were sampled from videos at fixed time intervals (Section 4.5), consecutive video frames do not appear back-to-back inside the labeling pool, so any residual leakage between train and test is small.

The ground-truth set is **not** split — it is held out entirely for end-to-end evaluation.

---

## 4.5. Data Preprocessing

Preprocessing steps applied to every image, handled internally by the Ultralytics dataloader:

1. **Frame extraction** from the source videos using OpenCV at fixed intervals (to avoid near-duplicate consecutive frames).
2. **Resize** to 640 × 640 with letterbox padding, which preserves the original aspect ratio by adding gray bars instead of stretching the image.
3. **Pixel normalization** to the `[0, 1]` range.
4. **Label normalization** to the `[0, 1]` range, as required by the YOLO format.

No additional color-space conversion or cropping is applied. The same preprocessing path is used at inference time.

---

## 4.6. Data Cleaning

Noisy data was filtered to keep the training signal clean:

- **Blurred / occluded frames** were removed from the labeling pool during the secondary review pass in Roboflow.
- **Mislabeled boxes** (for example, a bus annotated as a truck) were corrected in the same review pass.
- **One whole batch (`_REMOVED_file2_dense`)** was discarded because the preliminary auto-labels were unreliable in dense traffic.
- **Near-duplicate frames** are filtered by Roboflow's "Generate" step during export.

For the ground-truth set, the only cleaning step is the deletion of the placeholder `EXAMPLE_DELETE_ME` entry before evaluation; the evaluator refuses to run if any placeholder remains.

---

## 4.7. Data Augmentation

Augmentation was applied to the training set only, to make the model more robust to camera and lighting differences.

**Roboflow export-time augmentations** (baked into the exported images):

- Horizontal flipping
- Brightness and contrast jitter
- Gaussian blur
- Pixel (Gaussian) noise

**Training-time augmentations** (applied on the fly by YOLOv8):

- Mosaic
- HSV jitter (hue, saturation, value)
- Horizontal flip

Vertical flips and rotations were intentionally **disabled**, because the overpass camera angle has a fixed up-direction and the hatched zone the pipeline must detect is direction-sensitive — rotating the training image would teach the model to recognize traffic that no real camera ever sees.

---

## 4.8. Class Imbalance Handling

The class distribution in IHTD reflects real Istanbul traffic, which is dominated by passenger cars:

| Class | Approximate share |
|---|---:|
| `car` | ≈ 75 % |
| `truck` | ≈ 10 % |
| `bus` | ≈ 10 % |
| `motorcycle` | ≈ 5 % |

No explicit oversampling or class weighting was used. Instead, the imbalance is mitigated implicitly by two mechanisms already built into the training pipeline:

1. **Distribution Focal Loss (DFL)**, which is part of the YOLOv8 loss and penalizes minority-class errors more strongly than plain regression losses.
2. **Mosaic augmentation**, which combines four images into one composite during training and so increases the probability that a minority-class object appears in every batch.

Class imbalance does not apply to the plate dataset, which has a single class.

---

## 4.9. Feature Engineering

No handcrafted features (HOG, SIFT, color histograms, etc.) were extracted. YOLOv8 learns its own features end-to-end from raw pixels through its convolutional backbone.

The only feature derived from raw outputs is the **bottom-center point** of each predicted bounding box, which the pipeline uses as the vehicle's "footprint" when checking whether it has entered the hatched polygon. This is a geometric projection computed at inference time, not a learned feature.

---

## 4.10. Data Visualization & Exploration

Three aspects of the dataset were explored: how the production export is composed, how the manual frames are distributed across the five cameras, and how the four classes are distributed overall.

### Composition of `final_v3` (per-source breakdown)

The 6 631 frames in `final_v3` come from four distinct sources. Table 4.2 reports the exact per-split counts.

**Table 4.2.** Per-source breakdown of `final_v3`.

| Source | Origin | Train | Validation | Test | **Total** | Share |
|---|---|---:|---:|---:|---:|---:|
| cam1–cam5 | Own recordings (Mall of Istanbul + Güneşli), manual labels | 2 034 | 579 | 306 | **2 919** | 44.0 % |
| `file1_night` | İBB MOBESE, night | 1 219 | 368 | 164 | **1 751** | 26.4 % |
| `file3_day` | İBB MOBESE, day | 808 | 225 | 118 | **1 151** | 17.4 % |
| `file4_evening` | İBB MOBESE, evening | 582 | 155 | 76 | **813** | 12.3 % |
| **Total** | — | **4 642** | **1 326** | **663** | **6 631** | **100 %** |

Two observations matter for later chapters:

- More than half (56 %) of the production training data was auto-labeled and then reviewed by hand, rather than labeled from scratch. Any residual label noise in those batches propagates into the final model.
- The `file1_night` batch (1 751 frames) adds genuine night-time coverage that the original cam1–cam5 collection did not have.

### Per-viewpoint distribution (cam1–cam5)

Inside the manually labeled Phase 1 subset, frames are grouped by recording location: `cam1`, `cam2`, `cam3` were filmed at **Mall of Istanbul** and `cam4`, `cam5` at **Güneşli**. Counts are not evenly distributed across the five viewpoints.

**Table 4.3.** Per-camera frame counts inside the author's own Phase 1 subset.

| Camera | Location | Frames | Share of Phase 1 |
|---|---|---:|---:|
| `cam1` | Mall of Istanbul | 865 | 29.6 % |
| `cam2` | Mall of Istanbul | 828 | 28.4 % |
| `cam3` | Mall of Istanbul | 44 | **1.5 %** |
| `cam4` | Güneşli | 653 | 22.4 % |
| `cam5` | Güneşli | 529 | 18.1 % |
| **Total** | — | **2 919** | 100 % |

The notable point is that **`cam3` is severely under-represented**: only 44 frames, compared with 529–865 for the other four viewpoints. This is an artifact of the recording session at Mall of Istanbul — the third viewpoint was added late and only one short session was captured before the dataset was frozen. Conclusions about model behavior specifically on the `cam3` viewpoint should therefore be treated as exploratory.

### Sample frames and class distribution

A preview grid of one representative frame from each of the five cameras is shown in **Figure 4.1**. The class-frequency histogram automatically produced by Ultralytics during training is shown in **Figure 4.2** — it confirms the long-tail distribution described in Section 4.8. Training and validation loss curves for the production model are shown in **Figure 4.3**.

**Patterns observed.** `cam1` and `cam2` give wider field-of-view coverage and are well suited for multi-lane analysis; `cam4` and `cam5` are closer and more top-down, which makes them easier to define precise polygons over. This viewpoint diversity directly motivated the brightness/contrast augmentation chosen in Section 4.7.

---

## 4.11. Data Limitations

The dataset has several limitations that bound how far the trained model can be trusted in Chapter 7:

1. **Modest size.** 6 631 frames is much smaller than mainstream benchmarks (COCO, BDD100K). The original target was 10 000+ instances; the gap has been partly closed but remains.
2. **Weather bias.** Day, evening, and night illumination are all represented, but rain, fog, and snow are not. The model is not expected to generalize to adverse weather.
3. **Geographic bias.** All footage is from Istanbul. Non-Istanbul vehicle silhouettes may be misclassified.
4. **Camera-angle coverage.** All footage is from elevated pedestrian overpasses, five fixed viewpoints. Ground-level CCTV and significantly different overpass heights are out of distribution.
5. **Severely under-represented `cam3` viewpoint.** Only 44 frames, versus 529–865 for the other cameras (Table 4.3). Any per-camera metric on `cam3` should be reported as exploratory.
6. **Class imbalance.** Despite DFL and mosaic mitigation, minority-class recall (especially motorcycles) is expected to be lower than majority-class recall.
7. **Single annotator.** All bounding boxes were drawn by one person. Small systematic biases cannot be ruled out.
8. **Semi-supervised label noise.** 56 % of the production dataset was auto-labeled and then reviewed, rather than labeled from scratch. Residual bias from the bootstrap detector may persist.
9. **Small ground-truth evaluation set.** `[TODO: state the number of annotated videos and total violation events]`.

These limitations are revisited in Chapter 10.

---

## 4.12. Ethical and Legal Considerations

The dataset and the runtime pipeline are designed to fit Turkish data-protection rules (KVKK).

**Privacy and PII.** Faces are never detected, never recognized, and never used at any stage. License plates *are* read on confirmed violators by the secondary plate model — but only on confirmed violations, never on every vehicle in the frame. Plate text and the matching province code are stored locally in a SQLite database. The `TR-PLAKA-1` dataset contains only license-plate images, and the IHTD dataset contains no readable plate text. The system has no network output channel by default — nothing leaves the local machine.

**Lawfulness of source footage.** Both sources of footage are public in nature: (i) the author's own recordings (Phase 1) were captured from publicly accessible overpass locations (Mall of Istanbul and Güneşli) for academic research, with no private property monitored; (ii) the Phase 2 still frames originate from the İstanbul Büyükşehir Belediyesi MOBESE network, which is a publicly accessible municipal traffic-camera feed. No subscription, no credential, and no private API was used to obtain them. `[TODO: optionally cite the relevant KVKK clause and confirm that İBB MOBESE feeds may be used for non-commercial academic research; check İBB Açık Veri terms of use]`.

**Licenses.** IHTD is author-produced; `[TODO: declare a license]`. `TR-PLAKA-1` follows its original Roboflow license. The ground-truth files are distributed under the same license as the source code.

**Use of recognized plates.** Plate recognition in this project is a research demonstration, not a production enforcement system. Any operational use would require additional safeguards (retention policy, access control, human-in-the-loop review before any sanction), discussed in Chapter 10.

---

## 4.13. Reproducibility

The source code, configuration, ground-truth files, and this thesis chapter are hosted on GitHub at [github.com/sertacakalin/hatched-area-violation-detection](https://github.com/sertacakalin/hatched-area-violation-detection).

| Artifact | Location |
|---|---|
| Source code repository | github.com/sertacakalin/hatched-area-violation-detection |
| IHTD (`final_v3`) Roboflow export | `[TODO: paste URL]` (private workspace) |
| Path lists used by Ultralytics | `data/datasets/final_v3/{train,valid,test}.txt` |
| `TR-PLAKA-1` Roboflow export | `[TODO: paste URL]` |
| Ground-truth annotations | `data/ground_truth/*.json` (tracked in git) |
| Dataset provenance & hyperparameters | `configs/dataset_info.yaml` |
| Random / split seed | 42 |

To reproduce the vehicle model, pull `final_v3` from Roboflow, open the training notebook in Google Colab (T4 GPU), and run with `epochs_max=100`, `batch_size=16`, `image_size=640`, `lr0=0.001`, base model `yolov8m.pt`. The resulting weights should match `weights/best_v3.pt` to within stochastic-training tolerance.

---

## Table 4.1. Dataset and preprocessing summary

| Property | Value |
|---|---|
| Dataset name | Istanbul Hatched-Area Traffic Dataset (IHTD) |
| Sources | (1) Author's own video recordings at Mall of Istanbul (`cam1`–`cam3`) and Güneşli (`cam4`–`cam5`); (2) İBB MOBESE traffic-camera stills from 4–10 distinct city locations |
| Phase 1 (own recordings, manual labels) | 2 919 labeled frames |
| Phase 2 (İBB MOBESE, auto-label + manual review) | 3 715 labeled frames |
| **Final size (`final_v3`)** | **6 631 labeled frames** |
| Classes | 4 — `bus`, `car`, `motorcycle`, `truck` |
| Class distribution | car ≈ 75 %, truck ≈ 10 %, bus ≈ 10 %, motorcycle ≈ 5 % |
| Train / Val / Test split | 70 / 20 / 10 % (4 642 / 1 326 / 663) |
| Data type | JPEG images + YOLO-format `.txt` labels |
| Preprocessing | OpenCV frame extraction, 640 × 640 letterbox resize, `[0, 1]` normalization |
| Roboflow augmentations | Horizontal flip, brightness/contrast jitter, Gaussian blur, pixel noise |
| YOLOv8 augmentations | Mosaic, HSV jitter, horizontal flip |
| Class imbalance handling | Distribution Focal Loss (DFL) + mosaic |
| Annotation tool | Roboflow (single annotator + secondary review pass) |
| Production model | YOLOv8m (`weights/best_v3.pt`) |
| Plate dataset | `TR-PLAKA-1`, 1 class, YOLOv8n → `weights/plate.pt` |
| Evaluation set | `data/ground_truth/*.json` (3 violation types) |
| Random seed | 42 |

---

## Figure 4.1. Sample frames from the five fixed-camera viewpoints

`[TODO: insert a 5-panel grid with one representative frame from each of cam1, cam2, cam3, cam4, cam5. Suggested source: hand-pick from data/datasets/frames_for_labeling/.]`

**Caption.** Sample frames from the five fixed-camera viewpoints (cam1–cam5) used for the Istanbul Hatched-Area Traffic Dataset. `cam1` and `cam2` give wider multi-lane coverage; `cam4` and `cam5` give a closer, more top-down perspective.

## Figure 4.2. Class distribution and bounding-box statistics

`[TODO: include docs/figures/training/labels.jpg — the per-class histogram and bounding-box statistics generated automatically by Ultralytics during training.]`

## Figure 4.3. Training and validation loss curves

`[TODO: include the production v3 fine-tune's results.png. The artifacts currently in docs/figures/training/ are from the v1 run; pull the v3 run from Drive before final submission.]`

**Caption.** Per-epoch training and validation loss curves for the production `weights/best_v3.pt` fine-tune.
