# Tez Figür Kılavuzu — GENİŞLETİLMİŞ (~71 figür)

Maksimum görsel zenginlik için tüm bölümler. Her satır: figür no | yol | caption.

**Proje kök:**
`/Users/sertacakalin/Desktop/Projects/bitirme/hatched-area-violation-detection/`

**Kısaltmalar:**
- `A/` = `runs/detect/mobese_v4/`
- `B/` = `docs/thesis/figures/comparison/`
- `C/` = `docs/thesis/figures/`
- `D/` = `data/datasets/havd_v4_dataset/` (örnek frame'ler için)
- `E/` = `/Users/sertacakalin/Desktop/Projects/bitirme/adsız klasör/` (eski ama kullanışlı)
- `(✏️)` = Sen draw.io / SmartArt ile çizeceksin
- `(📸)` = Sen screenshot alacaksın
- `(🆕)` = Ben üreteceğim (yeni script)

---

## Chapter 1 — Introduction
*Figür yok.*

---

## Chapter 2 — Problem Definition and Objectives
*Figür yok (sadece liste tablolar).*

---

## Chapter 3 — Related Work (+3 figür)

| # | Yol | Caption |
|---|-----|---------|
| 3.1 | (✏️) draw.io | *YOLO architecture family overview (v5 / v8 / NAS) — single-stage detection paradigm.* |
| 3.2 | (✏️) draw.io | *Tracker comparison schematic: SORT (Kalman + Hungarian) vs DeepSORT (+ appearance) vs ByteTrack (low-conf recovery).* |
| 3.3 | (✏️) draw.io | *Detection paradigm: two-stage (Faster R-CNN) vs single-stage (YOLO).* |

**Ch3 toplam: 3 figür** (ya draw.io'da çiz ya da kaynak paperdan alıntı yap)

---

## Chapter 4 — Dataset and Preprocessing (+16 figür)

### 4.1. Dataset Description
| # | Yol | Caption |
|---|-----|---------|
| 4.1 | `E/` `figure_4_1_cam_grid.png` | *Five fixed-camera viewpoints (cam1–cam5) covered by the manually annotated Phase 1 portion of the IHTD dataset.* |
| 4.2 | `E/` `figure_4_3_mobese_grid.png` | *MOBESE CCTV samples spanning night, day, and evening illumination conditions (Phase 2, semi-supervised).* |
| 4.3 | `C/` `08_sample_grid.png` | *Representative frames from the six dataset sources.* |

### 4.3. Data Structure and Features
| # | Yol | Caption |
|---|-----|---------|
| 4.4 | `C/` `09_bbox_overlay_examples.png` | *Example frames with overlaid YOLO bounding boxes (one per source).* |
| 4.5 | (🆕) class showcase grid | *Per-class examples (bus, car, motorcycle, truck) × 4 instances each.* |

### 4.4. Data Splitting
| # | Yol | Caption |
|---|-----|---------|
| 4.6 | `C/` `01_source_distribution.png` | *Per-source image distribution across the v4 dataset.* |

### 4.5. Data Preprocessing
| # | Yol | Caption |
|---|-----|---------|
| 4.7 | `A/` `train_batch0.jpg` | *Augmented training batch sample (mosaic + mixup + flip + HSV).* |
| 4.8 | (🆕) letterbox demo | *Letterbox preprocessing: 4K source frame (3840 × 2160) → 640 × 640 model input.* |

### 4.7. Data Augmentation
| # | Yol | Caption |
|---|-----|---------|
| 4.9 | `A/` `train_batch1.jpg` | *Second augmented training batch sample.* |
| 4.10 | `A/` `train_batch2.jpg` | *Third augmented training batch sample.* |

### 4.8. Class Imbalance Handling
| # | Yol | Caption |
|---|-----|---------|
| 4.11 | `C/` `02_class_distribution.png` | *Object-count distribution across the four target classes.* |
| 4.12 | `C/` `03_source_class_heatmap.png` | *Source × class object-count heatmap.* |

### 4.10. Data Visualization & Exploration
| # | Yol | Caption |
|---|-----|---------|
| 4.13 | `C/` `04_bbox_size_distribution.png` | *Bounding-box size distribution (COCO small / medium / large).* |
| 4.14 | `C/` `05_bbox_position_heatmap.png` | *Bounding-box centre-point density heatmap.* |
| 4.15 | `C/` `06_objects_per_image.png` | *Per-image object-count histogram.* |
| 4.16 | `C/` `07_manual_vs_pseudo.png` | *Manual vs. auto-labeled annotation breakdown.* |

**Ch4 toplam: 16 figür**

---

## Chapter 5 — Methodology (+6 figür — ÇİZ)

| # | Yol | Caption |
|---|-----|---------|
| 5.1 | (✏️) draw.io | *End-to-end system architecture: FrameProvider → YOLOv8s + ByteTrack → ZoneManager → State Machine → ViolationLogger / Visualizer.* |
| 5.2 | (✏️) draw.io | *Vehicle state machine: OUTSIDE → ENTERING → INSIDE → VIOLATION with cooldown loop back to OUTSIDE.* |
| 5.3 | (✏️) draw.io | *Polygon overlap geometry — Shapely `contains` and `intersection` operations.* |
| 5.4 | (✏️) draw.io | *Trajectory analyser: entry point, penetration depth, exit angle, dwell time.* |
| 5.5 | (✏️) draw.io | *Severity scorer formula breakdown — weighted sum of trajectory features.* |
| 5.6 | (🆕) letterbox demo | *Letterbox preprocessing diagram (aspect-preserving resize).* |

**Ch5 toplam: 6 figür** (5 ✏️ + 1 🆕)

---

## Chapter 6 — Implementation Details (+5 figür)

| # | Yol | Caption |
|---|-----|---------|
| 6.1 | (🆕) `tree` screenshot | *Repository folder structure (src/, configs/, scripts/, runs/).* |
| 6.2 | (📸) config.yaml screenshot | *Runtime configuration file annotated with key parameters.* |
| 6.3 | (📸) bytetrack.yaml screenshot | *Tracker configuration annotated with thresholds and buffer.* |
| 6.4 | (📸) Colab notebook | *Training notebook (08_train_mobese_v4.ipynb) executing the model.train(…) cell on a Tesla T4.* |
| 6.5 | (📸) GitHub commit history | *Project commit history — `git log --oneline` view on GitHub.* |

**Ch6 toplam: 5 figür** (sen alacaksın, hepsi screenshot)

---

## Chapter 7 — Experiments and Evaluation (+17 figür)

### 7.5. Training and Validation Results
| # | Yol | Caption |
|---|-----|---------|
| 7.1 | `A/` `results.png` | *v4 fine-tuning loss and validation metric curves (28 epochs, early-stopped from 60).* |
| 7.2 | `A/` `labels.jpg` | *v4 training-set class instance distribution.* |

### 7.6. Final Test Results
| # | Yol | Caption |
|---|-----|---------|
| 7.3 | `B/` `11_v3_v4_metrics_table.png` | *Overall metric comparison of v3 vs. v4 on the held-out test partition.* |
| 7.4 | `B/` `12_v3_v4_per_class_map.png` | *Per-class mAP@50 with delta annotations.* |

### 7.9. Robustness and Generalization
| # | Yol | Caption |
|---|-----|---------|
| 7.5 | `B/` `15_v4_by_subset.png` | *v4 performance across three data sub-populations.* |

### 7.10. Error Analysis (Ch8'e yönlendirme)
| # | Yol | Caption |
|---|-----|---------|
| 7.6 | `A/` `confusion_matrix.png` | *Validation confusion matrix (raw counts).* |
| 7.7 | `A/` `confusion_matrix_normalized.png` | *Validation confusion matrix (row-normalised).* |

### 7.11. Computational Performance
| # | Yol | Caption |
|---|-----|---------|
| 7.8 | (🆕) latency bar chart | *Single-frame inference latency: CPU (M2) vs. GPU (T4) per pipeline stage.* |

### 7.13. Visualization of Results
| # | Yol | Caption |
|---|-----|---------|
| 7.9 | `B/` `14_v3_v4_side_by_side.png` | *v3 vs. v4 side-by-side predictions on 8 test images.* |
| 7.10 | `A/` `val_batch0_pred.jpg` | *v4 prediction sample, validation batch 0.* |
| 7.11 | `A/` `val_batch1_pred.jpg` | *v4 prediction sample, validation batch 1.* |
| 7.12 | `A/` `val_batch2_pred.jpg` | *v4 prediction sample, validation batch 2.* |
| 7.13 | `A/` `val_batch0_labels.jpg` | *Ground-truth boxes for validation batch 0.* |
| 7.14 | `A/` `BoxF1_curve.png` | *Per-class F1 vs. confidence threshold.* |
| 7.15 | `A/` `BoxP_curve.png` | *Per-class precision curve.* |
| 7.16 | `A/` `BoxR_curve.png` | *Per-class recall curve.* |
| 7.17 | `A/` `BoxPR_curve.png` | *Per-class precision-recall trade-off.* |

**Ch7 toplam: 17 figür**

---

## Chapter 8 — Error Analysis (+14 figür)

### 8.3.1. Confusion matrix
| # | Yol | Caption |
|---|-----|---------|
| 8.1 | `A/` `confusion_matrix.png` | *Confusion matrix (raw counts).* |
| 8.2 | `A/` `confusion_matrix_normalized.png` | *Confusion matrix (row-normalised).* |

### 8.3.2. Per-class breakdown
| # | Yol | Caption |
|---|-----|---------|
| 8.3 | `A/` `BoxP_curve.png` | *Per-class precision curve.* |
| 8.4 | `A/` `BoxR_curve.png` | *Per-class recall curve.* |
| 8.5 | `A/` `BoxF1_curve.png` | *Per-class F1 curve.* |

### 8.3.3. Pipeline field test
| # | Yol | Caption |
|---|-----|---------|
| 8.6 | `B/` `17_empirical_pipeline.png` | *Field-test confusion matrix and metric bars.* |

### 8.4. Qualitative Error Analysis
| # | Yol | Caption |
|---|-----|---------|
| 8.7 | `A/` `val_batch0_pred.jpg` | *Predictions (v4) for validation batch 0.* |
| 8.8 | `A/` `val_batch0_labels.jpg` | *Ground truth for the same batch.* |
| 8.9 | (🆕) failure crop 1 | *Failure case — small motorcycle missed in 4K drone view.* |
| 8.10 | (🆕) failure crop 2 | *Failure case — pickup truck classified as car.* |
| 8.11 | (🆕) failure crop 3 | *Failure case — track break under bus occlusion (3-frame sequence).* |

### 8.5. Root Cause / 8.7 Failure Categorization
| # | Yol | Caption |
|---|-----|---------|
| 8.12 | (🆕) error pie chart | *Error type distribution (E-DET, E-TRK, E-STM, E-LOC).* |

### 8.6. Sensitivity Analysis
| # | Yol | Caption |
|---|-----|---------|
| 8.13 | `B/` `16_pipeline_metrics.png` | *Pipeline metric sensitivity envelope.* |

### 8.8. Comparison with Baseline Errors
| # | Yol | Caption |
|---|-----|---------|
| 8.14 | `B/` `12_v3_v4_per_class_map.png` | *Per-class v3 vs. v4 mAP@50 comparison (reused from 7.4).* |

**Ch8 toplam: 14 figür** (3 tanesi Ch7 ile ortak — net yeni 11)

---

## Chapter 9 — System / Application Demo (+10 screenshot)

| # | Yol | Caption |
|---|-----|---------|
| 9.1 | (📸) Gradio landing | *Gradio web interface — initial view with upload area.* |
| 9.2 | (📸) Polygon drawing — başlangıç | *First click on the frame to begin polygon definition.* |
| 9.3 | (📸) Polygon drawing — yarısı | *Polygon definition in progress (4 vertices placed).* |
| 9.4 | (📸) Polygon drawing — tamamlanmış | *Closed polygon overlay on the source frame.* |
| 9.5 | (📸) Processing progress | *Pipeline progress bar during inference.* |
| 9.6 | (📸) Video tab — sonuç | *Annotated output video with bounding boxes, zone overlay, and violation flags.* |
| 9.7 | (📸) Analytics tab | *Severity distribution and class-breakdown charts.* |
| 9.8 | (📸) Violations tab — table | *Violation log table with timestamps, classes, severity scores.* |
| 9.9 | (📸) Violations tab — gallery | *Cropped vehicle snapshot gallery of confirmed violations.* |
| 9.10 | (📸) `test_clip.sh` terminal | *Command-line pipeline execution and summary.* |

### Ek (CLI / DB)
| # | Yol | Caption |
|---|-----|---------|
| 9.11 | `results/<run>/output.mp4` key frame | *Annotated output video — sample violation frame.* |
| 9.12 | `results/<run>/crops/*.jpg` 6'lık grid | *Confirmed-violation cropped vehicle snapshots.* |
| 9.13 | (📸) DB Browser screenshot | *SQLite violation database viewed in DB Browser for SQLite.* |

**Ch9 toplam: 10-13 figür** (sen Gradio'yu çalıştırıp screenshot alacaksın)

---

## Chapter 10 — Ethics and Limitations
*Figür yok.*

---

## Chapter 11 — Conclusion and Future Work
*Figür yok (opsiyonel: tek bir "future work roadmap" diyagramı).*

---

## Chapter 12 — References
*Figür yok.*

---

# 📊 TOPLAM FİGÜR

| Bölüm | Hazır | (🆕) Üretilecek | (✏️) Çizilecek | (📸) Screenshot | TOPLAM |
|-------|-------|----------------|-----------------|------------------|--------|
| Ch3 | 0 | 0 | 3 | 0 | 3 |
| Ch4 | 14 | 2 | 0 | 0 | 16 |
| Ch5 | 0 | 1 | 5 | 0 | 6 |
| Ch6 | 0 | 1 | 0 | 4 | 5 |
| Ch7 | 16 | 1 | 0 | 0 | 17 |
| Ch8 | 9 | 5 | 0 | 0 | 14 |
| Ch9 | 0 | 0 | 0 | 13 | 13 |
| **TOPLAM** | **39** | **10** | **8** | **17** | **74** |

**Hazır (39):** Hemen Word'e sürükle.
**Ben üreteceğim (10):** Onay verirsen 30 dk içinde hazır.
**Sen çizeceksin (8):** draw.io ile, her biri 5-10 dk.
**Sen screenshot alacaksın (17):** Gradio + config + terminal, ~1 saat.

---

# 🚀 Sıralı Yapacaklar Listesi

## Aşama 1 — Hazır 39 figürü Word'e ekle (~2 saat)

İki klasörü aç:
```bash
open /Users/sertacakalin/Desktop/Projects/bitirme/hatched-area-violation-detection/runs/detect/mobese_v4/
open /Users/sertacakalin/Desktop/Projects/bitirme/hatched-area-violation-detection/docs/thesis/figures/
```

Bölüm sırasıyla yukarıdaki tablodan eksiksiz takip et.

## Aşama 2 — Ben 10 yeni figür üreteyim (~30 dk)

Onay ver, hemen başlarım:
- `class_showcase.png` (Ch4.5)
- `letterbox_demo.png` (Ch4.8 + Ch5.6)
- `latency_chart.png` (Ch7.8)
- 3 × failure case crop (Ch8.9-8.11)
- `error_pie_chart.png` (Ch8.12)
- Repo `tree` PNG (Ch6.1)
- (Bonus 2 ekstra varsa)

## Aşama 3 — Sen draw.io diyagramlarını çiz (~1.5 saat)

8 diyagram, her biri 5-15 dk:
- Ch3: 3 mimari karşılaştırma
- Ch5: 5 sistem diyagramı

**draw.io şablon önerim:**
1. https://app.diagrams.net açin
2. Sol panelden basit kutu/ok kullan
3. **File → Export As → PNG (300 DPI)**
4. PNG'yi `docs/thesis/figures/diagrams/` altına koy

## Aşama 4 — Gradio screenshot'ları al (~1 saat)

```bash
cd ~/Desktop/Projects/bitirme/hatched-area-violation-detection
python app.py
```

Tarayıcıdan http://localhost:7860 aç, 13 screenshot al (Cmd+Shift+4).

---

# ✅ Aşama 2 Hazırlık

Bana onay ver:
- **"Hadi 10 figürü üret"** → Hemen başlarım, 30 dk sonra hazır
- **"Önce sadece X, Y, Z"** → Belirttiğini üretirim

Hangileri öncelikli sence?
