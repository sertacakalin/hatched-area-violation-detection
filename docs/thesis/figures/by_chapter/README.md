# Tez Görselleri — Bölüm Bazlı Toplu Klasör

Tüm görsellerin **bölüm + sıra numarasıyla** yeniden adlandırıldığı tek klasör.
Dosya adı = direkt gideceği yer.

**Konum:** `docs/thesis/figures/by_chapter/`

**Açmak için:**
```bash
open /Users/sertacakalin/Desktop/Projects/bitirme/hatched-area-violation-detection/docs/thesis/figures/by_chapter/
```

---

## Chapter 4 — Dataset and Preprocessing (13 figür)

| Dosya | Word Bölümü | Caption |
|-------|-------------|---------|
| `ch4_01_sample_grid.png` | 4.1 | Representative frames from the six dataset sources. |
| `ch4_02_bbox_overlay.png` | 4.3 | Example frames with overlaid YOLO bounding boxes. |
| `ch4_03_source_distribution.png` | 4.4 | Per-source image distribution across the v4 dataset. |
| `ch4_04_class_distribution.png` | 4.4 | Object-count distribution across the four target classes. |
| `ch4_05_class_showcase.png` | 4.4 | Per-class examples (bus, car, motorcycle, truck) × 4 instances each. |
| `ch4_06_per_source_heatmaps.png` | 4.4 | Per-source bbox center-position density heatmaps. |
| `ch4_07_augmentation.png` | 4.7 | Online augmentation pipeline — same source, different variants. |
| `ch4_08_dataset_growth.png` | 4.4 / 4.13 | Dataset growth across project versions (v1 → v4). |
| `ch4_09_source_class_heatmap.png` | 4.8 | Source × class object-count heatmap. |
| `ch4_10_bbox_size.png` | 4.10 | Bounding-box size distribution (COCO small / medium / large). |
| `ch4_11_bbox_position.png` | 4.10 | Bounding-box centre-point density heatmap (whole dataset). |
| `ch4_12_objects_per_image.png` | 4.10 | Per-image object-count histogram. |
| `ch4_13_manual_vs_auto.png` | 4.10 | Manual vs. auto-labeled annotation breakdown. |

---

## Chapter 6 — Implementation Details (1 figür)

| Dosya | Word Bölümü | Caption |
|-------|-------------|---------|
| `ch6_01_repo_structure.png` | 6.1 | Repository folder structure (src/, configs/, scripts/, runs/). |

---

## Chapter 7 — Experiments and Evaluation (16 figür)

| Dosya | Word Bölümü | Caption |
|-------|-------------|---------|
| `ch7_01_training_curves.png` | 7.5 | v4 fine-tuning loss and validation metric curves (28 epochs, early-stopped from 60). |
| `ch7_02_class_distribution.jpg` | 7.5 | v4 training-set class instance distribution. |
| `ch7_03_v3_v4_overall.png` | 7.6 | Overall metric comparison of v3 vs. v4 on the test set. |
| `ch7_04_v3_v4_per_class.png` | 7.6 | Per-class mAP@50 with delta annotations. |
| `ch7_05_subset_performance.png` | 7.9 | v4 performance across three data sub-populations. |
| `ch7_06_confusion_matrix.png` | 7.10 | Validation confusion matrix (raw counts). |
| `ch7_07_confusion_normalized.png` | 7.10 | Validation confusion matrix (row-normalised). |
| `ch7_08_inference_latency.png` | 7.11 | Single-frame inference latency — CPU vs GPU per pipeline stage. |
| `ch7_09_per_class_radar.png` | 7.6 | Per-class performance radar (v4) across P, R, mAP@50, mAP@50-95. |
| `ch7_10_predictions_side_by_side.png` | 7.13 | v3 vs. v4 side-by-side predictions on 8 test images. |
| `ch7_11_val_pred_sample.jpg` | 7.13 | v4 prediction sample, validation batch 0. |
| `ch7_12_F1_curve.png` | 7.13 | Per-class F1 vs. confidence threshold. |
| `ch7_13_P_curve.png` | 7.13 | Per-class precision curve. |
| `ch7_14_R_curve.png` | 7.13 | Per-class recall curve. |
| `ch7_15_PR_curve.png` | 7.13 | Per-class precision-recall trade-off. |
| `ch7_16_train_batch_sample.jpg` | 7.5 | Augmented training batch sample (mosaic + mixup + flip + HSV). |

---

## Chapter 8 — Error Analysis (11 figür)

| Dosya | Word Bölümü | Caption |
|-------|-------------|---------|
| `ch8_01_confusion_matrix.png` | 8.3.1 | Confusion matrix (raw counts). |
| `ch8_02_confusion_normalized.png` | 8.3.1 | Confusion matrix (row-normalised). |
| `ch8_03_field_test.png` | 8.3.3 | Field-test confusion matrix and metric bars (30 s, 10 events). |
| `ch8_04_val_predictions.jpg` | 8.4 | Predicted boxes (v4) for validation batch 0. |
| `ch8_05_val_ground_truth.jpg` | 8.4 | Ground-truth boxes for the same batch. |
| `ch8_06_failure_case_1.png` | 8.4 | Failure case 1 — small motorcycle missed in 4K drone view. |
| `ch8_07_failure_case_2.png` | 8.4 | Failure case 2 — pickup truck classified as car. |
| `ch8_08_failure_case_3.png` | 8.4 | Failure case 3 — track break under bus occlusion. |
| `ch8_09_error_distribution.png` | 8.7 | Residual error type distribution (E-DET, E-LOC, E-TRK, E-STM). |
| `ch8_10_tracker_timeline.png` | 8.4 | ByteTrack identity lifetimes with occlusion-induced ID switch. |
| `ch8_11_sensitivity.png` | 8.6 | Pipeline metric sensitivity envelope under tracker-quality scenarios. |

---

## Chapter 9 — System / Application Demo (1 figür + screenshot'lar)

| Dosya | Word Bölümü | Caption |
|-------|-------------|---------|
| `ch9_01_severity_distribution.png` | 9.X (Analytics) | Distribution of violation severity scores. |

**Ek (📸 sen alacaksın):** Gradio screenshot'ları (10-12 tane), DB Browser screenshot, terminal çıktıları. Ayrıntı: `FIGURE_GUIDE_EXTENDED.md`'de.

---

## Chapter 11 — Conclusion and Future Work (1 figür)

| Dosya | Word Bölümü | Caption |
|-------|-------------|---------|
| `ch11_01_project_timeline.png` | 11.X (Project log) | Project timeline (Gantt) — approximate. |

---

# 📊 Toplam

| Bölüm | Figür Sayısı |
|-------|--------------|
| Ch4 | 13 |
| Ch6 | 1 |
| Ch7 | 16 |
| Ch8 | 11 |
| Ch9 | 1 (+ screenshot'lar) |
| Ch11 | 1 |
| **TOPLAM** | **43** |

---

# 🚀 Word'e Eklemenin En Kolay Yolu

1. Bu klasörü Finder'da aç:
   ```bash
   open /Users/sertacakalin/Desktop/Projects/bitirme/hatched-area-violation-detection/docs/thesis/figures/by_chapter/
   ```

2. Word'de tezi aç, **Chapter 4'ün altına in.**

3. **Sırayla** `ch4_01_*`, `ch4_02_*`, ... dosyalarını Word'e sürükle-bırak.

4. Her görselin altına `Figure 4.X. <caption>` yaz (caption'lar yukarıdaki tabloda).

5. Aynısını Ch7, Ch8, Ch9, Ch11 için yap.

---

# 💡 Bonus: Word'de Tek Tıkla Sırayla Ekleme

Word'de **Insert → Pictures → This Device** açılınca dosyalar **alfabetik** sıralanır. Bizim isimlendirme alfabetik = bölüm sıralı olduğu için **Cmd+A → Insert** ile bir bölümün tüm figürlerini tek seferde ekleyebilirsin (sonra captionlarını ekle).
