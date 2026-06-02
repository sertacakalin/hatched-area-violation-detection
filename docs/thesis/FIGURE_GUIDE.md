# Tez Figür Kılavuzu (A → Z)

Tezdeki **her bölüm için hangi figürü, nereden alıp, hangi caption ile**
ekleyeceğin tek dosyada. Sırayla uygula.

**Proje kök yolu:**
`/Users/sertacakalin/Desktop/Projects/bitirme/hatched-area-violation-detection/`

**3 ana klasör (sadece bunları kullan):**
- `A/` = `runs/detect/mobese_v4/`
- `B/` = `docs/thesis/figures/comparison/`
- `C/` = `docs/thesis/figures/`

---

## Chapter 1 — Introduction
Figür yok (sadece metin).

---

## Chapter 2 — Problem Definition and Objectives
Figür yok (sadece metin / liste).

---

## Chapter 3 — Related Work
Figür yok (literatür özeti).

---

## Chapter 4 — Dataset and Preprocessing

### 4.1. Dataset Description
- **Figure 4.1** | `C/` `08_sample_grid.png`
  - Caption: *Representative frames from the six dataset sources used in the v4 production export.*

### 4.3. Data Structure and Features
- **Figure 4.2** | `C/` `09_bbox_overlay_examples.png`
  - Caption: *Example frames with overlaid YOLO bounding boxes (one per source).*

### 4.4. Data Splitting
- **Figure 4.3** | `C/` `01_source_distribution.png`
  - Caption: *Per-source image distribution across the v4 dataset.*

### 4.5. Data Preprocessing
- **Figure 4.4** | `A/` `train_batch0.jpg`
  - Caption: *Sample augmented training batch (mosaic + mixup + flip + HSV jitter).*

### 4.8. Class Imbalance Handling
- **Figure 4.5** | `C/` `02_class_distribution.png`
  - Caption: *Object-count distribution across the four target classes.*
- **Figure 4.6** | `C/` `03_source_class_heatmap.png`
  - Caption: *Source × class object-count heatmap.*

### 4.10. Data Visualization & Exploration
- **Figure 4.7** | `C/` `04_bbox_size_distribution.png`
  - Caption: *Bounding-box size distribution (small / medium / large, COCO definition).*
- **Figure 4.8** | `C/` `05_bbox_position_heatmap.png`
  - Caption: *Bounding-box centre-point density heatmap (whole dataset).*
- **Figure 4.9** | `C/` `06_objects_per_image.png`
  - Caption: *Per-image object-count histogram.*
- **Figure 4.10** | `C/` `07_manual_vs_pseudo.png`
  - Caption: *Manual vs. auto-labeled annotation breakdown by image and object counts.*

**Toplam Ch4: 10 figür**

---

## Chapter 5 — Methodology

### 5.2. System Architecture
- **Figure 5.1** | (✏️ ÇİZİLECEK — Word'de SmartArt / draw.io)
  - Caption: *End-to-end block architecture (FrameProvider → YOLOv8s + ByteTrack → ZoneManager → State Machine → ViolationLogger / Visualizer).*
  - **Yapacaksın:** kalem-kağıt veya draw.io'da 4 bloklu basit akış diyagramı.

### 5.4. Temporal State Machine (varsa)
- **Figure 5.2** | (✏️ ÇİZİLECEK — basit state diagram)
  - Caption: *Vehicle state machine: OUTSIDE → ENTERING → INSIDE → VIOLATION (with cooldown loop).*

**Toplam Ch5: 2 figür** (ikisi de elle çizilecek, draw.io önerim)

---

## Chapter 6 — Implementation Details
Figür gerek yok (kod blokları + tablolar yeterli). Opsiyonel olarak Gradio ekran görüntüsü Ch9'a saklanır.

---

## Chapter 7 — Experiments and Evaluation

### 7.5. Training and Validation Results
- **Figure 7.1** | `A/` `results.png`
  - Caption: *v4 fine-tuning loss and validation metric curves across 28 epochs (early-stopped from 60).*
- **Figure 7.2** | `A/` `labels.jpg`
  - Caption: *v4 training-set class instance distribution.*

### 7.6. Final Test Results
- **Figure 7.3** | `B/` `11_v3_v4_metrics_table.png`
  - Caption: *Overall metric comparison of v3 vs. v4 on the held-out test partition.*
- **Figure 7.4** | `B/` `12_v3_v4_per_class_map.png`
  - Caption: *Per-class mAP@50 with delta annotations between v3 and v4.*

### 7.9. Robustness and Generalization
- **Figure 7.5** | `B/` `15_v4_by_subset.png`
  - Caption: *v4 performance across the three data sub-populations (Roboflow, auto-labeled CCTV, manually verified drone).*

### 7.10. Error Analysis (Chapter 8'e yönlendirme)
- **Figure 7.6** | `A/` `confusion_matrix.png`
  - Caption: *Validation confusion matrix (raw counts).*
- **Figure 7.7** | `A/` `confusion_matrix_normalized.png`
  - Caption: *Validation confusion matrix (row-normalised).*

### 7.13. Visualization of Results
- **Figure 7.8** | `B/` `14_v3_v4_side_by_side.png`
  - Caption: *v3 vs. v4 side-by-side predictions on eight randomly sampled test images.*
- **Figure 7.9** | `A/` `val_batch0_pred.jpg`
  - Caption: *v4 prediction sample from validation batch 0.*
- **Figure 7.10** | `A/` `BoxF1_curve.png`
  - Caption: *Per-class F1 score as a function of confidence threshold.*
- **Figure 7.11** | `A/` `BoxPR_curve.png`
  - Caption: *Per-class precision-recall trade-off.*

**Toplam Ch7: 11 figür**

---

## Chapter 8 — Error Analysis

### 8.3.1. Confusion Matrix
- **Figure 8.1** | `A/` `confusion_matrix.png`
  - Caption: *Confusion matrix (raw counts).*
- **Figure 8.2** | `A/` `confusion_matrix_normalized.png`
  - Caption: *Confusion matrix (row-normalised).*

### 8.3.2. Per-Class Curves
- **Figure 8.3** | `A/` `BoxP_curve.png`
  - Caption: *Per-class precision curve.*
- **Figure 8.4** | `A/` `BoxR_curve.png`
  - Caption: *Per-class recall curve.*

### 8.3.3. Pipeline Field Test
- **Figure 8.5** | `B/` `17_empirical_pipeline.png`
  - Caption: *Field-test confusion matrix and metric bars (30-s window, 10 ground-truth events).*

### 8.4. Qualitative Error Analysis
- **Figure 8.6** | `A/` `val_batch0_pred.jpg`
  - Caption: *Predicted boxes (v4) for validation batch 0.*
- **Figure 8.7** | `A/` `val_batch0_labels.jpg`
  - Caption: *Ground-truth boxes for the same batch.*

### 8.6. Sensitivity Analysis
- **Figure 8.8** | `B/` `16_pipeline_metrics.png`
  - Caption: *Pipeline metric sensitivity envelope under three tracker-quality scenarios.*

### 8.8. Comparison with Baseline Errors
- **Figure 8.9** | `B/` `12_v3_v4_per_class_map.png`
  - Caption: *Per-class baseline (v3) vs. v4 mAP@50 comparison.*

**Toplam Ch8: 9 figür** (3 tanesi Ch7 ile ortak)

---

## Chapter 9 — System / Application Demo

### Gradio Web UI (📸 SCREENSHOT ALACAKSIN)

Bu bölümün figürleri **Gradio'yu çalıştırıp ekran görüntüsü almakla** üretilecek.

#### Adımlar:
1. `python app.py`
2. http://localhost:7860 aç
3. Aşağıdaki ekranlardan screenshot al (Cmd+Shift+4):

- **Figure 9.1** | (📸 SCREENSHOT) Ana sayfa, boş upload alanı
  - Caption: *Gradio web interface — initial view.*
- **Figure 9.2** | (📸 SCREENSHOT) Video upload + polygon çizim ekranı
  - Caption: *Interactive polygon definition by clicking on the first frame.*
- **Figure 9.3** | (📸 SCREENSHOT) Processing tamamlandıktan sonra "Video" tab
  - Caption: *Annotated output video with bounding boxes, zone overlay, and violation flags.*
- **Figure 9.4** | (📸 SCREENSHOT) "Violations" tab — tablo + thumbnail galerisi
  - Caption: *Violation log table and cropped vehicle snapshot gallery.*
- **Figure 9.5** | (📸 SCREENSHOT) "Analytics" tab
  - Caption: *Violation analytics chart (severity distribution / class breakdown).*

#### CLI Pipeline Demo (opsiyonel)
- **Figure 9.6** | (📸 SCREENSHOT) Terminal — `./test_clip.sh` çıktısı
  - Caption: *Command-line pipeline execution with violation count summary.*
- **Figure 9.7** | İhlal ânı snapshot (yapılan ihlalin tek bir frame'i)
  - Caption: *Sample violation frame from `results/<run>/frames/`.*

**Toplam Ch9: 5-7 screenshot** (kendi alırsın, ~10 dk)

---

## Chapter 10 — Ethics and Limitations
Figür yok (sadece metin).

---

## Chapter 11 — Conclusion and Future Work
Figür yok.

---

## Chapter 12 — References
Figür yok.

---

# 📊 Toplam Figür Sayısı

| Bölüm | Figür Sayısı | Kaynak |
|-------|--------------|--------|
| Ch4 | 10 | Tümü `C/` ve `A/`'dan hazır |
| Ch5 | 2 | ✏️ Sen çizeceksin (draw.io) |
| Ch7 | 11 | Tümü `A/` ve `B/`'den hazır |
| Ch8 | 9 (3 ortak) | Tümü `A/` ve `B/`'den hazır |
| Ch9 | 5-7 | 📸 Sen Gradio'dan screenshot alacaksın |
| **NET** | **~35 figür** | |

---

# 🚀 Hızlı Komutlar (Finder Aç)

İki ana klasörü açmak için:

```bash
open /Users/sertacakalin/Desktop/Projects/bitirme/hatched-area-violation-detection/runs/detect/mobese_v4/
open /Users/sertacakalin/Desktop/Projects/bitirme/hatched-area-violation-detection/docs/thesis/figures/comparison/
open /Users/sertacakalin/Desktop/Projects/bitirme/hatched-area-violation-detection/docs/thesis/figures/
```

Üçü açılır, Word'e sürükle-bırak.

---

# 🚫 KULLANMA — Eski / yedek dosyalar

| Dosya / Klasör | Sebep |
|----------------|-------|
| `/Users/sertacakalin/Desktop/Projects/bitirme/adsız klasör/` | Eski kopyalar, kafa karıştırır |
| `~/Desktop/Projects/bitirme/results.png` | v3'ün eski sonucu, v4 değil |
| `~/Desktop/Projects/bitirme/Resim1.png`, `Resim2.png` | Bilinmiyor, eski screenshot |
| `~/Desktop/Projects/bitirme/figure_4_*` | Var ama bizim daha iyilerimiz `C/`'de |
| `~/Desktop/Projects/bitirme/BoxR_curve.png` (üst seviye) | Eski v3 versiyonu |

---

# ✅ Çalışma Sırası Önerim

1. **Önce Ch4 figürlerini ekle** (10 figür, hepsi hazır)
2. **Ch7 figürlerini ekle** (11 figür, hepsi hazır)
3. **Ch8 figürlerini ekle** (9 figür, 3'ü Ch7 ile ortak — sadece 6 yeni)
4. **Ch5 diyagramlarını çiz** (2 figür — draw.io 30 dk)
5. **Ch9 screenshot'ları al** (Gradio çalıştır, 5-7 screenshot, 30 dk)

Toplam tahmini süre: **2-3 saat** Word'de yapıştırma + caption yazma.

---

# 🎨 Caption Stili Kuralı

Tezdeki diğer bölümlerle aynı format:

```
Figure X.Y. Sentence describing the figure. (Period at end.)
```

Örnekler:
- ✅ `Figure 7.3. Overall metric comparison of v3 vs. v4 on the test set.`
- ✅ `Figure 4.5. Object-count distribution across the four target classes.`
- ❌ "Comparison of v3 and v4" (eksik numarası, başlangıç noktası yok)
