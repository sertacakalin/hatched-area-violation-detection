# Notebook Review Raporu
Generated: 2026-04-05

## KRITIK SORUN (ONCE COZULMELI)

### HARDCODED API KEY — 2 notebook'ta var
- `03_vehicle_detection_finetuning.ipynb` Cell 4: `api_key="[ROTATED — use Colab Secrets]"`
- `11_master_pipeline.ipynb` Cell 7: `api_key='[ROTATED — use Colab Secrets]'`

Bu key zaten daha once bulunmustu. Hala duzeltilmemis. Roboflow'a giris yap, API key'i ROTATE et, sonra her iki notebook'ta da su hale getir:
```python
import os
rf = Roboflow(api_key=os.environ.get('ROBOFLOW_API_KEY', 'BURAYA_YAZMA'))
```
Colab'da: Secrets panelinden ROBOFLOW_API_KEY ekle, `userdata.get('ROBOFLOW_API_KEY')` ile oku.

---

## Notebook Bazli Durum

### 00_frame_extraction.ipynb — DURUM: OK (kucuk sorunlar)
**Amac:** Drive'daki videolardan her 2 saniyede 1 frame cikarip tez_frames klasorune yazar.

**Calisir mi:** Evet, Colab'da calisir.
- Drive mount: var ve dogru
- pip install: YOK — cv2 ve tqdm Colab'da zaten yuklu, sorun yok
- Hardcoded secret: YOK

**Sorunlar:**
1. VIDEO_DIR = `"/content/drive/MyDrive/İstanbul Trafiği Kayıt"` — Turkce karakter ve bosluk var. Drive'daki gercek klasor adi bu degilse hata verir. Kullanicinin kontrol etmesi gerekiyor.
2. Cell 8'de cam1..cam5 sabit listelenmis, ama video_name'den cam_name uretiliyor. Eger videolar cam1.mp4 gibi isimlendirilmemisse dosya adi uyusmaz, ancak bu minor.

**Cikti:** Frame'ler Drive'a kaydedilir — dogrudan Roboflow'a yuklenecek ham veri.

---

### 01_data_exploration.ipynb — DURUM: SORUN VAR
**Amac:** Video ozellikleri + YOLO formatinda etiketli veri setinin istatistikleri.

**Calisir mi:** Hayir, as-is Colab'da calisMAZ.

**Sorunlar:**
1. Drive mount YOK — `from google.colab import drive` hic yok.
2. `VIDEO_DIR = Path('data/videos/raw')` — relative path. Colab'da /content/... olmali. Videolar Drive'da, burasi bos.
3. pip install yorum satirinda (`# !pip install ultralytics...`) — ultralytics yuklu olmayabilir.
4. `import pandas as pd` Cell 4'te yokken kullaniliyor — Cell 8'de yoktan import ediliyor (Cell 2'de import yok, Cell 8 calisir ama Cell 4 fail eder).
5. Kaydedilen grafik: `results/sample_frames.png` — bu klasor yoksa FileNotFoundError.

**Duzeltme:** Drive mount ekle, path'leri `/content/drive/MyDrive/...` yap, pip install aktif hale getir, `results/` klasorunu `os.makedirs` ile olustur.

---

### 02_vehicle_detection_baseline.ipynb — DURUM: SORUN VAR
**Amac:** Pretrained YOLOv8n/s/m modellerinin trafik videosundaki performans karsilastirmasi.

**Calisir mi:** Kismi — pip install var ama bazi eksikler var.

**Sorunlar:**
1. Drive mount YOK — `VIDEO_PATH = 'data/videos/test/sample.mp4'` relative, Colab'da bu path bos.
2. pip install satiri yorum olarak yazilmis: `# !pip install ultralytics opencv-python matplotlib` — aktif degil. Colab'da ultralytics olmayabilir.
3. `results/baseline_comparison.png` ve `results/detection_sample.png` kaydediliyor ama `results/` klasoru olusturulmuyor.
4. GPU gereksinimleri icin runtime degistirme notu YOK (bu notebook GPU'dan fayda saglar).

**Cikti:** Grafik + FPS karsilastirmasi — tez icin kullanilabilir.

---

### 03_vehicle_detection_finetuning.ipynb — DURUM: KRITIK
**Amac:** YOLOv8s'i Istanbul trafik veri seti uzerinde fine-tune eder. Tezin en kritik adimi.

**Calisir mi:** API key degistirilince calisir.

**Sorunlar:**
1. HARDCODED API KEY: `api_key="[ROTATED — use Colab Secrets]"` — KRITIK, hemen duzelt.
2. Drive mount: VAR ve dogru.
3. pip install: `!pip install ultralytics roboflow -q` — var, dogru.
4. GPU kontrolu: var ve uyariliyor.
5. data.yaml'da `test` key'i eklenmemis, sadece train/val. Kucuk sorun.

**Cikti:** Fine-tuned best.pt modeli Drive'a kaydedilir, pretrained vs fine-tuned karsilastirmasi yapilir. Tez icin kritik sonuc.

---

### 04_tracking_comparison.ipynb — DURUM: OK (minor)
**Amac:** ByteTrack vs BoT-SORT tracker karsilastirmasi (FPS, ID switch, unique track sayisi).

**Calisir mi:** Evet, Drive mount ve pip install dogru.

**Sorunlar:**
1. Drive path: `'/content/drive/MyDrive/İstanbul trafiği kayıt'` — Turkce karakter + bosluk. Kamera ismi 03'tekiyle uyumsuz olabilir (03'te `istanbul_traffic_v1`, 04'te Turkce yol).
2. `botsort.yaml` Colab'da otomatik gelir mi? ultralytics ile gelir, sorun yok.
3. ID switch hesaplama yontemi basit (sadece kayip track'leri sayiyor), gercek MOTA degil. Tez'de bunu belirtmek gerekebilir.

**Cikti:** `tracker_comparison.png` Drive'a kaydedilir. Tez icin kullanilabilir.

---

### 05_plate_detection_training.ipynb — DURUM: SORUN VAR
**Amac:** Plaka tespit modelinin egitimi ve testi.

**Calisir mi:** Hayir, eksik.

**Sorunlar:**
1. Drive mount YOK.
2. pip install YOK.
3. `PLATE_MODEL = 'weights/plate_detector.pt'` — relative path, Colab'da dosya yok.
4. Egitim kodu tamamen yorum satirinda (`# model = YOLO(...)`) — egitim yapilmiyor, sadece test.
5. `data/datasets/plate_detection/images/test` — Colab'da bu klasor bos olacak.
6. `configs/dataset_plate.yaml` relative path, Colab'da yok.

**Not:** Bu notebook tez icin hazir degil. Ya egitim kodunu aktif et ve Drive path'lerini duzenle, ya da "Roboflow'dan hazir model indir" akisini tamamla.

---

### 06_ocr_comparison.ipynb — DURUM: SORUN VAR
**Amac:** PaddleOCR vs EasyOCR karsilastirmasi, Turk plaka okuma dogrulugu.

**Calisir mi:** Hayir, birden fazla kritik sorun.

**Sorunlar:**
1. Drive mount YOK.
2. pip install yorum satirinda: `# !pip install paddlepaddle paddleocr easyocr` — aktif degil.
3. `sys.path.insert(0, '..')` — Colab'da `..` = `/content`, src/ oraya yuklu olmak zorunda.
4. `from src.alpr.plate_reader import PlateReader` — Colab'da bu path sorunu yaratir. Pipeline kodu yuklu olmak zorunda.
5. `GROUND_TRUTH = {}` — bos. Karsilastirma hicbir zaman calisMAZ: `if GROUND_TRUTH:` blogu atlaniyor, sadece "Ground truth verisi girilmedi" basilacak.

**Cikti:** GROUND_TRUTH doldurulmadan hicbir metrik uretilmez. Bu notebook eksik.

---

### 07_violation_detection_eval.ipynb — DURUM: OK (en kapsamli notebook)
**Amac:** Uctan uca ihlal tespit pipeline'i. Video → Tespit → Takip → Zone kontrolu → Ihlal.

**Calisir mi:** Evet, pip install ve Drive mount dogru.

**Sorunlar:**
1. `ZONE_POINTS` placeholder koordinatlar var: `[400,350], [700,350]...` — kullanicinin kendi videosuna gore degistirmesi gerekiyor (notebook bunu acikca soyleye).
2. Colab'da matplotlib interaktif degil, zone secimi manuel koordinat girisiyle yapiliyor — bu kabul edilebilir.
3. `violation_frames` dict'te bbox `.astype(int)` cagrisi: Cell 6'da `v_data['bbox'].astype(int)` — bbox numpy array mi list mi? Cell 5'te `[float(x) for x in bbox]` list'e donusturuluyor. `.astype(int)` list'de calisMAZ — AttributeError verir.

**Duzeltme icin:** 
```python
# Degistir:
x1, y1, x2, y2 = v_data['bbox'].astype(int)
# Yap:
x1, y1, x2, y2 = [int(x) for x in v_data['bbox']]
```

**Cikti:** `violation_gallery.png`, `violation_stats.png`, `violations.csv` — tez icin guclu ciktilar.

---

### 08_condition_analysis.ipynb — DURUM: SORUN VAR (beklenebilir)
**Amac:** Gunduz/gece, yagmur/gunes, seyrek/yogun trafik kosullarinda performans karsilastirmasi.

**Calisir mi:** Hayir, 07'nin ciktilarına bagimli.

**Sorunlar:**
1. Drive mount YOK.
2. pip install YOK.
3. `results/violation_eval_results.csv` bekleniyor — bu dosya 07'de uretilmiyor! 07 `violations.csv` uretiyor, `violation_eval_results.csv` degil. Dosya adi uyumsuzlugu.
4. `VIDEO_CONDITIONS` dictionary'si hardcoded video isimlerle (`test_01.mp4` vs.) — gercek video isimleriyle uyusmayabilir.
5. `sys.path.insert(0, '..')` — Colab path sorunu.

**Duzeltme:** 07 ciktisinin ismini buraya gore uyarla veya tam tersi.

---

### 09_confidence_threshold_analysis.ipynb — DURUM: SORUN VAR
**Amac:** Farkli confidence threshold'larinda precision-recall trade-off.

**Calisir mi:** Hayir.

**Sorunlar:**
1. Drive mount YOK.
2. pip install YOK.
3. `sys.path.insert(0, '..')` — Colab path sorunu.
4. `from src.pipeline.pipeline_factory import create_pipeline` — src/ path sorunu.
5. `VIDEO = 'data/videos/test/sample.mp4'` — relative path, Colab'da yok.
6. `pipeline.run()` — bu metod `src/pipeline/pipeline.py`'de var mi? Kontrol edilmeli.
7. `stats['total_violations']` ve `stats['average_fps']` — run() metodunun bu anahtarlari dondurdugundan emin olmak lazim.

**Not:** Bu notebook 11_master_pipeline'da parcali olarak zaten var. Bagimsiz calistirma icin Drive path'leri ve kurulum eklemek gerekiyor.

---

### 10_final_results.ipynb — DURUM: SORUN VAR (yapay veri)
**Amac:** Tum deneylerin ozeti, tez tablolari, final grafik.

**Calisir mi:** Kismi — csv'ler varsa calisir, yoksa placeholder gosterir.

**Sorunlar:**
1. Drive mount YOK.
2. pip install YOK (seaborn, pandas lazim — Colab'da var ama ultralytics lazim).
3. Tablo 1 (`vehicle_comparison`) tamamen `None` — doldurulmayi bekliyor.
4. Tablo 2 (`tracker_comparison`) tamamen `None`.
5. Final ozet grafik (Cell 11) tum placeholder: `placeholder_f1 = [0.85, 0.80, 0.75, 0.65]` — bu degerler tez'e girmemeli!
6. Dosya path'leri relative (`results/...`) — Colab'da yok.

**Kritik not:** Bu notebook su anda teze hazir degil. Diger notebook'lardan CSV'ler uretilip burada toplandi. Placeholder'lari gercek degerlerle doldurmadan teze konmasin.

---

### 11_master_pipeline.ipynb — DURUM: KRITIK + SORUN VAR
**Amac:** Tek notebook'ta tum pipeline — egitimden demo video'ya kadar. Tezin ana notebook'u.

**Calisir mi:** API key duzeltilince ve pipeline_code.zip Drive'a yuklendikten sonra calisir.

**Sorunlar:**
1. HARDCODED API KEY: Cell 7'de `api_key='[ROTATED — use Colab Secrets]'` — KRITIK, hemen duzelt.
2. `pipeline_code.zip` Drive'a yuklenmeyi bekliyor — Cell 4 bunu kontrol ediyor ve uyariliyor. Bu akis dogru.
3. VIDEO_DIR: `'istanbul_trafik_kayit'` — 00'daki `'İstanbul Trafiği Kayıt'` ile uyumsuz. Kendi Drive'indaki gercek ismi yaz.
4. Cell 21 `GROUND_TRUTH` tamamen bos — P/R/F1 hesaplanabilmesi icin videolari izleyip doldurman gerekiyor.
5. Cell 14: `ZONE_POLYGONS` sozlugu var ama tum kameralar icin placeholder. Her kamera icin gercek koordinatlar girilmeli.
6. Cell 15: Interaktif zone secimi (`onclick`) Colab'da matplotlib backend sorunu cikabilir — test edilmeli.

**Cikti:** En kapsamli notebook. Dogru calistirildiginda:
- Fine-tuned model (Drive'da)
- Tum kameralar icin ihlal tablosu
- P/R/F1 metrikleri (GT doldurulursa)
- Demo video (ihlalleri isaretlenmi)
- Tez grafikleri

---

## Ozet Tablo

| Notebook | Durum | En Buyuk Sorun |
|----------|-------|----------------|
| 00_frame_extraction | OK | Video path Turkce karakter |
| 01_data_exploration | SORUN VAR | Drive mount yok, relative path'ler |
| 02_vehicle_detection_baseline | SORUN VAR | Drive mount yok, pip install yorum |
| 03_vehicle_detection_finetuning | KRITIK | Hardcoded API key |
| 04_tracking_comparison | OK | Drive path Turkce karakter uyumu |
| 05_plate_detection_training | SORUN VAR | Drive mount yok, egitim kodu yorum |
| 06_ocr_comparison | SORUN VAR | GROUND_TRUTH bos, pip yorum |
| 07_violation_detection_eval | OK* | bbox.astype(int) bug, zone placeholder |
| 08_condition_analysis | SORUN VAR | 07 cikti dosyasi adi uyumsuz |
| 09_confidence_threshold_analysis | SORUN VAR | Drive mount yok, relative path |
| 10_final_results | SORUN VAR | Tum tablolar None/placeholder |
| 11_master_pipeline | KRITIK | Hardcoded API key, GT bos |

*07 en kullanilabilir notebook, minor bug var

---

## Oncelik Sirasi (Yapilacaklar)

### HEMEN (blockers):
1. 03 ve 11'deki API key'i rotate et ve degistir (Colab Secrets kullan)

### ONCE BUNLAR (tez icin zorunlu):
2. 11_master_pipeline'daki VIDEO_DIR'i kendi Drive'indaki gercek klasor ismiyle eslestir
3. 11_master_pipeline Cell 21'de GROUND_TRUTH'u videolari izleyerek doldur (en az 10-15 ihlal/kamera)
4. Her kamera icin ZONE_POLYGONS koordinatlarini gir (11 Cell 14)
5. 07 Cell 6: `v_data['bbox'].astype(int)` → `[int(x) for x in v_data['bbox']]`

### SONRA (tez kalitesi icin):
6. 10_final_results tablolarini gercek deney degerleriyle doldur
7. 01, 02, 05, 08, 09 notebook'lara Drive mount + pip install + path duzeltmeleri ekle

### YAPMA (zaman kaybi):
- 05 ve 06'yi tamamen yeniden yazmak — 11 master bunlari kapsiyor
- Her notebook'u ayri ayri debug etmek — sadece 11'i calistir

---

## Pipeline Akisi (Dogru Sira)

```
11_master_pipeline (tek notebook, tum islemleri kapsar)
  └─ Adim 1: Drive mount + kurulum
  └─ Adim 2: YOLOv8 fine-tune (Roboflow → egitim → best.pt)
  └─ Adim 3: Pretrained vs fine-tuned karsilastirma
  └─ Adim 4: Her kamera icin pipeline calistir
  └─ Adim 5: Ground truth ile P/R/F1 hesapla
  └─ Adim 6: Grafik + tablo uret
  └─ Adim 7: Demo video uret
```

---

## Tez Icin Hazir Ciktilar (Dogru Calisinca)

| Cikti | Uretici Notebook | Tez Bolumu |
|-------|------------------|-----------|
| YOLOv8 egitim grafikleri (results.png, confusion_matrix.png) | 03, 11 | Yontem / Deneyler |
| Pretrained vs Fine-tuned tablo | 03, 11 | Sonuclar |
| ByteTrack vs BoT-SORT karsilastirma | 04 | Yontem |
| Ihlal galeri gorselleri | 07, 11 | Sonuclar |
| violations.csv | 07, 11 | Ekler |
| P/R/F1 metrikleri | 11 | Sonuclar |
| Demo video | 11 | Sunum |

