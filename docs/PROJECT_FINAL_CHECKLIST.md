# Tezi Bitirmek İçin Kalan İşler

Bu dosya senin **manuel olarak yapman gereken** her şeyi listeler.
Otomatik düzeltilebilen kısımlar zaten kodda; burada kalanlar:
- Veri toplama / etiketleme (manuel iş)
- Roboflow / Drive bilgisi alma (sadece senin hesaplarında var)
- Tez metni / figürleri (yazılı iş)

> **Hedef:** Tez savunmasında elinde sayısal sonuç + reproducibility +
> demo videosu olsun. Aşağıdaki her madde, savunmada sorulduğunda
> cevabın olmasını sağlayacak somut bir çıktı üretiyor.

---

## 0. Şu anki durum (otomatik kontrol edildi)

| Bileşen | Durum |
|--------|-------|
| Vehicle detection (`best_v3.pt`, yolov8m, mAP50=0.808) | ✅ Çalışıyor |
| Plate detection (`plate.pt`, yolov8n, mAP50=0.982) | ✅ Çalışıyor |
| Plate OCR (EasyOCR + TR 81-il validation) | ✅ Çalışıyor |
| ByteTrack + state machine + severity | ✅ Çalışıyor |
| SQLite (WAL mode, idempotent migrations) | ✅ Çalışıyor |
| Gradio web demo | ✅ Çalışıyor |
| `evaluate_with_ground_truth.py` Pipeline kullanıyor | ✅ Refactor tamam |
| Ground truth annotations | ❌ **Boş — Bölüm 1** |
| `dataset_info.yaml` TODO'lar | ❌ **Bölüm 2** |
| Unit test suite | ❌ **Bölüm 4** |
| Tez metni | ❌ **Bölüm 6** |

---

## 1. ⭐ ZORUNLU: Ground Truth Etiketleme

> **Neden:** Bu olmadan Precision/Recall/F1 hesaplanamaz. Tez Section 7
> (Experiments) tamamen boş kalır. Jüri "modelin kalitesi nedir?"
> diye sorar, sayı veremezsin.

### Hedef

| Video | Etiketlenmesi gereken ihlal sayısı | Süre tahmini |
|-------|-----------------------------------|--------------|
| `data/videos/test/cam1_test.mp4` | 20-30 ihlal | ~1 saat |
| `data/videos/test/cam4_30s.mp4`  | 20-30 ihlal | ~1 saat |
| (opsiyonel) 3. kamera açısı | 20+ ihlal | bonus |

**Toplam minimum: 40-60 ihlal etiketi** — istatistiksel anlamlı P/R/F1 için.

### Adım adım

1. **VLC veya MPV aç**: `vlc data/videos/test/cam1_test.mp4`
2. **Frame numarasını göster**: VLC'de `Tools → Current Frame Info` veya
   `time_seconds × fps` hesapla.
3. **Her ihlal için**:
   - Aracın taralı alana **girdiği** kareyi bul → `start_frame`
   - Aracın **çıktığı** kareyi bul → `end_frame`
   - Tipi belirle:
     - `LANE_CHANGE` — kısa çapraz geçiş (<0.5s), şerit değişiyor
     - `CRUISING` — bölge içinde >0.5s ilerliyor
     - `EDGE_CONTACT` — sadece kenara değdi (sınır vakası)
   - Aracı sınıflandır: `car`, `truck`, `bus`, `motorcycle`
   - Kısa not yaz: "beyaz Renault sağdan sola kaydı"
4. **JSON'a yaz**: `data/ground_truth/<video_basename>.json`

### JSON şablonu

```json
{
  "video": "cam1_test.mp4",
  "fps": 30,
  "annotator": "Sertaç Akalın",
  "description": "E-5 Avcılar üst geçidi, gündüz, ılık hava",
  "violations": [
    {
      "id": "v001",
      "start_frame": 245,
      "end_frame": 281,
      "vehicle_class": "car",
      "type": "LANE_CHANGE",
      "notes": "Beyaz sedan sağdan sola çapraz şerit değiştirdi"
    },
    {
      "id": "v002",
      "start_frame": 510,
      "end_frame": 605,
      "vehicle_class": "truck",
      "type": "CRUISING",
      "notes": "Kamyon ~3 saniye taralı alanda ilerledi"
    }
  ]
}
```

> Tam şema: `data/ground_truth/README.md`

### Sınır vakaları için politika kararları (önceden belirle, tutarlı kal)

- Ön tekerlek girip arka çıktıysa? → **EDGE_CONTACT** olarak etiketle.
- Bir araç bölgeye girip çıkıp tekrar girdiyse? → **2 ayrı ihlal**.
- Motosiklet 2 saniye bölgede ama küçük kaldığı için detector kaçırırsa?
  → Yine de etiketle (recall ölçmek için).
- Kamyon, sürüş yönü doğru ama kısa kenar dokunuşu varsa?
  → **EDGE_CONTACT** + notes'ta belirt.

### Doğrulama

Ground truth'u tamamladıktan sonra:

```bash
source venv/bin/activate
python scripts/evaluate_with_ground_truth.py \
  --video data/videos/test/cam1_test.mp4 \
  --ground-truth data/ground_truth/cam1_test.json \
  --zone configs/zones/cam1_test.json
```

Çıktı: `results/evaluation/eval_cam1_test_bytetrack.json` — Precision/Recall/F1 + severity dağılımı.

**Beklenen P/R aralıkları (sağlık çek):**
- Precision > 0.70 normal (ground truth dahil etmediğin gerçek
  ihlaller "FP" gibi görünebilir)
- Recall > 0.60 yeterli (kaçırma daha kritik problem)
- F1 > 0.65 makbul

Bu sayılardan çok daha düşükse: ya ground truth etiketin yanlış, ya
config tuning gerekli (`min_frames_in_zone`, `confidence_threshold`).

---

## 2. ⭐ ZORUNLU: `configs/dataset_info.yaml` TODO'larını doldur

> **Neden:** Tez Section 4 (Dataset) bu sayıları kullanır.
> "Kaç frame etiketledim, hangi sınıflar nasıl dağıldı?" sorusuna
> cevap olmazsa Dataset bölümü yazılamaz.

YAML dosyasında 3 dataset var: `dataset_v1`, `dataset_v3` (production),
`dataset_plate`. Her biri için **şu bilgileri Roboflow'dan al**:

### Roboflow'dan alınacak bilgiler

Roboflow'a giriş yap → her dataset için:

1. **Workspace + project slug**:
   - URL: `https://app.roboflow.com/<workspace>/<project>`
   - Örnek: `app.roboflow.com/sertacs-workspace/istanbul-traffic-vehicles`
   - workspace = `sertacs-workspace`, project = `istanbul-traffic-vehicles`

2. **Latest version number**:
   - Sol menüde "Versions" → en son versiyon numarası
   - v1 için zaten `2` yazıyor (model checkpoint'inden çıktı)
   - v3 ve plate için kendin gir

3. **Total images + train/val/test split**:
   - "Generate" sayfasına git → "Total Images" + "Split"
   - veya: dataset'i indirip `labels/train`, `labels/val`, `labels/test`
     klasörlerindeki `.txt` dosyalarını say

4. **Class counts** (her sınıfta kaç bbox?):
   - "Health Check" sekmesinde class distribution grafiği
   - veya: tüm `labels/*/*.txt` dosyalarındaki ilk sütunu (class_id) say

5. **Public URL (varsa)**:
   - Eğer dataset'ini public yaptıysan: "Versions → Download → API"
     altında URL var
   - Yapmadıysan: TODO_PASTE'i `private` olarak işaretle

### Drive'dan alınacak bilgiler (training section)

Her training run için Drive'da `MyDrive/tez_models/<run_dir>/` içinde:

| Dosya | Ne için |
|-------|---------|
| `results.csv` | `epochs_actual` (son satır), `best_epoch` (en iyi mAP50-95 satırı) |
| `train_log.txt` veya Colab log | `training_hours` (toplam süre) |
| Drive dosya tarihi | `run_date` |
| Klasör adı | `drive_run_dir` |

### Doldurma kontrol listesi

`configs/dataset_info.yaml` içinde **TODO içeren her satırı** doldur:

```bash
# Hızlı tespit:
grep -n "TODO" configs/dataset_info.yaml | wc -l  # şu an: ~25 satır
```

Doldurma sonrası:

```bash
# YAML hâlâ valid mi?
python -c "import yaml; yaml.safe_load(open('configs/dataset_info.yaml'))"
# TODO sayısı:
grep -c "TODO" configs/dataset_info.yaml  # hedef: 0
```

### Source videos

`source_videos.cameras` için:
- Drive'da `MyDrive/İstanbul Trafiği Kayıt/` içinde kaç farklı kamera açısı var?
- Her açı = 1 kamera lokasyonu (E-5 Avcılar üst geçit-1, üst geçit-2, vb.)

Yaz: `cameras: 3` (ya da kaç tane varsa).

---

## 3. ⭐ ZORUNLU: Untracked dosyaları triaj et

Şu anda untracked dosyalar:

```
archive/                                    # Eski prototype kodu
configs/zones/cam1_test.json                # Test için zone (track edilmeli)
configs/zones/cam1_test_4k.json             # 4K test zone (track edilmeli)
docs/BITIRME.md                             # Tez ana dosyan?
docs/DATASET_ARASTIRMA.md                   # Dataset araştırma notu
docs/IYILESTIRME.md                         # İyileştirme notları
docs/KOMUTLAR.md                            # Komut cheatsheet
docs/METODOLOJI.md                          # Metodoloji notu
docs/SADELESTIRME_TEST.md                   # Test notları
docs/YAPILACAKLAR.md                        # Eski TODO listesi
docs/figures/                               # Tez figürleri
notebooks/05_train_mobese_v3.ipynb          # ⚠ v3 production training notebook!
scripts/generate_dataset_visualizations.py  # Dataset stat figürleri üretiyor
scripts/live_stream.py                      # Canlı stream test
scripts/live_with_zone.py                   # Live + zone test
scripts/show_violations.py                  # Sonuçları göster
test_cam1.sh                                # Test shell script
```

### Karar matrisi

**Mutlaka commit:**
- `notebooks/05_train_mobese_v3.ipynb` → v3 production modelin reproducibility'si bu
- `configs/zones/cam1_test.json`, `cam1_test_4k.json` → test için zone tanımları
- `docs/figures/` → tez figürleri (binary olabilir, dikkatli)
- `docs/METODOLOJI.md` → tez metodoloji notu
- `scripts/generate_dataset_visualizations.py` → tez figürlerini üretiyorsa
- `scripts/show_violations.py` → results inspection için

**Triaj et (içeriğe bakıp karar ver):**
- `docs/BITIRME.md` — tez metnin mi yoksa not mu? Tez metniyse commit. Not'sa `.gitignore` ekle.
- `docs/IYILESTIRME.md`, `KOMUTLAR.md`, `YAPILACAKLAR.md` — kişisel notlar mı? Eklemek istiyor musun? `.gitignore`'a ekleyebilirsin.
- `archive/` — README "historical reference" diyor, içine bakıp commit etmeye karar ver.

**Sil veya gitignore:**
- `test_cam1.sh` → tek seferlik test'se sil
- `scripts/live_stream.py`, `live_with_zone.py` → tezde kullanmıyorsan sil

### Komutlar

```bash
# Mutlaka commit edileceklerin için:
git add notebooks/05_train_mobese_v3.ipynb \
        configs/zones/cam1_test.json \
        configs/zones/cam1_test_4k.json \
        docs/METODOLOJI.md \
        scripts/generate_dataset_visualizations.py \
        scripts/show_violations.py

# Personal notlar için (.gitignore'a ekle):
echo "docs/BITIRME.md" >> .gitignore
echo "docs/IYILESTIRME.md" >> .gitignore
echo "docs/YAPILACAKLAR.md" >> .gitignore
echo "docs/KOMUTLAR.md" >> .gitignore
echo "docs/SADELESTIRME_TEST.md" >> .gitignore
echo "docs/DATASET_ARASTIRMA.md" >> .gitignore
echo "test_cam1.sh" >> .gitignore

# Sonra commit et
git commit -m "chore: track v3 training notebook + test zones + thesis docs"
```

---

## 4. ÖNEMLİ: Unit test suite kur

> **Neden:** Refactor sırasında state machine, severity, plaka
> validasyonu silently bozulabilir. Tez jürisine "kodum çalışıyor"
> demek yerine "kodum testlerle korunuyor" demek istiyorsun.

### Klasör yapısı oluştur

```bash
mkdir -p tests/unit
touch tests/__init__.py tests/unit/__init__.py
```

### En kritik 6 test dosyası

**`tests/unit/test_tr_plate.py`** — TR plaka format validasyonu:
```python
from src.plate.tr_plate import normalize_tr_plate, validate_tr_plate, TR_CITY_CODES

def test_normalize_strips_spaces_and_uppercases():
    assert normalize_tr_plate("34 abc 1234") == "34ABC1234"

def test_validate_istanbul_plate():
    valid, code, name = validate_tr_plate("34ABC1234")
    assert valid is True
    assert code == "34"
    assert name == "İstanbul"

def test_validate_rejects_invalid_city_code():
    valid, code, name = validate_tr_plate("99XYZ123")
    assert valid is False

def test_validate_short_plate_06_X_1234():
    # "06 K 1234" formatı (1 harfli)
    valid, code, _ = validate_tr_plate("06K1234")
    assert valid is True

def test_all_81_city_codes_present():
    for i in range(1, 82):
        assert f"{i:02d}" in TR_CITY_CODES
```

**`tests/unit/test_state_machine.py`** — durum geçişleri:
```python
from src.violation.state_machine import VehicleStateMachine
from src.core.data_models import VehicleState

def test_outside_to_violation_after_min_frames():
    sm = VehicleStateMachine(min_frames_in_zone=5, cooldown_frames=10, exit_frames=3)
    states = []
    for _ in range(5):
        new_state, is_new = sm.update(track_id=1, is_in_zone=True)
        states.append(new_state)
    # 1: OUTSIDE→ENTERING, 2: ENTERING→INSIDE, 3-4: INSIDE,
    # 5: INSIDE→VIOLATION (frames_in_zone >= 5)
    assert states[-1] == VehicleState.VIOLATION

def test_no_double_violation_within_cooldown():
    sm = VehicleStateMachine(min_frames_in_zone=2, cooldown_frames=10)
    new_violations = 0
    for _ in range(20):
        _, is_new = sm.update(1, True)
        if is_new:
            new_violations += 1
    # cooldown nedeniyle sadece 1 ihlal
    assert new_violations == 1

def test_exit_after_3_frames_outside():
    sm = VehicleStateMachine(min_frames_in_zone=2, exit_frames=3)
    for _ in range(2):
        sm.update(1, True)
    for _ in range(3):
        sm.update(1, False)
    state = sm.get_state(1)
    assert state.state == VehicleState.OUTSIDE
    assert state.frames_in_zone == 0
```

**`tests/unit/test_severity.py`**:
```python
from src.violation.severity import SeverityScorer, SeverityLevel
from src.violation.trajectory import TrajectoryMetrics

def test_low_score_for_brief_edge_contact():
    metrics = TrajectoryMetrics(
        duration_frames=3, distance_in_zone=10, max_depth=5,
        crossing_angle=15, total_distance=20,
    )
    result = SeverityScorer().score(metrics)
    assert result.score < 30
    assert result.level in (SeverityLevel.DUSUK, SeverityLevel.ORTA)

def test_high_score_for_prolonged_cruising():
    metrics = TrajectoryMetrics(
        duration_frames=120, distance_in_zone=300, max_depth=80,
        crossing_angle=5, total_distance=350,
    )
    result = SeverityScorer().score(metrics)
    assert result.score > 50
```

**`tests/unit/test_zone_manager.py`**:
```python
from src.zones.zone_manager import ZoneManager
import json, tempfile, os

def test_point_in_polygon():
    zone_data = {
        "zones": [{
            "zone_id": "z1", "name": "test",
            "polygon": [[0,0],[100,0],[100,100],[0,100]],
        }]
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(zone_data, f); path = f.name
    try:
        zm = ZoneManager(path, polygon_buffer=0)
        is_in, zid = zm.is_point_in_zone((50, 50))
        assert is_in
        assert zid == "z1"
        is_in, _ = zm.is_point_in_zone((150, 150))
        assert not is_in
    finally:
        os.unlink(path)
```

**`tests/unit/test_database_migration.py`**:
```python
import sqlite3, tempfile, os
from src.storage.database import ViolationDatabase

def test_migration_adds_missing_columns_to_old_db():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "old.db")
        # Eski şema
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE violations (
                id INTEGER PRIMARY KEY, event_id TEXT UNIQUE,
                track_id INTEGER, frame_number INTEGER,
                timestamp_sec REAL, vehicle_class TEXT,
                vehicle_confidence REAL, vehicle_bbox TEXT,
                zone_id TEXT, frames_in_zone INTEGER,
                vehicle_crop_path TEXT, frame_image_path TEXT
            );
        """)
        conn.commit(); conn.close()

        db = ViolationDatabase(db_path)
        cols = {row[1] for row in
                db._get_connection().execute("PRAGMA table_info(violations)")}
        for c in ("plate_text", "plate_crop_path", "severity_score",
                  "trajectory_metrics", "created_at", "video_source"):
            assert c in cols, f"{c} migration eksik"
        db.close()
```

**`tests/unit/test_trajectory.py`**:
```python
from src.violation.trajectory import TrajectoryAnalyzer

def test_trajectory_tracks_in_zone_duration():
    ta = TrajectoryAnalyzer()
    for i in range(10):
        ta.update(track_id=1, position=(50, 50 + i), is_in_zone=True)
    state = ta._states[1]
    assert state.frames_in_zone == 10
```

### Çalıştırma

```bash
pip install pytest
pytest tests/ -v
```

Hedef: **tüm 6 test PASS**. Coverage isteğe bağlı:

```bash
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 5. ÖNEMLİ: Demo videosu hazırla

> **Neden:** Tez savunmasında jüri "çalıştığını gösterin" der. Önceden
> hazırlanmış 30-60 saniyelik bir clip, canlı debug riski sıfırlar.

### Hedef

`docs/figures/demo_30s.mp4` — Pipeline output'u (annotated video):
- ~30 saniye kompakt clip
- En az 2-3 ihlal görünüyor olsun (bbox kırmızı + IHLAL etiketi)
- Plaka okuma çalışıyorsa plaka da görünsün
- Heatmap ekranı (opsiyonel)

### Üretim adımları

```bash
# 1. Tam pipeline çalıştır
source venv/bin/activate
python scripts/run_pipeline.py \
  --config configs/config.yaml \
  --video data/videos/test/cam1_test.mp4 \
  --zone configs/zones/cam1_test.json \
  --output results/demo_run

# 2. Output video → 30 saniyelik kesit
ffmpeg -i results/demo_run/output.mp4 \
       -ss 00:00:10 -t 30 -c:v libx264 -preset fast -crf 23 \
       -movflags +faststart \
       docs/figures/demo_30s.mp4

# 3. (Opsiyonel) GIF da yap, README'ye embed
ffmpeg -i docs/figures/demo_30s.mp4 \
       -vf "fps=10,scale=720:-1:flags=lanczos" \
       -loop 0 docs/figures/demo.gif
```

Video commit edilecek mi? Boyutuna bak:
- 30 sn @ 720p → ~5-15 MB → git'e koymakta sorun yok
- 30 sn @ 4K → ~50-100 MB → Drive'a koy, README'de link ver

---

## 6. ÖNEMLİ: Tez metni — Section 7 (Experiments)

> Bu sadece sen yapabilirsin (yazılı iş). Burada ne yazacağının
> outline'ını veriyorum.

### Section 7 outline (en az 8-12 sayfa)

#### 7.1 Deneysel Düzen

**Subsection 7.1.1 — Donanım**
- "Eğitim Google Colab Tesla T4 (16GB VRAM)
- "Inference Mac M-series (CPU/MPS) ve Linux + RTX (CUDA) üzerinde test edildi"
- "Test seti: %X frame Roboflow'dan, ground truth %Y video manuel etiketli"

**Subsection 7.1.2 — Hyperparametreler**
- `configs/dataset_info.yaml`'dan alıntı (training.v3 bloğu)
- Pipeline config: `configs/config.yaml`'dan critical değerler
  (`min_frames_in_zone=5`, `cooldown_frames=600`, `confidence=0.55`)

#### 7.2 Vehicle Detection Sonuçları

**Tablo 7.1 — Model karşılaştırması (`weights/README.md`'den):**

| Model | Base | Epoch | Precision | Recall | mAP50 | mAP50-95 |
|-------|------|-------|-----------|--------|-------|----------|
| yolov8s.pt (COCO baseline) | — | — | * | * | * | * |
| best.pt (v1, yolov8s) | yolov8s | 50 | 0.6389 | 0.5656 | 0.5969 | 0.4496 |
| **best_v3.pt (yolov8m) — production** | yolov8m | 100 | **0.7647** | **0.7873** | **0.8075** | **0.6530** |

> COCO baseline değerleri için: `notebooks/02_vehicle_detection_baseline.ipynb`'i çalıştır, val sonuçlarını al.

**Şekil 7.1 — Loss curves**: `MyDrive/tez_models/final_v3/results.png`
**Şekil 7.2 — Confusion matrix**: `confusion_matrix.png`
**Şekil 7.3 — PR curve**: `PR_curve.png`

#### 7.3 İhlal Tespiti Sonuçları (P/R/F1)

Bu **ground truth tamamlandıktan sonra** yazılabilir:

```bash
python scripts/evaluate_with_ground_truth.py \
  --video data/videos/test/cam1_test.mp4 \
  --ground-truth data/ground_truth/cam1_test.json \
  --zone configs/zones/cam1_test.json \
  --tolerance 30
```

**Tablo 7.2 — İhlal P/R/F1 (cam1_test.mp4):**

| Metric | Value | TP | FP | FN |
|--------|-------|----|----|-----|
| Precision | TODO | TODO | TODO | — |
| Recall | TODO | TODO | — | TODO |
| F1 | TODO | — | — | — |

**Tablo 7.3 — Şiddet skor dağılımı (KAYNAK / SEYİR / KENAR_TEMASI)**

#### 7.4 False Positive Analizi

Eval script otomatik üretiyor — `fp_analysis.threshold_impact`:

| min_severity | TP kalan | FP kalan | Yeni Precision |
|--------------|----------|----------|----------------|
| 15 | TODO | TODO | TODO |
| 25 | TODO | TODO | TODO |
| 35 | TODO | TODO | TODO |
| 50 | TODO | TODO | TODO |

Yorum: "Severity threshold N'in altındaki ihlalleri filtrelemek
precision'ı X→Y'ye çıkarıyor; recall trade-off Z."

#### 7.5 Plate Recognition Accuracy

**Bu ayrı bir manuel etiketleme istiyor** (yan-iş):
- Plate detection için: video'dan 30-50 plaka görüntüsü topla, manuel oku
- Pipeline'ın okuduğu vs. gerçek plaka karşılaştır:
  - **Detection rate** = (plate algılanan ihlal) / (toplam ihlal)
  - **OCR exact match rate** = (doğru okunan plaka) / (algılanan plaka)
  - **TR format validation rate** = (validation ✓ olan plaka) / (algılanan plaka)

#### 7.6 Performans

```bash
# FPS ölçümü
python scripts/run_pipeline.py --config configs/config.yaml \
  --video data/videos/test/cam1_test.mp4 \
  --zone configs/zones/cam1_test.json --no-save
# Çıktıdaki "average_fps" değerini al
```

**Tablo 7.4 — Donanıma göre FPS:**

| Donanım | Vehicle only | + Tracking | + Plate (only on violation) |
|---------|--------------|------------|------------------------------|
| Mac M-series CPU | TODO | TODO | TODO |
| RTX 3060 (CUDA) | TODO | TODO | TODO |
| Colab T4 | TODO | TODO | TODO |

---

## 7. NICE-TO-HAVE: CI / Pre-commit

> **Neden:** Tez teslimi sırasında "kod kaliteli" demek için
> objektif kanıt. PR'ları (varsa) korur.

`.github/workflows/test.yml`:

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt pytest pytest-cov
      - run: pytest tests/ -v
```

`.pre-commit-config.yaml` (opsiyonel):

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

```bash
pip install pre-commit
pre-commit install
```

---

## 8. NICE-TO-HAVE: Tez şablonu — diğer bölümler

| Section | Ne yazacak | Kaynak |
|---------|------------|--------|
| 1. Giriş | Problem, motivasyon | Kendin |
| 2. Literatür | YOLO, ByteTrack, OCR previous work | Google Scholar |
| 3. Yöntem | Architecture diagram + state machine | `README.md` + `src/` |
| 4. Dataset | Roboflow stats + örnek frame'ler | `dataset_info.yaml` (Bölüm 2'den sonra dolu olacak) |
| 5. Implementasyon | Code organization + tech stack | `README.md` |
| 6. Sistem entegrasyonu | Pipeline + Gradio + DB | `app.py` ekran görüntüleri |
| 7. Experiments | **Bölüm 6 outline** | Eval sonuçları |
| 8. Tartışma | Ne işe yarıyor, ne çalışmıyor | Eval'ın FN'leri, FP analizi |
| 9. Sonuç | Özet + future work | Kendin |

---

## Önerilen sıra

> Toplam tahmin: ~10-15 saat dağıtık iş. Hafta hafta plan:

**Hafta 1 (kritik veri):**
1. Bölüm 2 — `dataset_info.yaml` Roboflow TODO'ları (1-2 saat)
2. Bölüm 3 — Untracked dosyaları triaj + commit (30 dk)
3. Bölüm 1 — Ground truth `cam1_test.mp4` (1 video, ~1 saat)

**Hafta 2 (test + eval):**
4. Bölüm 4 — 6 unit test yaz + çalıştır (2-3 saat)
5. Bölüm 1 — Ground truth `cam4_30s.mp4` (1 video, ~1 saat)
6. Bölüm 5 — Demo video üret (30 dk)

**Hafta 3 (tez yazımı):**
7. Bölüm 6 — Section 7 (Experiments) yaz
8. Bölüm 6 — Section 4 (Dataset) yaz
9. Bölüm 7 — CI workflow ekle (opsiyonel)

Her bölüm bittikçe geri dön ve `Şu anki durum` tablosundaki ❌'leri ✅'ye çevir.

---

## Hazır olduğunda — savunma günü kontrol listesi

- [ ] `pytest tests/ -v` → tüm testler PASS
- [ ] `python scripts/run_pipeline.py ...` → cam1_test üzerinde çalışıyor
- [ ] `python scripts/evaluate_with_ground_truth.py ...` → P/R/F1 hesaplanıyor
- [ ] `python app.py` → Gradio açılıyor, video upload + zone tanım + Run çalışıyor
- [ ] `docs/figures/demo_30s.mp4` mevcut, 2+ ihlal gösteriyor
- [ ] `weights/best_v3.pt`, `weights/plate.pt` Drive'da yedekli
- [ ] Tez PDF'i template'e uygun, 12 section dolu
- [ ] `git status` temiz (her şey commit edilmiş)
- [ ] Roboflow dataset public link (varsa) çalışıyor
- [ ] Demo videosu yedek olarak USB'de
