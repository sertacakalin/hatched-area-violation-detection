# PROJE MASTER PLAN
## Taralı Alanlarda Kaynak Yapan Araçların Tespiti
### Sertaç Akalın — 220303053 — İstanbul Arel Üniversitesi

---

# 1. PROJE NE YAPIYOR?

Bir video veriyorsun. Sistem şunu yapıyor:

1. Videodaki araçları buluyor (YOLOv8)
2. Her aracı kareler boyunca takip ediyor (ByteTrack)
3. Taralı alana giren araçları tespit ediyor
4. Gerçekten kaynak mı yapıyor yoksa anlık temas mı ayırt ediyor
5. İhlal yapan aracın plakasını okumaya çalışıyor (PaddleOCR)
6. Sonuçları kaydediyor (SQLite + ekran çıktısı)

```
GİRDİ                          ÇIKTI
─────                          ─────
Trafik videosu          →      İhlal listesi
(mp4, kamera stream)          (araç tipi, plaka, zaman,
                               ihlal şiddeti, kanıt görüntüsü)
```

---

# 2. ALGORİTMA — SİSTEM NASIL ÇALIŞIYOR?

## 2.1 Genel Akış

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│  Video   │───▶│  YOLOv8  │───▶│ ByteTrack│───▶│   Bölge   │
│  Kaynağı │    │  Tespit  │    │  Takip   │    │  Kontrolü │
└─────────┘    └──────────┘    └──────────┘    └─────┬─────┘
                                                      │
                        ┌─────────────────────────────┘
                        │
                        ▼
               ┌────────────────┐    ┌──────────┐    ┌────────┐
               │  State Machine │───▶│  Yörünge │───▶│ Şiddet │
               │  (4 Durum)     │    │  Analizi │    │  Skoru │
               └────────────────┘    └──────────┘    └───┬────┘
                                                          │
                        ┌─────────────────────────────────┘
                        │
                        ▼
               ┌──────────┐    ┌──────────┐    ┌──────────┐
               │  Plaka   │───▶│   Kayıt  │───▶│  Çıktı   │
               │  OCR     │    │  (SQLite)│    │  Video   │
               └──────────┘    └──────────┘    └──────────┘
```

## 2.2 Adım Adım Algoritma

### ADIM 1: Araç Tespiti (Hazır — YOLOv8)

Her video karesinde araçları bul.

```
Girdi:  Video karesi (1920x1080 piksel)
Çıktı:  Araç listesi → [{kutu, sınıf, güven}, ...]

Örnek çıktı:
  Araç 1: kutu=[100,200,300,350], sınıf="car",   güven=0.92
  Araç 2: kutu=[500,180,650,380], sınıf="truck",  güven=0.87
  Araç 3: kutu=[800,250,900,320], sınıf="car",   güven=0.76
```

Sınıflar: car, bus, truck, motorcycle, minibus

### ADIM 2: Araç Takibi (Hazır — ByteTrack)

Ardışık karelerde aynı araca aynı ID'yi ver.

```
Kare 1:  Araç A (ID:42) pozisyon (100, 300)
Kare 2:  Araç A (ID:42) pozisyon (120, 305)   ← aynı araç
Kare 3:  Araç A (ID:42) pozisyon (145, 310)   ← hala aynı
```

Bu olmazsa her karede "yeni araç gördüm" der, ihlal sayamazsın.

### ADIM 3: Bölge Kontrolü (Mevcut — Shapely)

Taralı alan önceden polygon olarak tanımlanmış (elle çizilmiş).
Her karede, her aracın alt merkez noktasının bu polygon içinde
olup olmadığı kontrol ediliyor.

```
Alt merkez noktası = aracın tekerlek seviyesi

  ┌──────────┐
  │          │
  │   ARAÇ   │
  │          │
  └────*─────┘  ← Bu nokta (x_merkez, y_alt)
       │
       ▼
  Polygon içinde mi? → Shapely: polygon.contains(Point(x, y))
```

### ADIM 4: State Machine (Mevcut — 4 Durum)

Her araç için ayrı bir durum makinesi çalışıyor:

```
OUTSIDE ──(bölgeye girdi)──▶ ENTERING
                                  │
                          (1+ kare devam)
                                  │
                                  ▼
                              INSIDE
                                  │
                         (5+ kare bölgede)
                                  │
                                  ▼
                            VIOLATION ⚠️
```

Mevcut parametreler:
- min_frames_in_zone = 5  (ihlal için minimum kare)
- cooldown_frames = 90    (aynı araca tekrar ihlal yazma bekleme)
- exit_frames = 3         (bölgeden çıkış için minimum kare)

---

# 3. SENİN KATKIN — YÖRÜNGE ANALİZİ + ŞİDDET SKORLAMA

Mevcut sistem sadece "bölgede mi?" + "kaç kare?" bilgisiyle karar
veriyor. Bu çok basit ve eksik.

Senin ekleyeceğin algoritma iki parçadan oluşuyor:

## 3.1 Yörünge (Trajectory) Analizi

Aracın bölge ile nasıl etkileştiğini analiz et.
Her takip edilen araç için pozisyon geçmişini tut.

```python
# Her araç için konum listesi
trajectory[track_id] = [
    (x1, y1),    # kare 1
    (x2, y2),    # kare 2
    (x3, y3),    # kare 3
    ...
]
```

Bu yörüngeden şu bilgiler çıkarılır:

### a) Giriş/Çıkış Noktaları

```
Araç bölgeye nereden girdi, nereden çıktı?

          GİRİŞ NOKTASI
              │
              ▼
    ╔═══════════════════╗
    ║                   ║
    ║   TARALI ALAN     ║
    ║                   ║
    ╚═════════════╤═════╝
                  │
                  ▼
          ÇIKIŞ NOKTASI

→ Giriş ve çıkış noktalarının bölge kenarındaki konumu
  ihlal tipini belirler.
```

### b) Bölge İçi Hareket Yönü

```
Araç bölge içinde nasıl hareket etti?

TİP A — Çapraz geçiş (şerit değiştirme):
    ╔════════════╗
    ║  ↗         ║    Araç sola/sağa çapraz geçti
    ║    ↗       ║    → Kaynak yapma ihlali
    ║      ↗     ║
    ╚════════════╝

TİP B — Düz geçiş (bölge içinde ilerleme):
    ╔════════════╗
    ║  →  →  →   ║    Araç bölge boyunca ilerledi
    ║            ║    → Taralı alanda seyir ihlali
    ╚════════════╝

TİP C — Kısa temas (kenardan sürtme):
    ╔════════════╗
    ║↗           ║    Araç kenardan girip hemen çıktı
    ╚════════════╝    → Hafif ihlal veya false positive
```

### c) Hesaplanacak Metrikler

```
Her ihlal için:

1. bölge_içi_mesafe
   Aracın bölge içinde kat ettiği toplam piksel mesafesi.
   Uzun mesafe = ciddi ihlal

2. bölge_içi_süre
   Aracın bölgede kaldığı kare sayısı (zaten var, genişletilecek)

3. geçiş_açısı
   Aracın hareket yönü ile bölge kenarı arasındaki açı.
   90° = dik geçiş (tam kesme)
   0°  = kenar boyunca sürtme

4. nüfuz_derinliği
   Aracın bölge merkezine en fazla ne kadar yaklaştığı.
   Bölge merkezine kadar = tam giriş
   Kenarda kaldı = kısa temas
```

## 3.2 Şiddet (Severity) Skorlama

Yukarıdaki metrikleri birleştirerek 0-100 arası şiddet skoru hesapla:

```
şiddet_skoru = (
    w1 × normalize(bölge_içi_süre)     +
    w2 × normalize(bölge_içi_mesafe)   +
    w3 × normalize(nüfuz_derinliği)    +
    w4 × normalize(geçiş_açısı)
)

Ağırlıklar (deneylerle belirlenecek):
  w1 = 0.30  (süre)
  w2 = 0.25  (mesafe)
  w3 = 0.30  (derinlik)
  w4 = 0.15  (açı)
```

Skor aralıkları:

```
 0-25  → DÜŞüK   : Kenardan sürtüp geçmiş, muhtemelen false positive
25-50  → ORTA     : Kısa süreli giriş, hafif ihlal
50-75  → YÜKSEK   : Net kaynak yapma, taralı alandan geçiş
75-100 → KRİTİK   : Uzun süre taralı alanda seyir
```

## 3.3 İhlal Tipi Sınıflandırma

Yörünge + skor birleştirilerek ihlal tipi belirlenir:

```
IF geçiş_açısı > 45° AND bölge_içi_süre < 15 kare:
    tip = "KAYNAK"          # Çapraz geçiş, şerit değiştirme

ELIF geçiş_açısı ≤ 45° AND bölge_içi_süre ≥ 15 kare:
    tip = "SEYİR"           # Bölge içinde ilerleme

ELIF nüfuz_derinliği < %30:
    tip = "KENAR_TEMASI"    # Kenardan geçmiş, hafif

ELSE:
    tip = "DİĞER"
```

## 3.4 Bu Neden Özgün Bir Katkı?

```
Mevcut literatürdeki yaklaşımlar:
  ✗ Sadece bölge içinde mi? (binary kontrol)
  ✗ Sadece kare sayısı eşiği (temporal filter)

Senin katkın:
  ✓ Yörünge tabanlı analiz (nasıl girdi, nasıl çıktı)
  ✓ Çok boyutlu şiddet skorlama (süre + mesafe + derinlik + açı)
  ✓ İhlal tipi sınıflandırma (kaynak / seyir / kenar teması)
  ✓ Parametrelerin deneysel optimizasyonu
```

---

# 4. PROJE İÇİN GEREKENLER

## 4.1 Veri

```
VİDEOLAR (kesinlikle lazım):
├── 5 test videosu (her biri 5-10 dakika)
│   ├── Taralı alan NET görünmeli
│   ├── İçinde gerçek ihlal olmalı
│   └── En az 2-3 farklı koşul
│
├── Kaynaklar:
│   ├── IBB kameraları — ekran kaydı (OBS)
│   ├── YouTube — sabit kamera videoları (yt-dlp)
│   └── Kendi çekimin — üst geçitten (telefon + tripod)
│
└── Toplam: ~1-2 saat video yeterli

ETİKETLİ VERİ (fine-tuning deneyi için):
├── 300-500 görüntü (Roboflow'da etiketle)
├── Sınıflar: car, bus, truck, motorcycle, minibus
├── Format: YOLOv8 (train/val/test split)
└── Roboflow auto-label ile hızlandırılabilir

GROUND TRUTH (değerlendirme için — kesinlikle lazım):
├── 5 test videosu için elle işaretlenmiş ihlaller
├── Her ihlal: başlangıç karesi, bitiş karesi, araç tipi
├── Toplam: 30-50 etiketli ihlal
└── Format: JSON
```

## 4.2 Çalışan Sistem

```
OLMAZSA OLMAZ:                     BONUS (vakit kalırsa):
├── Araç tespiti (YOLOv8)    ✅    ├── Dashboard (Streamlit)    ✅
├── Araç takibi (ByteTrack)  ✅    ├── Demo videosu
├── Bölge kontrolü           ✅    └── Plaka okuma              ✅
├── State machine            ✅
├── Yörünge analizi          ❌ ← SENİN YAZACAĞIN
├── Şiddet skorlama          ❌ ← SENİN YAZACAĞIN
└── İhlal sınıflandırma      ❌ ← SENİN YAZACAĞIN
```

## 4.3 Deneyler (minimum 3)

```
DENEY 1: Sistem doğruluğu
  Girdi:  5 test videosu + ground truth
  Çıktı:  Precision, Recall, F1 skoru
  Soru:   Sistem ihlalleri ne kadar doğru tespit ediyor?

DENEY 2: Pretrained vs Fine-tuned
  Girdi:  Aynı videolar, 2 farklı model
  Çıktı:  mAP karşılaştırma tablosu
  Soru:   İstanbul verisiyle eğitim fark yarattı mı?

DENEY 3: Koşul analizi
  Girdi:  Farklı koşullardaki test videoları
  Çıktı:  Koşul bazlı F1 karşılaştırması
  Soru:   Gece/yağmur performansı ne kadar düşüyor?

DENEY 4 (Senin algoritman):
  Girdi:  İhlal sonuçları + yörünge verileri
  Çıktı:  Şiddet skoru dağılımı, ihlal tipi dağılımı
  Soru:   Şiddet skorlama false positive'leri azaltıyor mu?
          Düşük skorlu "ihlaller" gerçekten ihlal mi?
```

## 4.4 Tez Dökümanı

```
~40-50 sayfa:

Bölüm 1 — Giriş (5 sayfa)
  Problem, motivasyon, kapsam

Bölüm 2 — Literatür (8 sayfa)
  YOLO ailesi, tracking, trafik ihlali tespiti

Bölüm 3 — Yöntem (12 sayfa)  ← EN ÖNEMLİ BÖLÜM
  Pipeline mimarisi
  State machine tasarımı
  ★ Yörünge analizi algoritması (senin katkın)
  ★ Şiddet skorlama formülü (senin katkın)
  ★ İhlal tipi sınıflandırma kuralları (senin katkın)

Bölüm 4 — Deneysel Sonuçlar (10 sayfa)
  4 deney + tablolar + grafikler

Bölüm 5 — Tartışma + Sonuç (5 sayfa)
  Başarılar, limitasyonlar, gelecek çalışma
```

---

# 5. BİLİNEN SORUNLAR VE LİMİTASYONLAR

Bunları tezde açıkça belirteceksin:

```
SORUN 1: Sıkışık trafikte yanlış alarm
  Araç mecburen taralı alanda → sistem ihlal diyor
  ÇÖZÜM: Şiddet skoru düşük çıkar → filtrelenebilir

SORUN 2: Gece performans düşüşü
  YOLOv8 gece %15-25 daha kötü
  ÇÖZÜM: Limitasyon olarak raporla

SORUN 3: Oklüzyon (araçlar birbirini kapatma)
  Tracker ID kaybeder → state machine sıfırlanır → ihlal kaçar
  ÇÖZÜM: Limitasyon olarak raporla

SORUN 4: Bounding box titremesi
  Kenar ihlalleri kaçırılır veya yanlış tetiklenir
  ÇÖZÜM: Polygon buffer (-10px) + overlap yöntemi

SORUN 5: Plaka okuma düşük doğruluk
  Gerçekçi beklenti: %60-80
  ÇÖZÜM: Preprocessing + limitasyon olarak raporla

SORUN 6: Manuel polygon tanımlama
  Her kamera için elle çizim gerekiyor
  ÇÖZÜM: Tez kapsamında sorun değil (5 video)
```

---

# 6. DOSYA YAPISI — SON HALİ

```
hatched-area-violation-detection/
│
├── src/
│   ├── core/
│   │   ├── data_models.py         # Veri modelleri (Detection, TrackedObject, vb.)
│   │   ├── config.py              # YAML konfigürasyon
│   │   ├── frame_provider.py      # Video okuma
│   │   └── visualizer.py          # Çizim
│   │
│   ├── detection/
│   │   └── vehicle_detector.py    # YOLOv8 araç tespiti
│   │
│   ├── tracking/
│   │   ├── bytetrack_wrapper.py   # ByteTrack
│   │   ├── botsort_wrapper.py     # BoT-SORT (karşılaştırma için)
│   │   └── deepsort_wrapper.py    # DeepSORT (karşılaştırma için)
│   │
│   ├── zones/
│   │   ├── zone_manager.py        # Polygon yönetimi
│   │   └── roi_selector.py        # Mouse ile polygon çizme
│   │
│   ├── violation/
│   │   ├── state_machine.py       # 4 durumlu state machine
│   │   ├── violation_detector.py  # İhlal tespit mantığı
│   │   ├── trajectory.py          # ★ YENİ — Yörünge analizi
│   │   └── severity.py            # ★ YENİ — Şiddet skorlama
│   │
│   ├── alpr/
│   │   ├── plate_detector.py      # Plaka tespiti (YOLOv8n)
│   │   ├── plate_reader.py        # OCR (PaddleOCR)
│   │   ├── plate_preprocessor.py  # Görüntü ön işleme
│   │   └── plate_validator.py     # Türk plaka format doğrulama
│   │
│   ├── storage/
│   │   ├── database.py            # SQLite
│   │   └── violation_logger.py    # İhlal kayıt
│   │
│   ├── dashboard/
│   │   └── app.py                 # Streamlit (bonus)
│   │
│   └── pipeline/
│       ├── pipeline.py            # Ana orkestratör
│       └── pipeline_factory.py    # Pipeline oluşturucu
│
├── configs/
│   ├── config.yaml                # Ana konfigürasyon
│   └── zones/                     # Polygon JSON'ları
│
├── scripts/
│   ├── run_pipeline.py            # Pipeline çalıştır
│   ├── select_roi.py              # ROI polygon çiz
│   └── run_evaluation.py          # Deney çalıştır
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_vehicle_detection_baseline.ipynb
│   ├── 03_vehicle_detection_finetuning.ipynb
│   └── 04_evaluation.ipynb
│
├── data/
│   ├── videos/
│   │   ├── raw/                   # Ham videolar
│   │   └── test/                  # 5 test videosu
│   ├── datasets/
│   │   └── vehicle_detection/     # Etiketli veri (Roboflow export)
│   └── ground_truth/              # Elle işaretlenmiş ihlaller (JSON)
│
├── weights/                       # Model ağırlıkları (.pt)
├── results/                       # Deney sonuçları (CSV, PNG)
└── requirements.txt
```

---

# 7. NEREDEN BAŞLIYORSUN?

## Sıra ve öncelik:

```
AŞAMA 1 — VERİ TOPLA (Bu hafta)
│
│  □ IBB kameralarından taralı alan bul
│  □ OBS ile video kaydet (5+ video)
│  □ YouTube'dan ek video indir
│  □ Videoları kırp → data/videos/test/
│  □ Pipeline'ı pretrained model ile test et
│  □ ROI polygon çiz (her video için)
│
▼
AŞAMA 2 — ALGORİTMANI YAZ (Hafta 2)
│
│  □ trajectory.py — yörünge takip modülü
│  □ severity.py — şiddet skorlama modülü
│  □ violation_detector.py'a entegre et
│  □ data_models.py'a yeni alanlar ekle
│  □ Pipeline'da test et, çıktıları kontrol et
│
▼
AŞAMA 3 — ETİKETLE + EĞİT (Hafta 3-4)
│
│  □ Videolardan frame çıkar
│  □ 300-500 görüntü etiketle (Roboflow)
│  □ Fine-tuning deneyi (Colab)
│  □ Pretrained vs fine-tuned karşılaştır
│
▼
AŞAMA 4 — DEĞERLENDİR (Hafta 5)
│
│  □ Ground truth oluştur (5 video, elle)
│  □ Precision / Recall / F1 hesapla
│  □ Koşul analizi (gündüz vs gece)
│  □ Şiddet skoru analizi
│
▼
AŞAMA 5 — TEZ YAZ (Hafta 6-7)
│
│  □ Yöntem bölümü (algoritmanı anlat)
│  □ Sonuçlar bölümü (tabloları koy)
│  □ Tartışma (limitasyonlar)
│  □ Giriş + literatür
│  □ Son kontrol + teslim
│
▼
TAMAMLANDI ✓
```

---

# 8. JÜRİYE NASIL SUNACAKSIN?

```
"İstanbul trafiğinde taralı alanlara girerek kaynak yapan
araçları otomatik tespit eden bir bilgisayarlı görü sistemi
geliştirdim.

Sistem YOLOv8 ile araç tespiti, ByteTrack ile takip yapıyor.
Benim katkım: tespit edilen ihlalleri yörünge analizi ile
sınıflandıran ve çok boyutlu şiddet skorlama formülü ile
değerlendiren algoritmayı tasarladım.

Bu sayede basit bir 'bölgede mi?' kontrolünden öteye geçerek
kaynak, seyir ve kenar teması ihlallerini ayırt edebilen
ve false positive oranını %X azaltan bir sistem elde ettim.

Sistem 5 farklı test videosunda ortalama %Y F1 skoru ile
çalışmaktadır."
```

---

*Son güncelleme: 12 Mart 2026*
