"""Tek komutla demo — video ver, gerisini sistem halleder.

Kullanım:
    python scripts/demo.py video.mp4
    python scripts/demo.py video.mp4 --auto          # Taralı alanı otomatik bul
    python scripts/demo.py video.mp4 --draw           # Mouse ile çiz (varsayılan)
    python scripts/demo.py video.mp4 --zone zone.json  # Hazır polygon kullan

Akış:
    1. Zone yok → mouse ile çiz VEYA otomatik tespit
    2. Pipeline çalıştır
    3. Çıktı videosu + ihlal raporu oluştur
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np


def draw_zone_interactive(video_path: str) -> list[list[int]]:
    """Mouse ile taralı alanı çiz. Sol tık = nokta, sağ tık = geri, q = tamam."""
    from src.zones.roi_selector import ROISelector
    print("\n" + "=" * 50)
    print("TARALI ALANI ÇİZ")
    print("=" * 50)
    print("  Sol tık  → köşe noktası ekle")
    print("  Sağ tık  → son noktayı sil")
    print("  R tuşu   → sıfırla")
    print("  Q tuşu   → tamamla")
    print("=" * 50 + "\n")

    selector = ROISelector()
    polygon = selector.select_from_video(video_path, frame_number=90)
    return polygon


def detect_zone_auto(video_path: str) -> list[list[int]] | None:
    """Otomatik taralı alan tespiti."""
    from src.zones.hatched_detector import HatchedAreaDetector
    print("Taralı alan otomatik tespit ediliyor...")
    detector = HatchedAreaDetector()
    result = detector.detect(video_path)

    if result["polygon"] and result["confidence"] > 0.2:
        print(f"  Taralı alan bulundu (güven: {result['confidence']:.2f})")
        return result["polygon"]
    else:
        print("  Otomatik tespit başarısız — mouse ile çizime geçiliyor...")
        return None


def run_demo(video_path: str, zone_polygon: list[list[int]],
             output_dir: str) -> dict:
    """Pipeline'ı çalıştır ve sonuçları döndür."""
    from src.core.config import Config
    from src.pipeline.pipeline import Pipeline
    from src.zones.zone_manager import ZoneManager

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Geçici zone JSON oluştur
    zone_path = str(out / "_zone_temp.json")
    cap_tmp = cv2.VideoCapture(video_path)
    w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_tmp.release()

    zone_data = {
        "camera_id": "demo",
        "frame_width": w,
        "frame_height": h,
        "zones": [{
            "zone_id": "zone_01",
            "name": "Tarali Alan",
            "polygon": zone_polygon,
            "type": "hatched_area",
        }],
    }
    with open(zone_path, "w") as f:
        json.dump(zone_data, f, indent=2)

    # Config
    config = Config("configs/config.yaml")
    config._data["general"]["video_source"] = video_path
    config._data["general"]["output_dir"] = output_dir
    config._data["general"]["save_video"] = True
    config._data["general"]["show_display"] = False
    config._data["zone"]["zone_file"] = zone_path
    config._data["vehicle_detection"]["half_precision"] = False

    # Pipeline
    pipeline = Pipeline(config)
    print(f"\nPipeline çalışıyor: {video_path}")
    print("  (CPU'da yavaş olabilir, bekleyin...)\n")

    t_start = time.time()
    stats = pipeline.run()
    elapsed = time.time() - t_start

    stats["elapsed_sec"] = round(elapsed, 1)
    return stats


def print_report(stats: dict, output_dir: str):
    """Sonuç raporu yazdır."""
    print("\n" + "=" * 50)
    print("SONUÇLAR")
    print("=" * 50)
    print(f"  Toplam ihlal    : {stats.get('total_violations', 0)}")
    print(f"  İşlenen kare    : {stats.get('total_frames_processed', 0)}")
    print(f"  Ortalama FPS    : {stats.get('average_fps', 0):.1f}")
    print(f"  Süre            : {stats.get('elapsed_sec', 0):.1f} saniye")

    class_dist = stats.get("class_distribution", {})
    if class_dist:
        print(f"  Araç dağılımı   : {class_dist}")

    print(f"\n  Çıktı videosu   : {output_dir}/output.mp4")
    print(f"  Veritabanı      : {output_dir}/violations.db")
    print(f"  Araç kırpmaları : {output_dir}/crops/")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Taralı alan ihlal tespiti — tek komutla demo",
        usage="python scripts/demo.py VIDEO [--auto | --draw | --zone FILE]",
    )
    parser.add_argument("video", help="Video dosya yolu")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--auto", action="store_true",
                       help="Taralı alanı otomatik tespit et")
    group.add_argument("--draw", action="store_true",
                       help="Mouse ile taralı alanı çiz (varsayılan)")
    group.add_argument("--zone", help="Hazır zone JSON dosyası")
    parser.add_argument("--output", default=None,
                        help="Çıktı dizini (varsayılan: results/demo_<video>)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Video kontrol
    if not Path(args.video).exists():
        print(f"HATA: Video bulunamadı: {args.video}")
        sys.exit(1)

    video_name = Path(args.video).stem
    output_dir = args.output or f"results/demo_{video_name}"

    # Zone polygon al
    polygon = None

    if args.zone:
        # Hazır JSON
        with open(args.zone) as f:
            data = json.load(f)
        polygon = data["zones"][0]["polygon"]
        print(f"Zone yüklendi: {args.zone}")

    elif args.auto:
        # Otomatik tespit, başarısız olursa mouse'a düş
        polygon = detect_zone_auto(args.video)
        if polygon is None:
            polygon = draw_zone_interactive(args.video)

    else:
        # Varsayılan: mouse ile çiz
        polygon = draw_zone_interactive(args.video)

    if not polygon or len(polygon) < 3:
        print("HATA: Geçerli bir polygon oluşturulamadı.")
        sys.exit(1)

    print(f"Taralı alan: {len(polygon)} köşe noktası")

    # Pipeline çalıştır
    stats = run_demo(args.video, polygon, output_dir)

    # Rapor
    print_report(stats, output_dir)


if __name__ == "__main__":
    main()
