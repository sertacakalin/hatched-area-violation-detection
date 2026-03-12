"""İnteraktif ROI seçim script'i."""

import argparse
import sys
from pathlib import Path

# Proje kökünü path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.zones.roi_selector import ROISelector


def main():
    parser = argparse.ArgumentParser(description="İnteraktif ROI polygon seçimi")
    parser.add_argument("--video", required=True, help="Video dosya yolu")
    parser.add_argument("--frame", type=int, default=0, help="Başlangıç kare numarası")
    parser.add_argument("--output", default="configs/zones/camera_01.json",
                        help="Çıktı JSON dosya yolu")
    parser.add_argument("--zone-id", default="zone_01", help="Bölge ID")
    parser.add_argument("--name", default="Taralı Alan", help="Bölge adı")
    args = parser.parse_args()

    selector = ROISelector()

    print("\n=== ROI Seçim Aracı ===")
    print("Sol tıklama : Nokta ekle")
    print("Sağ tıklama : Son noktayı geri al")
    print("'r' tuşu    : Tüm noktaları sıfırla")
    print("'q' tuşu    : Seçimi tamamla")
    print()

    polygon = selector.select_from_video(args.video, args.frame)

    if polygon:
        selector.save_zone(
            args.output, polygon,
            zone_id=args.zone_id,
            name=args.name,
        )
        print(f"\nBölge kaydedildi: {args.output}")
        print(f"Polygon noktaları: {polygon}")
    else:
        print("\nBölge seçilmedi (minimum 3 nokta gerekli).")


if __name__ == "__main__":
    main()
