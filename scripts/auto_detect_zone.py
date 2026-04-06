#!/usr/bin/env python3
"""Otomatik taralı alan tespiti — test ve görselleştirme.

Kullanım:
    python scripts/auto_detect_zone.py
    python scripts/auto_detect_zone.py --video data/videos/test/test_01.mp4
    python scripts/auto_detect_zone.py --samples 300 --eps 40
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

print("Otomatik Taralı Alan Tespiti Başlatılıyor...\n", flush=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from src.zones.hatched_detector import HatchedAreaDetector, calculate_iou


def main():
    parser = argparse.ArgumentParser(
        description="Otomatik taralı alan tespiti"
    )
    parser.add_argument(
        "--video",
        default="data/videos/test/test_01.mp4",
        help="Video dosya yolu",
    )
    parser.add_argument(
        "--output-zone",
        default="configs/zones/auto_detected.json",
        help="Çıktı zone JSON dosyası",
    )
    parser.add_argument(
        "--output-debug",
        default="results/auto_detect_debug",
        help="Debug görselleri dizini",
    )
    parser.add_argument(
        "--compare-with",
        help="Manuel zone JSON ile karşılaştır (IoU hesapla)",
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--eps", type=float, default=50.0)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--min-angle", type=float, default=25.0)
    parser.add_argument("--max-angle", type=float, default=65.0)
    parser.add_argument("--hough-threshold", type=int, default=60)
    parser.add_argument("--min-line-length", type=int, default=25)
    args = parser.parse_args()

    t_start = time.time()

    # Detector oluştur
    detector = HatchedAreaDetector(
        sample_count=args.samples,
        sample_interval=args.interval,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        hough_threshold=args.hough_threshold,
        min_line_length=args.min_line_length,
    )

    # Tespit et
    result = detector.detect(args.video)
    t_elapsed = time.time() - t_start

    # Sonuçları yazdır
    print(f"\n{'='*50}", flush=True)
    print("SONUÇLAR", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"Süre           : {t_elapsed:.1f} saniye", flush=True)
    print(f"Toplam çizgi   : {result['debug']['all_lines_count']}", flush=True)
    print(f"Çapraz çizgi   : {len(result['debug']['diagonal_lines'])}", flush=True)
    print(f"Küme sayısı    : {len(result['debug']['clusters'])}", flush=True)
    print(f"Polygon sayısı : {len(result['all_polygons'])}", flush=True)
    print(f"Güven skoru    : {result['confidence']:.2f}", flush=True)

    if result["polygon"]:
        print(f"Polygon        : {result['polygon']}", flush=True)

        # Zone JSON kaydet
        detector.save_zone_json(
            result["polygon"],
            args.output_zone,
        )
        print(f"Zone dosyası   : {args.output_zone}", flush=True)
    else:
        print("UYARI: Taralı alan tespit edilemedi!", flush=True)

    # Debug görselleri kaydet
    saved = detector.save_debug_images(result, args.output_debug)
    print(f"\nDebug görselleri:", flush=True)
    for name, path in saved.items():
        print(f"  {name:20s} → {path}", flush=True)

    # Manuel ile karşılaştırma
    if args.compare_with and result["polygon"]:
        import json
        with open(args.compare_with) as f:
            manual = json.load(f)
        manual_polygon = manual["zones"][0]["polygon"]
        iou = calculate_iou(result["polygon"], manual_polygon)
        print(f"\nManuel ile IoU : {iou:.3f} ({iou*100:.1f}%)", flush=True)

    print(f"\nGörselleri aç: open {args.output_debug}/", flush=True)


if __name__ == "__main__":
    main()
