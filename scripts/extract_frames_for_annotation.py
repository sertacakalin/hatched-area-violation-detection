"""Ground truth etiketleme için videolardan belirli aralıklarla frame çıkar.

Kullanım:
    python scripts/extract_frames_for_annotation.py \
        --video data/videos/test/test_01.mp4 \
        --interval 30 \
        --output data/ground_truth/frames_test_01

Her frame'in üzerine kare numarası ve zaman damgası yazılır.
Zone overlay'i de eklenir (polygon görünsün diye).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.zones.zone_manager import ZoneManager


def main():
    parser = argparse.ArgumentParser(description="Etiketleme için frame çıkar")
    parser.add_argument("--video", required=True)
    parser.add_argument("--interval", type=int, default=30,
                        help="Kaç karede bir çıkar (varsayılan: 30 = her 1 sn)")
    parser.add_argument("--zone", default="configs/zones/e5_avcilar.json",
                        help="Zone dosyası (overlay için)")
    parser.add_argument("--output", default=None,
                        help="Çıktı dizini (varsayılan: data/ground_truth/frames_<video>)")
    parser.add_argument("--start", type=int, default=0, help="Başlangıç karesi")
    parser.add_argument("--end", type=int, default=-1, help="Bitiş karesi (-1 = son)")
    args = parser.parse_args()

    video_name = Path(args.video).stem
    output_dir = Path(args.output or f"data/ground_truth/frames_{video_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Zone yükle
    zm = ZoneManager(zone_file=args.zone)
    zone_polygons = zm.get_zone_polygons_for_drawing()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = args.end if args.end > 0 else total

    print(f"Video: {args.video}")
    print(f"FPS: {fps:.1f}, Toplam: {total} kare")
    print(f"Aralık: her {args.interval} kare ({args.interval/fps:.1f} sn)")
    print(f"Çıktı: {output_dir}")

    count = 0
    frame_num = 0

    while frame_num < end:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num >= args.start and frame_num % args.interval == 0:
            # Zone overlay
            for name, polygon in zone_polygons:
                overlay = frame.copy()
                pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(overlay, [pts], (255, 165, 0))
                cv2.polylines(frame, [pts], True, (255, 165, 0), 2)
                frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

            # Bilgi yaz
            timestamp = frame_num / fps
            info = f"Kare: {frame_num} | Zaman: {timestamp:.1f}s"
            cv2.putText(frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Kaydet
            filename = f"frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(output_dir / filename), frame)
            count += 1

        frame_num += 1

    cap.release()
    print(f"\n{count} frame kaydedildi: {output_dir}")
    print(f"\nBu frameleri inceleyerek ground truth JSON'u doldur:")
    print(f"  data/ground_truth/{video_name}.json")


if __name__ == "__main__":
    main()
