"""Live stream pipeline — gerçek zamanlı CCTV/mobese tespit.

Kullanım:
    # Webcam'den canlı (Mac iç kamerası)
    python scripts/live_stream.py --source 0

    # RTSP stream (gerçek mobese feed)
    python scripts/live_stream.py --source rtsp://kamera-ip/live

    # Kayıt video'yu "canlı gibi" simüle et (demo)
    python scripts/live_stream.py --source data/videos/test/cam4_30s.mp4 --simulate-live

    # Zone ile birlikte (ihlal tespiti)
    python scripts/live_stream.py --source 0 --zone configs/zones/cam4_30s.json
"""
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tracking.bytetrack_wrapper import ByteTrackWrapper
from src.zones.zone_manager import ZoneManager
from src.violation.violation_detector import ViolationDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="0 = webcam, rtsp://... = stream, dosya yolu = video")
    ap.add_argument("--weights", default="weights/best_v3.pt")
    ap.add_argument("--zone", default=None,
                    help="Zone JSON path (opsiyonel)")
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--simulate-live", action="store_true",
                    help="Video dosyasını gerçek zaman hızında oku")
    ap.add_argument("--display-size", type=int, default=1280,
                    help="Görüntü genişliği (yükseklik korunur)")
    args = ap.parse_args()

    # Source: 0 webcam, sayı index, yoksa string (rtsp veya path)
    src = int(args.source) if args.source.isdigit() else args.source
    print(f"📹 Source: {src}")

    # Mac için AVFOUNDATION backend (Continuity Camera fix)
    import platform
    if platform.system() == "Darwin" and isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(f"Stream açılamadı: {src}")

    # Webcam warmup — ilk frame'ler genelde boş gelir
    if isinstance(src, int):
        print("🔥 Kamera ısıtılıyor...")
        for _ in range(10):
            ok, _ = cap.read()
            if ok:
                break
            time.sleep(0.1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_delay = 1.0 / fps if args.simulate_live else 0.0
    print(f"📊 FPS: {fps:.1f}, target_delay: {target_delay*1000:.1f}ms")

    # Model
    print(f"🤖 Model: {args.weights}")
    tracker = ByteTrackWrapper(
        model_path=args.weights,
        conf=args.conf,
        classes=[0, 1, 2, 3],  # bus, car, motorcycle, truck
        half=False,
        max_bbox_ratio=0.25,
    )

    # Zone (opsiyonel)
    zone_manager = None
    violation_detector = None
    if args.zone and Path(args.zone).exists():
        print(f"🟧 Zone: {args.zone}")
        zone_manager = ZoneManager(zone_file=args.zone, polygon_buffer=-10)
        violation_detector = ViolationDetector(
            zone_manager=zone_manager,
            min_frames_in_zone=5,
            cooldown_frames=600,
            per_track_lock=True,
        )

    # FPS counter
    fps_buffer = deque(maxlen=30)
    frame_count = 0
    violation_count = 0

    print("\n▶️  Başlıyor... (q = çıkış)")
    print("=" * 60)

    while True:
        loop_start = time.time()
        ok, frame = cap.read()
        if not ok:
            print("Stream sonu.")
            break

        # Resize for display (FPS için optimum)
        h, w = frame.shape[:2]
        if w > args.display_size:
            scale = args.display_size / w
            frame = cv2.resize(frame, (args.display_size, int(h * scale)))

        # Inference
        tracked = tracker.update(None, frame)

        # Zone check + violation
        new_violations = []
        if violation_detector:
            tracked, new_violations = violation_detector.process_frame(
                tracked, frame, frame_count, fps
            )
            violation_count += len(new_violations)

        # Annotate
        for obj in tracked:
            x1, y1, x2, y2 = map(int, obj.bbox)
            color = (0, 0, 255) if obj.is_violation else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{obj.track_id} {obj.detection.class_name} {obj.detection.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Zone overlay
        if zone_manager:
            polygons = zone_manager.get_zone_polygons_for_drawing()
            for poly in polygons:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 165, 255))
                frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
                cv2.polylines(frame, [pts], True, (0, 165, 255), 2)

        # FPS hesapla
        loop_time = time.time() - loop_start
        fps_buffer.append(1.0 / max(loop_time, 0.001))
        avg_fps = sum(fps_buffer) / len(fps_buffer)

        # HUD
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Tracks: {len(tracked)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Violations: {violation_count}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"LIVE", (frame.shape[1]-100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Yeni ihlali konsola yazdır
        for v in new_violations:
            print(f"⚠️  [{frame_count:5d}] İHLAL: track #{v.track_id} "
                  f"{v.vehicle_class} → {v.violation_type} ({v.severity_level})")

        # Show
        cv2.imshow("LIVE - Hatched Area Violation Detection", frame)

        # Sleep for live simulation
        if target_delay > 0:
            elapsed = time.time() - loop_start
            sleep_time = target_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("=" * 60)
    print(f"✓ Durdu. Toplam {frame_count} frame, {violation_count} ihlal.")


if __name__ == "__main__":
    main()
