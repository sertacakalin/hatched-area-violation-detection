"""Live stream + zone selection.

İki aşama:
1. Kameradan tek frame yakala, üzerinde polygon çiz (sol tık = nokta, q = bitir)
2. Live stream başlat — taralı alan kontrolü ile

Kullanım:
    python scripts/live_with_zone.py
    python scripts/live_with_zone.py --source 1   # USB kamera
"""
import argparse
import json
import platform
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


class PolygonClicker:
    def __init__(self, image):
        self.image = image.copy()
        self.original = image.copy()
        self.points = []

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()
            self.redraw()

    def redraw(self):
        self.image = self.original.copy()
        # Çizgileri çiz
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(self.image, self.points[i], self.points[i+1],
                         (0, 165, 255), 3)
        if len(self.points) >= 3:
            # Polygon kapatma çizgisi (kesikli için fonksiyon yok, normal)
            cv2.line(self.image, self.points[-1], self.points[0],
                     (0, 165, 255), 1)
            # Polygon dolgu
            pts = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
            overlay = self.image.copy()
            cv2.fillPoly(overlay, [pts], (0, 165, 255))
            self.image = cv2.addWeighted(overlay, 0.3, self.image, 0.7, 0)
        # Noktaları çiz
        for i, p in enumerate(self.points):
            cv2.circle(self.image, p, 6, (0, 255, 255), -1)
            cv2.putText(self.image, str(i+1), (p[0]+8, p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # HUD
        cv2.putText(self.image,
                    f"Sol tik=nokta ekle, Sag tik=geri al, q=bitir, r=sifirla. ({len(self.points)} nokta)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Tarali Alan Cizimi", self.image)


def select_zone_interactive(cap):
    """Kameradan frame yakala, kullanıcı polygon çizsin."""
    print("\n🎯 ADIM 1: Taralı alan polygon çizimi")
    print("   - Sol tık: nokta ekle")
    print("   - Sağ tık: son noktayı geri al")
    print("   - r: sıfırla")
    print("   - q veya Enter: bitir (min 3 nokta)")
    print()

    # Uzun warmup — sensör ışığa alışsın, AE/AWB stabilize olsun
    print("📸 Kamera stabilize ediliyor...")
    last_frame = None
    for i in range(60):  # ~3 saniye
        ok, frame = cap.read()
        if ok:
            last_frame = frame
            mean_brightness = frame.mean()
            if i % 10 == 0:
                print(f"   Frame {i+1}: parlaklık={mean_brightness:.0f}")
        time.sleep(0.05)

    if last_frame is None:
        raise RuntimeError("Kameradan frame alınamadı")

    if last_frame.mean() < 20:
        print("⚠️  Görüntü çok karanlık. Lens kapalı olabilir veya odaya ışık az.")
        print("   Ama yine de devam ediliyor...")

    frame = last_frame

    clicker = PolygonClicker(frame)
    cv2.namedWindow("Tarali Alan Cizimi")
    cv2.setMouseCallback("Tarali Alan Cizimi", clicker.callback)
    clicker.redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 13:  # 13 = Enter
            if len(clicker.points) >= 3:
                break
            else:
                print("⚠️  En az 3 nokta gerekli")
        elif key == ord('r'):
            clicker.points = []
            clicker.redraw()

    cv2.destroyWindow("Tarali Alan Cizimi")
    return clicker.points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--weights", default="weights/best_v3.pt")
    ap.add_argument("--conf", type=float, default=0.55)
    args = ap.parse_args()

    # --- Kamera aç ---
    src = int(args.source) if args.source.isdigit() else args.source
    if platform.system() == "Darwin" and isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Stream açılamadı: {src}")

    print("🔥 Kamera ısıtılıyor...")
    for _ in range(10):
        cap.read()
        time.sleep(0.1)

    # --- Zone seç ---
    points = select_zone_interactive(cap)
    print(f"✅ {len(points)} noktalı polygon seçildi")

    # Geçici zone dosyası yaz
    tmp_zone = Path("/tmp/live_zone.json")
    tmp_zone.write_text(json.dumps({
        "zones": [{
            "zone_id": "live_zone",
            "name": "Live Tarali Alan",
            "polygon": [[int(x), int(y)] for x, y in points],
        }]
    }))

    # --- Setup ---
    print(f"🤖 Model yükleniyor: {args.weights}")
    tracker = ByteTrackWrapper(
        model_path=args.weights,
        conf=args.conf,
        classes=[0, 1, 2, 3],
        half=False,
        max_bbox_ratio=0.25,
    )
    zone_manager = ZoneManager(zone_file=str(tmp_zone), polygon_buffer=-10)
    violation_detector = ViolationDetector(
        zone_manager=zone_manager,
        min_frames_in_zone=5,
        cooldown_frames=600,
        per_track_lock=True,
    )

    # --- Live loop ---
    fps_buffer = deque(maxlen=30)
    frame_count = 0
    violation_count = 0

    print("\n▶️  ADIM 2: LIVE inference başladı (q = çıkış)\n")

    while True:
        loop_start = time.time()
        ok, frame = cap.read()
        if not ok:
            break

        # Inference
        tracked = tracker.update(None, frame)
        tracked, new_violations = violation_detector.process_frame(
            tracked, frame, frame_count, 30.0
        )
        violation_count += len(new_violations)

        # Zone overlay
        for zone_name, coords in zone_manager.get_zone_polygons_for_drawing():
            pts = coords.reshape((-1, 1, 2))
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 165, 255))
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
            cv2.polylines(frame, [pts], True, (0, 165, 255), 2)

        # Bbox annotate
        for obj in tracked:
            x1, y1, x2, y2 = map(int, obj.bbox)
            color = (0, 0, 255) if obj.is_violation else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{obj.track_id} {obj.detection.class_name}"
            cv2.putText(frame, label, (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Yeni ihlal logu
        for v in new_violations:
            print(f"⚠️  [{frame_count:5d}] İHLAL: track #{v.track_id} "
                  f"{v.vehicle_class} → {v.violation_type} ({v.severity_level})")

        # FPS + HUD
        loop_time = time.time() - loop_start
        fps_buffer.append(1.0 / max(loop_time, 0.001))
        avg_fps = sum(fps_buffer) / len(fps_buffer)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Tracks: {len(tracked)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Violations: {violation_count}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "LIVE", (frame.shape[1]-100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("LIVE - Hatched Area Violation Detection", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Durdu. {frame_count} frame, {violation_count} ihlal.")


if __name__ == "__main__":
    main()
