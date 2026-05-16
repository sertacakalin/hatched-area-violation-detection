#!/usr/bin/env python3
"""Kısa pipeline testi — frame sayısı belirleyerek çalıştır.

Kullanım:
    python scripts/run_short.py                  # varsayılan 300 frame (10 sn)
    python scripts/run_short.py --frames 900     # 30 saniyelik
    python scripts/run_short.py --frames 1800    # 1 dakikalık
    python scripts/run_short.py --start 3000     # 100. saniyeden başla
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

print("Pipeline başlatılıyor...", flush=True)

import cv2
import numpy as np
from ultralytics import YOLO
from src.zones.zone_manager import ZoneManager
from src.violation.violation_detector import ViolationDetector
from src.core.data_models import Detection, TrackedObject, VehicleState
from src.core.visualizer import Visualizer

COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main():
    parser = argparse.ArgumentParser(description="Kısa pipeline testi")
    parser.add_argument("--video", default="data/videos/test/test_01.mp4")
    parser.add_argument("--zone", default="configs/zones/e5_avcilar.json")
    parser.add_argument("--frames", type=int, default=300, help="İşlenecek frame sayısı")
    parser.add_argument("--start", type=int, default=0, help="Başlangıç frame numarası")
    parser.add_argument("--output", default="results/short_test_output.mp4")
    args = parser.parse_args()

    # Video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.0f} FPS, {total} frame ({total/fps:.0f} sn)", flush=True)

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
        print(f"Başlangıç: frame {args.start} ({args.start/fps:.0f}. saniye)", flush=True)

    # Model + Zone
    model = YOLO("yolov8s.pt")
    zone_mgr = ZoneManager(zone_file=args.zone, polygon_buffer=-10)
    viol_det = ViolationDetector(
        zone_manager=zone_mgr, min_frames_in_zone=5,
        cooldown_frames=90, min_overlap_ratio=0.3,
    )
    vis = Visualizer(zone_alpha=0.3, font_scale=0.6, thickness=2)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    print(f"\n{args.frames} frame işlenecek (~{args.frames/fps:.0f} sn video)...\n", flush=True)

    total_violations = 0
    frame_times = []

    for i in range(args.frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Video bitti (frame {i})", flush=True)
            break

        t0 = time.time()
        frame_num = args.start + i + 1

        # Track
        results = model.track(
            source=frame, conf=0.35, iou=0.45, classes=[2, 3, 5, 7],
            imgsz=640, half=False, tracker="bytetrack.yaml",
            persist=True, verbose=False,
        )

        # Parse
        tracked_objects = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            for bbox, tid, conf, cls_id in zip(boxes, track_ids, confs, cls_ids):
                class_name = COCO_VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
                det = Detection(bbox=bbox, confidence=float(conf),
                                class_id=int(cls_id), class_name=class_name)
                obj = TrackedObject(track_id=int(tid), detection=det)
                tracked_objects.append(obj)

        # Violation
        tracked_objects, new_violations = viol_det.process_frame(
            tracked_objects, frame, frame_num, fps
        )
        total_violations += len(new_violations)

        # Visualize
        display = frame.copy()
        for name, polygon in zone_mgr.get_zone_polygons_for_drawing():
            display = vis.draw_zone(display, polygon, label=name)
        for obj in tracked_objects:
            display = vis.draw_tracked_object(display, obj)
        for event in new_violations:
            display = vis.draw_violation_event(display, event)
        display = vis.draw_info_panel(display, frame_num, fps,
                                      total_violations, len(tracked_objects))
        out.write(display)

        t1 = time.time()
        frame_times.append(t1 - t0)

        # Her 50 frame'de rapor
        if (i + 1) % 50 == 0 or len(new_violations) > 0:
            avg = 1.0 / (sum(frame_times[-50:]) / len(frame_times[-50:]))
            status = f"Frame {i+1}/{args.frames} | FPS: {avg:.1f} | Takip: {len(tracked_objects)} | İhlal: {total_violations}"
            if new_violations:
                status += " *** YENİ İHLAL! ***"
            print(status, flush=True)

    cap.release()
    out.release()

    # Son frame kaydet
    if 'display' in dir():
        cv2.imwrite(args.output.replace(".mp4", "_last_frame.jpg"), display)

    avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
    duration = sum(frame_times)

    print(f"\n{'='*50}", flush=True)
    print(f"SONUÇLAR", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"İşlenen frame  : {len(frame_times)}", flush=True)
    print(f"Toplam süre    : {duration:.0f} saniye ({duration/60:.1f} dakika)", flush=True)
    print(f"Ortalama FPS   : {avg_fps:.2f}", flush=True)
    print(f"Tespit edilen  : {total_violations} ihlal", flush=True)
    print(f"Çıktı video    : {args.output}", flush=True)
    print(f"Son frame      : {args.output.replace('.mp4', '_last_frame.jpg')}", flush=True)
    print(f"\nVideoyu aç: open {args.output}", flush=True)


if __name__ == "__main__":
    main()
