#!/usr/bin/env python3
"""Mini pipeline testi — 20 frame ile uçtan uca doğrulama."""

import sys
import time
import json

sys.path.insert(0, ".")

print("=" * 50, flush=True)
print("MINI PIPELINE TEST (20 frame)", flush=True)
print("=" * 50, flush=True)

print("\n[1/6] Importing modules...", flush=True)
import cv2
import numpy as np
from ultralytics import YOLO
from src.zones.zone_manager import ZoneManager
from src.violation.violation_detector import ViolationDetector
from src.core.data_models import Detection, TrackedObject, VehicleState
from src.core.visualizer import Visualizer

print("[2/6] Loading video + model + zones...", flush=True)
cap = cv2.VideoCapture("data/videos/test/test_01.mp4")
ret, _ = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"   Video: {w}x{h} @ {fps:.0f} FPS", flush=True)

model = YOLO("yolov8s.pt")
print("   YOLOv8s loaded", flush=True)

zone_mgr = ZoneManager(
    zone_file="configs/zones/e5_avcilar.json",
    polygon_buffer=-10,
)
print(f"   {len(zone_mgr.zones)} zone(s) loaded", flush=True)

viol_det = ViolationDetector(
    zone_manager=zone_mgr,
    min_frames_in_zone=5,
    cooldown_frames=90,
    min_overlap_ratio=0.3,
)

vis = Visualizer(zone_alpha=0.3, font_scale=0.6, thickness=2)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("results/mini_test_output.mp4", fourcc, fps, (w, h))

print("[3/6] Processing 20 frames with tracking...", flush=True)
total_violations = 0
frame_times = []

COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

for frame_idx in range(20):
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()

    # Track
    results = model.track(
        source=frame,
        conf=0.35,
        iou=0.45,
        classes=[2, 3, 5, 7],
        imgsz=640,
        half=False,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    )

    # Parse tracking results
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

    # Violation detection
    tracked_objects, new_violations = viol_det.process_frame(
        tracked_objects, frame, frame_idx + 1, fps
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
    display = vis.draw_info_panel(display, frame_idx + 1, fps,
                                  total_violations, len(tracked_objects))

    out.write(display)

    t1 = time.time()
    frame_times.append(t1 - t0)

    # Sadece birkaç frame'in detayını göster
    n_tracked = len(tracked_objects)
    n_in_zone = sum(1 for o in tracked_objects if o.state in (VehicleState.ENTERING, VehicleState.INSIDE, VehicleState.VIOLATION))
    print(f"   Frame {frame_idx+1:3d}: {n_tracked:2d} tracked, "
          f"{n_in_zone} in zone, {len(new_violations)} violations "
          f"({t1-t0:.1f}s)", flush=True)

cap.release()
out.release()

# Son frame'i kaydet
if frame is not None:
    cv2.imwrite("results/mini_test_last_frame.jpg", display)

avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

print(f"\n[4/6] Results:", flush=True)
print(f"   Frames processed: {len(frame_times)}", flush=True)
print(f"   Average FPS: {avg_fps:.2f}", flush=True)
print(f"   Total violations: {total_violations}", flush=True)
print(f"   Output: results/mini_test_output.mp4", flush=True)

print("\n[5/6] Saving annotated last frame...", flush=True)
print("   Saved: results/mini_test_last_frame.jpg", flush=True)

print("\n[6/6] PIPELINE TEST COMPLETE!", flush=True)
print("=" * 50, flush=True)
