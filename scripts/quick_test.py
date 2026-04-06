#!/usr/bin/env python3
"""Hızlı araç tespiti testi — tek frame üzerinde."""

import sys
import time

sys.path.insert(0, ".")

print("Step 1: Importing...", flush=True)
import cv2
from ultralytics import YOLO

print("Step 2: Opening video...", flush=True)
cap = cv2.VideoCapture("data/videos/test/test_01.mp4")
ret, frame = cap.read()
if not ret:
    print("ERROR: Video okunamadı!")
    sys.exit(1)
print(f"Frame shape: {frame.shape}", flush=True)

print("Step 3: Loading YOLOv8s...", flush=True)
model = YOLO("yolov8s.pt")

print("Step 4: Running detection on first frame...", flush=True)
t0 = time.time()
results = model.predict(
    source=frame,
    conf=0.35,
    iou=0.45,
    classes=[2, 3, 5, 7],
    imgsz=640,
    half=False,
    verbose=False,
)
t1 = time.time()
print(f"Detection time: {t1 - t0:.3f}s", flush=True)

if results and len(results) > 0:
    result = results[0]
    if result.boxes is not None and len(result.boxes) > 0:
        print(f"Detected {len(result.boxes)} vehicles:", flush=True)
        cls_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        for i, box in enumerate(result.boxes[:15]):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = cls_names.get(cls_id, f"cls_{cls_id}")
            print(f"  [{i + 1}] {name}: {conf:.2f}", flush=True)
    else:
        print("No vehicles detected", flush=True)

# Annotated frame kaydet
print("Step 5: Saving annotated frame...", flush=True)
annotated = results[0].plot()
cv2.imwrite("results/detection_test_frame.jpg", annotated)
print("Saved to results/detection_test_frame.jpg", flush=True)

cap.release()
print("\nDONE!", flush=True)
