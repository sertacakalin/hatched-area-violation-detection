"""
Auto-label frames using existing best.pt model.
Outputs YOLO format labels directly.

Usage:
    python scripts/auto_label_with_model.py \
        --frames-dir data/datasets/frames_for_labeling/cctv_v3/file3_day \
        --output-dir data/datasets/auto_labeled/file3_day \
        --weights weights/best.pt \
        --conf 0.35
"""
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def auto_label(frames_dir: Path, output_dir: Path, weights: Path, conf: float = 0.35):
    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    print(f"Model loaded: {weights}")
    print(f"Class names: {model.names}")

    frame_files = sorted(frames_dir.glob("*.jpg"))
    print(f"Total frames: {len(frame_files)}")

    labeled = 0
    no_detection = 0

    for i, frame_path in enumerate(frame_files):
        if i % 100 == 0:
            print(f"  [{i}/{len(frame_files)}] {frame_path.name}")

        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        results = model.predict(img, conf=conf, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            no_detection += 1
            continue

        label_path = output_labels / f"{frame_path.stem}.txt"
        with open(label_path, "w") as f:
            for box in boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # YOLO format: class cx cy w h (normalized)
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        # Copy image to output
        import shutil
        shutil.copy(frame_path, output_images / frame_path.name)
        labeled += 1

    print(f"\nDone:")
    print(f"  Labeled: {labeled}")
    print(f"  No detection (skipped): {no_detection}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    p.add_argument("--conf", type=float, default=0.35)
    args = p.parse_args()

    auto_label(args.frames_dir, args.output_dir, args.weights, args.conf)
