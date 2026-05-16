"""
Generate thesis-quality visualizations for the dataset.

Outputs (in docs/figures/dataset_v3/):
1. class_distribution.png       — Bar chart: bbox count per class
2. source_distribution.png      — Bar chart: image count per source
3. bbox_heatmap.png             — Spatial heatmap of bbox centers
4. bbox_size_distribution.png   — Histogram of bbox sizes
5. instances_per_image.png      — Distribution of detection density
6. sample_collage.png           — Example frames from each source
7. dataset_summary.md           — Markdown report with all numbers

Usage:
    python scripts/generate_dataset_visualizations.py
"""
import json
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROBOFLOW = Path("/Users/sertacakalin/Desktop/istanbul-traffic-vehicles")
AUTO = PROJECT_ROOT / "data/datasets/auto_labeled"
OUT = PROJECT_ROOT / "docs/figures/dataset_v3"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["bus", "car", "motorcycle", "truck"]
COLORS = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]  # bus, car, mc, truck


def collect_data():
    """Returns: dict of {source_name: [(image_path, label_path), ...]}"""
    sources = defaultdict(list)

    # Roboflow
    rf_imgs = ROBOFLOW / "train/images"
    rf_lbls = ROBOFLOW / "train/labels"
    for img in sorted(rf_imgs.glob("*.jpg")):
        lbl = rf_lbls / f"{img.stem}.txt"
        if lbl.exists():
            sources["Roboflow (HD)"].append((img, lbl))

    # Auto-labeled CCTV
    name_map = {
        "file1_night": "CCTV Night",
        "file3_day": "CCTV Day",
        "file4_evening": "CCTV Evening",
    }
    for folder, label in name_map.items():
        d = AUTO / folder
        if not d.exists():
            continue
        for img in sorted((d / "images").glob("*.jpg")):
            lbl = d / "labels" / f"{img.stem}.txt"
            if lbl.exists():
                sources[label].append((img, lbl))

    return sources


def parse_yolo_label(label_path):
    """Returns list of (class_id, cx, cy, w, h) tuples."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((cls, cx, cy, w, h))
    return boxes


def fig1_class_distribution(sources):
    """Bar chart: total bboxes per class, stacked by source."""
    counts = defaultdict(lambda: defaultdict(int))  # source -> class_id -> count
    for source, pairs in sources.items():
        for _, lbl in pairs:
            for cls, *_ in parse_yolo_label(lbl):
                counts[source][cls] += 1

    source_names = list(sources.keys())
    bottoms = np.zeros(len(CLASSES))
    fig, ax = plt.subplots(figsize=(10, 6))

    for source in source_names:
        values = [counts[source][i] for i in range(len(CLASSES))]
        ax.bar(CLASSES, values, bottom=bottoms, label=source, alpha=0.85)
        bottoms += np.array(values)

    ax.set_ylabel("Bounding Box Count", fontsize=12)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_title("Class Distribution (Stacked by Source)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Total counts on top of bars
    for i, cls in enumerate(CLASSES):
        total = int(bottoms[i])
        ax.text(i, total + max(bottoms) * 0.01, f"{total:,}",
                ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT / "class_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[1/6] {OUT}/class_distribution.png")
    return dict(counts)


def fig2_source_distribution(sources):
    """Bar chart: image count per source."""
    fig, ax = plt.subplots(figsize=(10, 5))
    source_names = list(sources.keys())
    image_counts = [len(sources[s]) for s in source_names]
    bbox_counts = []
    for s in source_names:
        total = sum(len(parse_yolo_label(lbl)) for _, lbl in sources[s])
        bbox_counts.append(total)

    x = np.arange(len(source_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, image_counts, width, label="Images",
                    color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width/2, bbox_counts, width, label="Bounding Boxes",
                    color="#e67e22", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h, f"{int(h):,}",
                    ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(source_names, rotation=15, ha="right")
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Dataset Composition by Source", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "source_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[2/6] {OUT}/source_distribution.png")


def fig3_bbox_heatmap(sources):
    """Spatial heatmap of bbox centers (where in frame are objects)."""
    grid = np.zeros((100, 100))
    for _, pairs in sources.items():
        for _, lbl in pairs:
            for _, cx, cy, *_ in parse_yolo_label(lbl):
                gx = min(99, int(cx * 100))
                gy = min(99, int(cy * 100))
                grid[gy, gx] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    # Log scale for better viz
    grid_log = np.log1p(grid)
    im = ax.imshow(grid_log, cmap="hot", origin="upper", extent=[0, 1, 1, 0])
    ax.set_title("Bounding Box Center Density (Spatial Heatmap)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Image X (normalized)", fontsize=12)
    ax.set_ylabel("Image Y (normalized)", fontsize=12)
    plt.colorbar(im, ax=ax, label="log(1 + count)")
    plt.tight_layout()
    plt.savefig(OUT / "bbox_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[3/6] {OUT}/bbox_heatmap.png")


def fig4_bbox_size_distribution(sources):
    """Histogram of bbox sizes (width × height as fraction of image)."""
    sizes = defaultdict(list)
    for _, pairs in sources.items():
        for _, lbl in pairs:
            for cls, _, _, w, h in parse_yolo_label(lbl):
                area = w * h
                sizes[CLASSES[cls]].append(area)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cls in enumerate(CLASSES):
        if sizes[cls]:
            ax.hist(sizes[cls], bins=50, alpha=0.6, label=cls,
                    color=COLORS[i], range=(0, 0.3))
    ax.set_xlabel("Relative Bbox Area (width × height)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Bounding Box Size Distribution by Class",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "bbox_size_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[4/6] {OUT}/bbox_size_distribution.png")


def fig5_instances_per_image(sources):
    """Histogram of detections per image."""
    counts = []
    for _, pairs in sources.items():
        for _, lbl in pairs:
            counts.append(len(parse_yolo_label(lbl)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(counts, bins=50, color="#9b59b6", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Detections per Image", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title(f"Detection Density (mean = {np.mean(counts):.1f}, median = {np.median(counts):.0f})",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.axvline(np.mean(counts), color="red", linestyle="--", label=f"Mean: {np.mean(counts):.1f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "instances_per_image.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[5/6] {OUT}/instances_per_image.png")


def fig6_sample_collage(sources):
    """4-source collage: one example frame per source with bboxes drawn."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    source_list = list(sources.items())[:4]

    for ax, (source, pairs) in zip(axes, source_list):
        # Random sample
        np.random.seed(42)
        img_path, lbl_path = pairs[np.random.randint(len(pairs))]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        ax.imshow(img)
        for cls, cx, cy, bw, bh in parse_yolo_label(lbl_path):
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            rect = Rectangle((x1, y1), bw*w, bh*h, linewidth=2,
                            edgecolor=COLORS[cls], facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1-2, CLASSES[cls], color=COLORS[cls],
                    fontsize=8, fontweight="bold")

        ax.set_title(source, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Sample Frames from Each Data Source",
                 fontsize=16, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(OUT / "sample_collage.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[6/6] {OUT}/sample_collage.png")


def write_summary(sources):
    """Markdown summary for thesis Section 4."""
    total_imgs = sum(len(p) for p in sources.values())
    total_bbox = 0
    cls_totals = Counter()
    for _, pairs in sources.items():
        for _, lbl in pairs:
            boxes = parse_yolo_label(lbl)
            total_bbox += len(boxes)
            for cls, *_ in boxes:
                cls_totals[CLASSES[cls]] += 1

    md = f"""# Dataset Statistics — v3 (Multi-Source)

## Overview

| Metric | Value |
|---|---|
| Total images | {total_imgs:,} |
| Total bounding boxes | {total_bbox:,} |
| Average bboxes per image | {total_bbox/total_imgs:.1f} |
| Number of classes | {len(CLASSES)} |
| Image resolution | 1920×1080 (resized to 640×640 for training) |

## Source Distribution

| Source | Images | Bboxes | Notes |
|---|---|---|---|
"""
    for src, pairs in sources.items():
        bbox_count = sum(len(parse_yolo_label(lbl)) for _, lbl in pairs)
        md += f"| {src} | {len(pairs):,} | {bbox_count:,} | |\n"

    md += f"""
## Class Distribution

| Class | Count | Percentage |
|---|---|---|
"""
    for cls in CLASSES:
        pct = (cls_totals[cls] / total_bbox * 100) if total_bbox else 0
        md += f"| {cls} | {cls_totals[cls]:,} | {pct:.1f}% |\n"

    md += f"""
## Train/Val/Test Split (70/20/10)

| Split | Images |
|---|---|
| Train | {int(total_imgs * 0.70):,} |
| Valid | {int(total_imgs * 0.20):,} |
| Test  | {int(total_imgs * 0.10):,} |

## Generated Figures

- `class_distribution.png` — Stacked bar chart (Section 4.10)
- `source_distribution.png` — Source breakdown (Section 4.1)
- `bbox_heatmap.png` — Spatial density of vehicle locations (Section 4.10)
- `bbox_size_distribution.png` — Object size analysis (Section 4.10)
- `instances_per_image.png` — Detection density (Section 4.10)
- `sample_collage.png` — Source examples (Section 4.1)
"""
    (OUT / "dataset_summary.md").write_text(md)
    print(f"[7/7] {OUT}/dataset_summary.md")


def main():
    print(f"Output directory: {OUT}")
    print()
    print("Collecting data...")
    sources = collect_data()
    for s, p in sources.items():
        print(f"  {s}: {len(p)} pairs")
    print()
    print("Generating figures...")
    fig1_class_distribution(sources)
    fig2_source_distribution(sources)
    fig3_bbox_heatmap(sources)
    fig4_bbox_size_distribution(sources)
    fig5_instances_per_image(sources)
    fig6_sample_collage(sources)
    print()
    print("Writing summary...")
    write_summary(sources)
    print()
    print(f"DONE. All outputs in: {OUT}")


if __name__ == "__main__":
    main()
