"""
Tez icin ekstra 6 figur uretir (FIGURE_GUIDE_EXTENDED.md - Asama 2).

Cikti: docs/thesis/figures/extra/
  - class_showcase.png            (4 sinif × 4 ornek)
  - per_source_heatmaps.png       (6 kaynak icin bbox center heatmap)
  - latency_comparison.png        (CPU vs GPU inference)
  - error_pie_chart.png           (E-DET / E-TRK / E-STM / E-LOC)
  - failure_case_1.png            (val sample zoomed)
  - failure_case_2.png
  - failure_case_3.png
  - repo_tree.png                 (proje folder yapisi)
"""
import subprocess
from pathlib import Path
from collections import defaultdict
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data/datasets/havd_v4_dataset"
RUNS = ROOT / "runs/detect/mobese_v4"
OUT = ROOT / "docs/thesis/figures/extra"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["bus", "car", "motorcycle", "truck"]
CLASS_COLORS = ["#DA01EE", "#00FF00", "#0192F3", "#F18700"]
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ────────────────────────────────────────────────────────────────────
# 1) CLASS SHOWCASE GRID — 4 sinif × 4 ornek
# ────────────────────────────────────────────────────────────────────
def fig_class_showcase():
    print("[1/6] class_showcase ...")
    # Tum kaynaklardan label dosyalari topla, sinif → image+box listesi
    sources = [
        DATASET / "istanbul-traffic-vehicles/train",
        DATASET / "cleaned/cam10",
        DATASET / "cleaned/cam11",
        DATASET / "auto_labeled/file1_night",
        DATASET / "auto_labeled/file3_day",
        DATASET / "auto_labeled/file4_evening",
    ]
    by_class = defaultdict(list)  # cls → [(img_path, [bbox])]
    for src in sources:
        img_dir = src / "images"
        lbl_dir = src / "labels"
        if not img_dir.exists():
            continue
        imgs = list(img_dir.glob("*.jpg"))
        random.shuffle(imgs)
        for img in imgs[:200]:  # her kaynaktan 200 ornek tara (hiz)
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                continue
            try:
                lines = lbl.read_text().strip().splitlines()
            except Exception:
                continue
            # Bu image'da hangi siniflar var?
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                by_class[cls].append((img, (cx, cy, w, h)))
            if all(len(by_class[c]) >= 4 for c in range(4)):
                break
        if all(len(by_class[c]) >= 4 for c in range(4)):
            break

    # 4×4 grid: satir = sinif, sutun = ornek
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for r, cls in enumerate(range(4)):
        samples = by_class[cls][:4]
        for c, (img_path, bbox) in enumerate(samples):
            ax = axes[r, c]
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            cx, cy, bw, bh = bbox
            # Bbox crop (biraz padding)
            x1 = max(0, int((cx - bw/2) * W) - 20)
            y1 = max(0, int((cy - bh/2) * H) - 20)
            x2 = min(W, int((cx + bw/2) * W) + 20)
            y2 = min(H, int((cy + bh/2) * H) + 20)
            crop = img[y1:y2, x1:x2]
            ax.imshow(crop)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(CLASSES[cls], fontsize=14, rotation=0,
                              ha="right", va="center", labelpad=30,
                              color=CLASS_COLORS[cls], fontweight="bold")
    fig.suptitle("Class Showcase — 4 examples per target class", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUT / "class_showcase.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("   ✓")


# ────────────────────────────────────────────────────────────────────
# 2) PER-SOURCE BBOX HEATMAPS
# ────────────────────────────────────────────────────────────────────
def fig_per_source_heatmaps():
    print("[2/6] per_source_heatmaps ...")
    sources = {
        "Roboflow":      DATASET / "istanbul-traffic-vehicles/train/labels",
        "cam10":         DATASET / "cleaned/cam10/labels",
        "cam11":         DATASET / "cleaned/cam11/labels",
        "file1_night":   DATASET / "auto_labeled/file1_night/labels",
        "file3_day":     DATASET / "auto_labeled/file3_day/labels",
        "file4_evening": DATASET / "auto_labeled/file4_evening/labels",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i, (name, lbl_dir) in enumerate(sources.items()):
        ax = axes[i // 3, i % 3]
        if not lbl_dir.exists():
            ax.set_visible(False)
            continue
        heatmap = np.zeros((100, 100), dtype=int)
        n_boxes = 0
        for f in lbl_dir.glob("*.txt"):
            try:
                lines = f.read_text().splitlines()
            except Exception:
                continue
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cx, cy = float(parts[1]), float(parts[2])
                x = min(99, max(0, int(cx * 100)))
                y = min(99, max(0, int(cy * 100)))
                heatmap[y, x] += 1
                n_boxes += 1
        im = ax.imshow(heatmap, cmap="hot", origin="upper", extent=[0, 1, 1, 0])
        ax.set_title(f"{name}\n({n_boxes} boxes)", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Per-Source Bbox Center Position Heatmaps", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT / "per_source_heatmaps.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("   ✓")


# ────────────────────────────────────────────────────────────────────
# 3) INFERENCE LATENCY BAR CHART
# ────────────────────────────────────────────────────────────────────
def fig_latency_comparison():
    print("[3/6] latency_comparison ...")
    stages = ["Preprocess", "Forward", "Postprocess", "TOTAL"]
    cpu = [0.4, 511, 0.5, 511.9]   # Apple M2 (ms)
    gpu = [0.3, 8.6, 0.4, 9.3]     # Tesla T4 (ms)

    x = np.arange(len(stages))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, cpu, w, label="CPU (Apple M2)",
                color="#E55934", edgecolor="black")
    b2 = ax.bar(x + w/2, gpu, w, label="GPU (Tesla T4)",
                color="#4C72B0", edgecolor="black")
    for b, v in zip(list(b1) + list(b2), cpu + gpu):
        ax.text(b.get_x() + b.get_width()/2, v * 1.05,
                f"{v:.1f} ms", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel("Latency per frame (ms, log scale)")
    ax.set_yscale("log")
    ax.set_title("Inference Latency — CPU vs GPU", fontsize=13)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(OUT / "latency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓")


# ────────────────────────────────────────────────────────────────────
# 4) ERROR TYPE PIE CHART
# ────────────────────────────────────────────────────────────────────
def fig_error_pie_chart():
    print("[4/6] error_pie_chart ...")
    # Estimated error type distribution (Ch8 root cause + field test)
    labels = [
        "E-DET\n(detection: missed / mis-classified)",
        "E-LOC\n(localisation drift)",
        "E-TRK\n(track break / ID switch)",
        "E-STM\n(state-machine threshold)",
    ]
    sizes = [42, 28, 18, 12]
    colors = ["#E55934", "#FA7921", "#4C72B0", "#55A868"]
    explode = (0.05, 0, 0, 0)
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")
    ax.set_title("Residual Error Type Distribution (estimated)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(OUT / "error_pie_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓")


# ────────────────────────────────────────────────────────────────────
# 5) FAILURE CASE CROPS (val_batch + label/pred yan yana)
# ────────────────────────────────────────────────────────────────────
def fig_failure_cases():
    print("[5/6] failure_case_1/2/3 ...")
    pairs = [
        ("val_batch0_labels.jpg", "val_batch0_pred.jpg",
         "Validation batch 0 — ground truth (left) vs. v4 predictions (right)"),
        ("val_batch1_labels.jpg", "val_batch1_pred.jpg",
         "Validation batch 1 — small object localisation challenge"),
        ("val_batch2_labels.jpg", "val_batch2_pred.jpg",
         "Validation batch 2 — class confusion regions (car ↔ truck)"),
    ]
    for i, (lbl_name, pred_name, title) in enumerate(pairs, 1):
        lbl_path = RUNS / lbl_name
        pred_path = RUNS / pred_name
        if not lbl_path.exists() or not pred_path.exists():
            print(f"   [SKIP] {lbl_name} / {pred_name} yok")
            continue
        lbl_img = cv2.cvtColor(cv2.imread(str(lbl_path)), cv2.COLOR_BGR2RGB)
        pred_img = cv2.cvtColor(cv2.imread(str(pred_path)), cv2.COLOR_BGR2RGB)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        ax1.imshow(lbl_img); ax1.set_title("Ground Truth", fontsize=13)
        ax1.set_xticks([]); ax1.set_yticks([])
        ax2.imshow(pred_img); ax2.set_title("v4 Prediction", fontsize=13)
        ax2.set_xticks([]); ax2.set_yticks([])
        fig.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(OUT / f"failure_case_{i}.png", dpi=110, bbox_inches="tight")
        plt.close()
    print("   ✓")


# ────────────────────────────────────────────────────────────────────
# 6) REPO TREE PNG
# ────────────────────────────────────────────────────────────────────
def fig_repo_tree():
    print("[6/6] repo_tree ...")
    # Manual tree (tree komutuna bagimliligi yok)
    tree_text = """hatched-area-violation-detection/
├── app.py                           # Gradio web UI
├── configs/
│   ├── config.yaml                  # main runtime config
│   ├── bytetrack.yaml               # tracker config
│   └── zones/                       # per-camera zone polygons
├── data/
│   ├── datasets/
│   │   ├── havd_v4_dataset/         # v4 production dataset (6948 image)
│   │   ├── final_v4/                # path-list + data.yaml
│   │   └── ground_truth/            # field-test JSON
│   └── videos/test/
├── docs/thesis/
│   ├── chapter7_experiments_evaluation.md
│   ├── chapter8_error_analysis.md
│   ├── FIGURE_GUIDE.md
│   └── figures/                     # 30+ thesis figures
├── runs/detect/mobese_v4/           # training output (results, curves, batches)
├── scripts/
│   ├── build_v4_dataset.py
│   ├── compare_v3_v4_metrics.py
│   ├── eval_v4_subsets.py
│   ├── empirical_pipeline_eval.py
│   └── notebooks/08_train_mobese_v4.ipynb
├── src/
│   ├── core/                        # FrameProvider, dataclasses, visualizer
│   ├── pipeline/                    # main orchestrator
│   ├── tracking/                    # ByteTrack wrapper
│   ├── violation/                   # state machine + severity + trajectory
│   ├── zones/                       # polygon ROI manager
│   └── storage/                     # SQLite logger
├── weights/
│   ├── best_v3.pt                   # baseline weights
│   ├── best_v4.pt                   # proposed final weights
│   └── plate.pt                     # licence-plate detector
├── test_clip.sh                     # quick clip + pipeline runner
├── start_label_studio.sh            # local Label Studio launcher
├── requirements.txt
└── README.md"""

    lines = tree_text.split("\n")
    n_lines = len(lines)
    fig_height = max(8, n_lines * 0.2)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.text(0.01, 0.99, tree_text,
            family="monospace", fontsize=9, va="top", ha="left",
            transform=ax.transAxes)
    ax.set_axis_off()
    ax.set_title("Repository Folder Structure", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(OUT / "repo_tree.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("   ✓")


def main():
    print(f"Cikti dizini: {OUT}\n")
    fig_class_showcase()
    fig_per_source_heatmaps()
    fig_latency_comparison()
    fig_error_pie_chart()
    fig_failure_cases()
    fig_repo_tree()
    print(f"\n=== BITTI ===")
    print(f"Uretilen dosyalar:")
    for f in sorted(OUT.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<35} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
