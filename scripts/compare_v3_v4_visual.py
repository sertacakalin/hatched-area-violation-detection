"""
v3 ve v4 modellerini ayni test image'larinda calistir, YAN YANA gorsel uret.

Cikti:
  docs/thesis/figures/comparison/
    14_v3_v4_side_by_side.png    8 image x 2 model (toplam 16 kare grid)
    14_v3_v4_individual/         Her ornek icin tek dosya (yuksek cozunurluk)
"""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TEST_TXT = ROOT / "data/datasets/final_v4/test.txt"
OUT = ROOT / "docs/thesis/figures/comparison"
INDIV = OUT / "14_v3_v4_individual"
INDIV.mkdir(parents=True, exist_ok=True)

CLASSES = ["bus", "car", "motorcycle", "truck"]
COLORS = ["#DA01EE", "#00FF00", "#0192F3", "#F18700"]  # bus/car/moto/truck
SEED = 42
N_SAMPLES = 8


def predict_and_draw(model: YOLO, img_path: Path, ax, title: str) -> None:
    res = model.predict(str(img_path), conf=0.35, verbose=False)[0]
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    ax.imshow(img)

    if res.boxes is not None:
        for box in res.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            color = COLORS[cls]
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                                     edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"{CLASSES[cls]} {conf:.2f}", color=color,
                    fontsize=7, bbox=dict(boxstyle="round,pad=0.15",
                                          facecolor="black", alpha=0.6))

    n_det = len(res.boxes) if res.boxes is not None else 0
    ax.set_title(f"{title} — {n_det} tespit", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    random.seed(SEED)
    test_paths = [Path(p.strip()) for p in TEST_TXT.read_text().splitlines() if p.strip()]
    samples = random.sample(test_paths, min(N_SAMPLES, len(test_paths)))

    v3 = YOLO(str(ROOT/"weights/best_v3.pt"))
    v4 = YOLO(str(ROOT/"weights/best_v4.pt"))

    # Grid: 8 image, her satirda 2 model (v3 sol, v4 sag)
    fig, axes = plt.subplots(N_SAMPLES, 2, figsize=(20, 5*N_SAMPLES))
    for i, img_path in enumerate(samples):
        if not img_path.exists():
            print(f"  ATLA: {img_path}")
            continue
        print(f"  [{i+1}/{N_SAMPLES}] {img_path.name}")
        predict_and_draw(v3, img_path, axes[i, 0], f"v3 | {img_path.stem[:30]}")
        predict_and_draw(v4, img_path, axes[i, 1], f"v4 | {img_path.stem[:30]}")

    plt.suptitle("v3 vs v4 Tahmin Karsilastirmasi (8 Test Image)",
                 fontsize=16, y=1.001)
    plt.tight_layout()
    plt.savefig(OUT/"14_v3_v4_side_by_side.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nGrid: {OUT/'14_v3_v4_side_by_side.png'}")

    # Individual files (yuksek cozunurluk)
    print("\n=== Bireysel yuksek cozunurluklu dosyalar ===")
    for i, img_path in enumerate(samples):
        if not img_path.exists():
            continue
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        predict_and_draw(v3, img_path, axes[0], "v3")
        predict_and_draw(v4, img_path, axes[1], "v4")
        plt.suptitle(f"Image: {img_path.name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(INDIV/f"sample_{i+1:02d}_{img_path.stem[:30]}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
    print(f"Bireysel: {INDIV}")


if __name__ == "__main__":
    main()
