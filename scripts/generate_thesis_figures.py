"""
Tez icin dataset analiz figurleri uretir.

Cikti: docs/thesis/figures/
  01_source_distribution.png       Kaynak pie chart
  02_class_distribution.png        Sinif bar chart
  03_source_class_heatmap.png      Kaynak x sinif heatmap
  04_bbox_size_distribution.png    Kucuk/orta/buyuk obje histogram
  05_bbox_position_heatmap.png     Obje pozisyon heatmap
  06_objects_per_image.png         Image basina obje sayisi
  07_manual_vs_pseudo.png          Manuel vs pseudo etiket karsi
  08_sample_grid.png               Kaynak basina 4 ornek (24 total)
  09_bbox_overlay_examples.png     Etiketli 8 frame
  10_summary_stats.txt             Numerik ozet
"""
from pathlib import Path
from collections import Counter
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import seaborn as sns

ROOT = Path("/Users/sertacakalin/Desktop/Projects/bitirme/hatched-area-violation-detection")
RB = Path("/Users/sertacakalin/Desktop/istanbul-traffic-vehicles")
OUT = ROOT / "docs/thesis/figures"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["bus", "car", "motorcycle", "truck"]
CLASS_COLORS = ["#DA01EE", "#00FF00", "#0192F3", "#F18700"]

SOURCES = {
    "Roboflow":     {"images": RB/"train/images",                          "labels": RB/"train/labels",                          "type": "manuel"},
    "cam10":        {"images": ROOT/"data/datasets/cleaned/cam10/images",  "labels": ROOT/"data/datasets/cleaned/cam10/labels",  "type": "manuel"},
    "cam11":        {"images": ROOT/"data/datasets/cleaned/cam11/images",  "labels": ROOT/"data/datasets/cleaned/cam11/labels",  "type": "manuel"},
    "file1_night":  {"images": ROOT/"data/datasets/auto_labeled/file1_night/images",  "labels": ROOT/"data/datasets/auto_labeled/file1_night/labels",  "type": "pseudo"},
    "file3_day":    {"images": ROOT/"data/datasets/auto_labeled/file3_day/images",    "labels": ROOT/"data/datasets/auto_labeled/file3_day/labels",    "type": "pseudo"},
    "file4_evening":{"images": ROOT/"data/datasets/auto_labeled/file4_evening/images","labels": ROOT/"data/datasets/auto_labeled/file4_evening/labels","type": "pseudo"},
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def collect_stats():
    """Tum kaynaklari tara: per-source counts, per-class counts, bbox boyut/pozisyon."""
    stats = {}
    all_bboxes = []   # (x_cx, y_cx, w, h, class)
    obj_counts = []   # image basina obje sayisi
    for name, info in SOURCES.items():
        cls_counter = Counter()
        n_img_with_label = 0
        for img_path in info["images"].glob("*.jpg"):
            lbl_path = info["labels"] / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            try:
                lines = lbl_path.read_text().strip().splitlines()
            except Exception:
                continue
            n_img_with_label += 1
            obj_counts.append(len(lines))
            for ln in lines:
                parts = ln.split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    cls_counter[cls] += 1
                    all_bboxes.append((cx, cy, w, h, cls, name))
        stats[name] = {"images": n_img_with_label, "classes": cls_counter, "type": info["type"]}
        print(f"  {name}: {n_img_with_label} image, {sum(cls_counter.values())} obj")
    return stats, all_bboxes, obj_counts


def fig_01_source_distribution(stats):
    sizes = [s["images"] for s in stats.values()]
    labels = [f"{n}\n({s['images']})" for n, s in stats.items()]
    colors = ["#4C72B0" if s["type"] == "manuel" else "#DD8452" for s in stats.values()]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140,
           wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("v4 Veri Seti — Kaynak Dağılımı (image)", fontsize=14, pad=20)
    # legend for color meaning
    legend_elements = [
        patches.Patch(color="#4C72B0", label="Manuel etiketli"),
        patches.Patch(color="#DD8452", label="Pseudo-label (v3-öncesi modelle)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT/"01_source_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_02_class_distribution(stats):
    totals = Counter()
    for s in stats.values():
        for k, v in s["classes"].items():
            totals[k] += v
    counts = [totals[i] for i in range(len(CLASSES))]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(CLASSES, counts, color=CLASS_COLORS, edgecolor="black")
    for b, c in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(counts)*0.01,
                f"{c}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Obje sayısı (annotated bbox)")
    ax.set_title("v4 Sınıf Dağılımı (toplam obje)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT/"02_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_03_source_class_heatmap(stats):
    matrix = np.zeros((len(SOURCES), len(CLASSES)), dtype=int)
    for i, (name, s) in enumerate(stats.items()):
        for j in range(len(CLASSES)):
            matrix[i, j] = s["classes"][j]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=list(stats.keys()),
                cmap="YlGnBu", ax=ax, cbar_kws={"label": "Obje sayısı"})
    ax.set_title("Kaynak × Sınıf Obje Dağılımı", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT/"03_source_class_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_04_bbox_size(all_bboxes):
    # COCO-style: small <32^2, medium 32^2-96^2, large >96^2 (normalize'i 640x640'a göre yap)
    areas_px = [(w*640)*(h*640) for _,_,w,h,_,_ in all_bboxes]
    bins = [0, 32**2, 96**2, 640**2]
    labels = ["Küçük\n(<32²)", "Orta\n(32²-96²)", "Büyük\n(>96²)"]
    counts = np.histogram(areas_px, bins=bins)[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=["#E55934", "#FA7921", "#9BC53D"], edgecolor="black")
    for b, c in zip(bars, counts):
        pct = 100*c/sum(counts)
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(counts)*0.01,
                f"{c}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Obje sayısı")
    ax.set_title("Bbox Boyut Dağılımı (COCO standardı, imgsz=640)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT/"04_bbox_size_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_05_bbox_position_heatmap(all_bboxes):
    heatmap = np.zeros((100, 100), dtype=int)
    for cx, cy, w, h, _, _ in all_bboxes:
        x = int(cx * 100); y = int(cy * 100)
        x = min(99, max(0, x)); y = min(99, max(0, y))
        heatmap[y, x] += 1
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heatmap, cmap="hot", origin="upper", extent=[0, 1, 1, 0])
    ax.set_title("Bbox Merkez Pozisyon Heatmap (tüm dataset)", fontsize=14)
    ax.set_xlabel("Normalize X")
    ax.set_ylabel("Normalize Y")
    plt.colorbar(im, ax=ax, label="Obje yoğunluğu")
    plt.tight_layout()
    plt.savefig(OUT/"05_bbox_position_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_06_objects_per_image(obj_counts):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, max(obj_counts)+2)
    ax.hist(obj_counts, bins=bins, color="#4C72B0", edgecolor="black", alpha=0.85)
    ax.axvline(np.mean(obj_counts), color="red", linestyle="--", label=f"Ortalama: {np.mean(obj_counts):.1f}")
    ax.axvline(np.median(obj_counts), color="orange", linestyle="--", label=f"Medyan: {np.median(obj_counts):.0f}")
    ax.set_xlabel("Image başına obje sayısı")
    ax.set_ylabel("Image sayısı")
    ax.set_title("Image Başına Obje Yoğunluğu", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT/"06_objects_per_image.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_07_manual_vs_pseudo(stats):
    manuel_imgs = sum(s["images"] for s in stats.values() if s["type"] == "manuel")
    pseudo_imgs = sum(s["images"] for s in stats.values() if s["type"] == "pseudo")
    manuel_objs = sum(sum(s["classes"].values()) for s in stats.values() if s["type"] == "manuel")
    pseudo_objs = sum(sum(s["classes"].values()) for s in stats.values() if s["type"] == "pseudo")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(["Manuel", "Pseudo"], [manuel_imgs, pseudo_imgs], color=["#4C72B0", "#DD8452"], edgecolor="black")
    for i, v in enumerate([manuel_imgs, pseudo_imgs]):
        ax1.text(i, v+max(manuel_imgs, pseudo_imgs)*0.01, f"{v}", ha="center", va="bottom", fontsize=12)
    ax1.set_ylabel("Image sayısı")
    ax1.set_title("Image — Manuel vs Pseudo")
    ax1.grid(axis="y", alpha=0.3)
    ax2.bar(["Manuel", "Pseudo"], [manuel_objs, pseudo_objs], color=["#4C72B0", "#DD8452"], edgecolor="black")
    for i, v in enumerate([manuel_objs, pseudo_objs]):
        ax2.text(i, v+max(manuel_objs, pseudo_objs)*0.01, f"{v}", ha="center", va="bottom", fontsize=12)
    ax2.set_ylabel("Obje sayısı")
    ax2.set_title("Obje — Manuel vs Pseudo")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT/"07_manual_vs_pseudo.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_08_sample_grid():
    fig, axes = plt.subplots(len(SOURCES), 4, figsize=(16, 4*len(SOURCES)))
    for i, (name, info) in enumerate(SOURCES.items()):
        imgs = list(info["images"].glob("*.jpg"))
        if not imgs:
            continue
        picks = random.sample(imgs, min(4, len(imgs)))
        for j in range(4):
            ax = axes[i, j]
            if j < len(picks):
                img = cv2.imread(str(picks[j]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                if j == 0:
                    ax.set_ylabel(f"{name}\n({info['type']})", fontsize=11, rotation=0,
                                  ha="right", va="center", labelpad=40)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Kaynak Başına Örnek Frame'ler", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT/"08_sample_grid.png", dpi=120, bbox_inches="tight")
    plt.close()


def fig_09_bbox_overlay():
    """8 image'a YOLO label'larini cizip kaydet."""
    candidates = []
    for name, info in SOURCES.items():
        for img_path in list(info["images"].glob("*.jpg"))[:30]:
            lbl = info["labels"] / f"{img_path.stem}.txt"
            if lbl.exists() and lbl.read_text().strip():
                candidates.append((img_path, lbl, name))
    picks = random.sample(candidates, min(8, len(candidates)))
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, (img_path, lbl, name) in enumerate(picks):
        ax = axes[i//4, i%4]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        for ln in lbl.read_text().splitlines():
            parts = ln.split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1 = int((cx-w/2)*W); y1 = int((cy-h/2)*H)
                bw = int(w*W); bh = int(h*H)
                color = CLASS_COLORS[cls]
                rect = patches.Rectangle((x1, y1), bw, bh, linewidth=2, edgecolor=color, facecolor="none")
                ax.add_patch(rect)
                ax.text(x1, y1-5, CLASSES[cls], color=color, fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
        ax.imshow(img)
        ax.set_title(f"{name} | {img_path.name}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Etiket Örnekleri (8 farklı kaynaktan)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT/"09_bbox_overlay_examples.png", dpi=120, bbox_inches="tight")
    plt.close()


def fig_10_summary(stats, all_bboxes, obj_counts):
    totals = Counter()
    for s in stats.values():
        for k, v in s["classes"].items():
            totals[k] += v
    total_imgs = sum(s["images"] for s in stats.values())
    total_objs = sum(totals.values())

    with open(OUT/"10_summary_stats.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("v4 DATASET — TEZ İÇİN SAYISAL ÖZET\n")
        f.write("="*60 + "\n\n")
        f.write(f"Toplam image:          {total_imgs}\n")
        f.write(f"Toplam annotated obje: {total_objs}\n")
        f.write(f"Image başına ortalama: {total_objs/total_imgs:.2f} obje\n\n")
        f.write("KAYNAK BAZLI:\n")
        for n, s in stats.items():
            f.write(f"  {n:<14}: {s['images']:>5} image, {sum(s['classes'].values()):>5} obj  [{s['type']}]\n")
        f.write("\nSINIF BAZLI:\n")
        for i, name in enumerate(CLASSES):
            f.write(f"  {name:<12}: {totals[i]:>6} obj  ({100*totals[i]/total_objs:>5.2f}%)\n")
        f.write("\nIMAGE BAŞINA OBJE:\n")
        f.write(f"  min: {min(obj_counts)}\n")
        f.write(f"  max: {max(obj_counts)}\n")
        f.write(f"  ortalama: {np.mean(obj_counts):.2f}\n")
        f.write(f"  medyan:   {np.median(obj_counts):.0f}\n")
        f.write(f"  std:      {np.std(obj_counts):.2f}\n")


def main():
    print("=== Stats topluyoruz (7K dosya, ~30 sn) ===")
    stats, all_bboxes, obj_counts = collect_stats()
    print(f"\nToplam bbox: {len(all_bboxes)}")
    print("\n=== Figürleri uretiyor ===")
    fig_01_source_distribution(stats);    print("  01 ✓")
    fig_02_class_distribution(stats);     print("  02 ✓")
    fig_03_source_class_heatmap(stats);   print("  03 ✓")
    fig_04_bbox_size(all_bboxes);         print("  04 ✓")
    fig_05_bbox_position_heatmap(all_bboxes); print("  05 ✓")
    fig_06_objects_per_image(obj_counts); print("  06 ✓")
    fig_07_manual_vs_pseudo(stats);       print("  07 ✓")
    fig_08_sample_grid();                 print("  08 ✓")
    fig_09_bbox_overlay();                print("  09 ✓")
    fig_10_summary(stats, all_bboxes, obj_counts); print("  10 ✓")
    print(f"\nTUM FIGURLER: {OUT}")


if __name__ == "__main__":
    main()
