"""
v3 ve v4 modellerini AYNI test setinde calistir, metrikleri karsilastir.

Cikti:
  docs/thesis/figures/comparison/
    11_v3_v4_metrics_table.png      Bar chart: mAP50, mAP50-95, P, R
    11_v3_v4_metrics_table.csv      Sayisal tablo
    12_v3_v4_per_class_map.png      Per-class mAP50 karsilastirma
    13_v3_v4_pr_curves.png          PR curve ust uste
    summary.txt                     Numerik ozet
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data/datasets/final_v4/data.yaml"
OUT = ROOT / "docs/thesis/figures/comparison"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["bus", "car", "motorcycle", "truck"]


def evaluate(weights_path: Path, name: str) -> dict:
    print(f"\n=== {name} degerlendiriliyor: {weights_path.name} ===")
    model = YOLO(str(weights_path))
    res = model.val(
        data=str(DATA_YAML),
        split="test",
        plots=False,
        save_json=False,
        verbose=False,
    )
    return {
        "name": name,
        "mAP50": float(res.box.map50),
        "mAP50_95": float(res.box.map),
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "per_class_mAP50": [float(x) for x in res.box.ap50.tolist()],
        "per_class_mAP50_95": [float(x) for x in res.box.maps.tolist()],
    }


def fig_overall(v3: dict, v4: dict) -> None:
    metrics = ["mAP50", "mAP50_95", "precision", "recall"]
    labels = ["mAP@50", "mAP@50-95", "Precision", "Recall"]
    v3_vals = [v3[m] for m in metrics]
    v4_vals = [v4[m] for m in metrics]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, v3_vals, w, label="v3 (baseline)", color="#DD8452", edgecolor="black")
    b2 = ax.bar(x + w/2, v4_vals, w, label="v4 (manuel + warm-start)", color="#4C72B0", edgecolor="black")
    for b, v in zip(list(b1) + list(b2), v3_vals + v4_vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Skor")
    ax.set_title("v3 vs v4 — Genel Metrik Karsilastirmasi (Test Set)", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT/"11_v3_v4_metrics_table.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_per_class(v3: dict, v4: dict) -> None:
    x = np.arange(len(CLASSES))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # mAP@50
    ax1.bar(x - w/2, v3["per_class_mAP50"], w, label="v3", color="#DD8452", edgecolor="black")
    ax1.bar(x + w/2, v4["per_class_mAP50"], w, label="v4", color="#4C72B0", edgecolor="black")
    for i, (a, b) in enumerate(zip(v3["per_class_mAP50"], v4["per_class_mAP50"])):
        ax1.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax1.text(i + w/2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
        delta = b - a
        color = "green" if delta > 0 else "red"
        ax1.text(i, max(a, b) + 0.07, f"Δ{delta:+.3f}",
                 ha="center", fontsize=10, color=color, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(CLASSES)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("mAP@50")
    ax1.set_title("Per-Class mAP@50")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    # mAP@50-95
    ax2.bar(x - w/2, v3["per_class_mAP50_95"], w, label="v3", color="#DD8452", edgecolor="black")
    ax2.bar(x + w/2, v4["per_class_mAP50_95"], w, label="v4", color="#4C72B0", edgecolor="black")
    for i, (a, b) in enumerate(zip(v3["per_class_mAP50_95"], v4["per_class_mAP50_95"])):
        ax2.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax2.text(i + w/2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
        delta = b - a
        color = "green" if delta > 0 else "red"
        ax2.text(i, max(a, b) + 0.07, f"Δ{delta:+.3f}",
                 ha="center", fontsize=10, color=color, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(CLASSES)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("mAP@50-95")
    ax2.set_title("Per-Class mAP@50-95")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("v3 vs v4 — Sinif Bazli Performans", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT/"12_v3_v4_per_class_map.png", dpi=150, bbox_inches="tight")
    plt.close()


def write_csv_and_txt(v3: dict, v4: dict) -> None:
    csv_path = OUT/"11_v3_v4_metrics_table.csv"
    txt_path = OUT/"summary.txt"

    rows = [
        ("Sinif", "v3 mAP50", "v4 mAP50", "Δ mAP50", "v3 mAP50-95", "v4 mAP50-95", "Δ mAP50-95"),
    ]
    for i, cls in enumerate(CLASSES):
        a50, b50 = v3["per_class_mAP50"][i], v4["per_class_mAP50"][i]
        a95, b95 = v3["per_class_mAP50_95"][i], v4["per_class_mAP50_95"][i]
        rows.append((cls, f"{a50:.4f}", f"{b50:.4f}", f"{b50-a50:+.4f}",
                          f"{a95:.4f}", f"{b95:.4f}", f"{b95-a95:+.4f}"))
    rows.append(("OVERALL",
                 f"{v3['mAP50']:.4f}", f"{v4['mAP50']:.4f}", f"{v4['mAP50']-v3['mAP50']:+.4f}",
                 f"{v3['mAP50_95']:.4f}", f"{v4['mAP50_95']:.4f}", f"{v4['mAP50_95']-v3['mAP50_95']:+.4f}"))

    csv_path.write_text("\n".join([",".join(r) for r in rows]))

    with open(txt_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("v3 vs v4 KARSILASTIRMA OZETI (Test Set, 694 image)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test setindeki obje sayisi degerlendirme ile elde edilir.\n\n")
        f.write(f"{'METRIK':<15} | {'v3':>8} | {'v4':>8} | {'DELTA':>8}\n")
        f.write("-"*50 + "\n")
        for m, lbl in [("mAP50","mAP@50"), ("mAP50_95","mAP@50-95"),
                       ("precision","Precision"), ("recall","Recall")]:
            f.write(f"{lbl:<15} | {v3[m]:>8.4f} | {v4[m]:>8.4f} | {v4[m]-v3[m]:>+8.4f}\n")
        f.write("\n")
        f.write("SINIF BAZLI mAP@50:\n")
        f.write(f"{'sinif':<14} | {'v3':>8} | {'v4':>8} | {'delta':>8}\n")
        f.write("-"*50 + "\n")
        for i, cls in enumerate(CLASSES):
            a, b = v3["per_class_mAP50"][i], v4["per_class_mAP50"][i]
            f.write(f"{cls:<14} | {a:>8.4f} | {b:>8.4f} | {b-a:>+8.4f}\n")


def main():
    v3 = evaluate(ROOT/"weights/best_v3.pt", "v3")
    v4 = evaluate(ROOT/"weights/best_v4.pt", "v4")

    # JSON raw kayit
    (OUT/"raw_metrics.json").write_text(json.dumps({"v3": v3, "v4": v4}, indent=2))

    print("\n=== Figurleri uretiyor ===")
    fig_overall(v3, v4); print("  11_v3_v4_metrics_table.png ✓")
    fig_per_class(v3, v4); print("  12_v3_v4_per_class_map.png ✓")
    write_csv_and_txt(v3, v4); print("  CSV + summary.txt ✓")

    print(f"\nCikti: {OUT}")
    print(f"\nOzet:")
    print(f"  v3 mAP50: {v3['mAP50']:.4f}  →  v4: {v4['mAP50']:.4f}  (Δ {v4['mAP50']-v3['mAP50']:+.4f})")


if __name__ == "__main__":
    main()
