"""
v4 modelini test setinin 3 alt-kume'sinde calistir, sonuc tablosu uret.

Subset'ler:
  A) Roboflow (cam1-5) — community manuel etiketli (genel sehir trafigi)
  B) Pseudo-labeled (file1/3/4) — eski model auto-label (CCTV)
  C) Manuel cam10/cam11 — sen Label Studio'da dogruladigin (drone 4K)

Cikti:
  docs/thesis/figures/comparison/
    15_v4_by_subset.png       Bar chart: her subset icin mAP50, mAP50-95, P, R
    15_v4_by_subset.csv       Sayisal tablo
    15_v4_subset_summary.txt  Analiz
"""
import json
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TEST_TXT = ROOT / "data/datasets/final_v4/test.txt"
OUT = ROOT / "docs/thesis/figures/comparison"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["bus", "car", "motorcycle", "truck"]

SUBSETS = {
    "A_roboflow":  ("istanbul-traffic-vehicles", "Roboflow (cam1-5, community manuel)"),
    "B_pseudo":    ("auto_labeled",              "Pseudo-label (file1/3/4, eski model)"),
    "C_manuel_new":("cleaned",                   "Manuel cam10/cam11 (sen, Label Studio)"),
}


def split_test_set() -> dict:
    paths = [p.strip() for p in TEST_TXT.read_text().splitlines() if p.strip()]
    splits = {k: [] for k in SUBSETS}
    for p in paths:
        for k, (marker, _) in SUBSETS.items():
            if marker in p:
                splits[k].append(p)
                break
    return splits


def eval_subset(model: YOLO, subset_paths: list[str], yaml_template: str) -> dict:
    if not subset_paths:
        return None
    # Gecici test.txt + data.yaml yaz
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp/"sub_test.txt").write_text("\n".join(subset_paths))
        yaml = f"""path: {tmp}
train: sub_test.txt
val: sub_test.txt
test: sub_test.txt
nc: 4
names: ['bus', 'car', 'motorcycle', 'truck']
"""
        (tmp/"data.yaml").write_text(yaml)
        res = model.val(data=str(tmp/"data.yaml"), split="test",
                        plots=False, save_json=False, verbose=False)
    return {
        "n_images": len(subset_paths),
        "mAP50": float(res.box.map50),
        "mAP50_95": float(res.box.map),
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "per_class_mAP50": [float(x) for x in res.box.ap50.tolist()],
        "per_class_mAP50_95": [float(x) for x in res.box.maps.tolist()],
    }


def fig_summary(results: dict) -> None:
    labels = ["mAP@50", "mAP@50-95", "Precision", "Recall"]
    keys   = ["mAP50", "mAP50_95", "precision", "recall"]
    subsets = [k for k in SUBSETS if results.get(k) is not None]
    sub_labels = [SUBSETS[k][1].split('(')[0].strip() for k in subsets]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, k in enumerate(subsets):
        vals = [results[k][m] for m in keys]
        bars = ax.bar(x + (i-1)*w, vals, w, label=sub_labels[i],
                      color=colors[i], edgecolor="black")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}",
                    ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Skor")
    ax.set_title("v4 — Test Setinin 3 Alt-Kumesinde Performans", fontsize=13)
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT/"15_v4_by_subset.png", dpi=150, bbox_inches="tight")
    plt.close()


def write_csv_and_txt(results: dict, splits: dict) -> None:
    csv = ["subset,n_images,mAP50,mAP50_95,precision,recall,bus,car,motorcycle,truck"]
    for k, r in results.items():
        if r is None:
            continue
        csv.append(f"{SUBSETS[k][1]},{r['n_images']},{r['mAP50']:.4f},"
                   f"{r['mAP50_95']:.4f},{r['precision']:.4f},{r['recall']:.4f},"
                   f"{r['per_class_mAP50'][0]:.4f},{r['per_class_mAP50'][1]:.4f},"
                   f"{r['per_class_mAP50'][2]:.4f},{r['per_class_mAP50'][3]:.4f}")
    (OUT/"15_v4_by_subset.csv").write_text("\n".join(csv))

    with open(OUT/"15_v4_subset_summary.txt", "w") as f:
        f.write("="*72 + "\n")
        f.write("v4 — TEST SETININ ALT-KUMELERINDE PERFORMANS\n")
        f.write("="*72 + "\n\n")
        for k, r in results.items():
            if r is None:
                f.write(f"\n[ {SUBSETS[k][1]} ] — VERI YOK\n")
                continue
            f.write(f"\n[ {SUBSETS[k][1]} ]\n")
            f.write(f"  Image: {r['n_images']}\n")
            f.write(f"  mAP@50     : {r['mAP50']:.4f}\n")
            f.write(f"  mAP@50-95  : {r['mAP50_95']:.4f}\n")
            f.write(f"  Precision  : {r['precision']:.4f}\n")
            f.write(f"  Recall     : {r['recall']:.4f}\n")
            f.write(f"  Per-class mAP@50:\n")
            for i, cls in enumerate(CLASSES):
                f.write(f"    {cls:<12}: {r['per_class_mAP50'][i]:.4f}\n")


def main():
    splits = split_test_set()
    print("Test seti alt-kumelere ayrildi:")
    for k, paths in splits.items():
        print(f"  {SUBSETS[k][1]}: {len(paths)} image")

    print("\n=== v4 yukleniyor ===")
    model = YOLO(str(ROOT/"weights/best_v4.pt"))

    results = {}
    for k, paths in splits.items():
        if not paths:
            print(f"\n[{k}] ATLA (path yok)")
            results[k] = None
            continue
        print(f"\n=== {k}: {len(paths)} image degerlendiriliyor ===")
        results[k] = eval_subset(model, paths, "")

    (OUT/"15_v4_subset_raw.json").write_text(json.dumps(results, indent=2))
    print("\n=== Figur + CSV uretiliyor ===")
    fig_summary(results); print("  15_v4_by_subset.png ✓")
    write_csv_and_txt(results, splits); print("  CSV + summary.txt ✓")

    print("\n=== OZET ===")
    for k, r in results.items():
        if r is None:
            continue
        print(f"  {SUBSETS[k][1]:<50}: mAP50={r['mAP50']:.4f}")


if __name__ == "__main__":
    main()
