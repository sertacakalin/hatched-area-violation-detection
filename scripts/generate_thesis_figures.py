"""Tez için tüm grafikleri üreten script.

Kullanım:
    python scripts/generate_thesis_figures.py --eval-report results/evaluation/eval_test_01_bytetrack.json

Üretilen grafikler:
    1. Şiddet skoru histogram
    2. İhlal tipi dağılımı (pasta grafik)
    3. TP vs FP şiddet skoru box plot
    4. Eşik değeri - Precision/Recall eğrisi
    5. Confusion matrix
    6. Bileşen ağırlık katkısı (radar chart)
    7. Zaman çizelgesi (timeline) — ihlallerin videodaki konumu
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fig_severity_histogram(predictions: list[dict], output_dir: Path):
    """1. Şiddet skoru dağılımı histogramı."""
    scores = [p["severity_score"] for p in predictions]
    if not scores:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = []
    for s in scores:
        if s < 25: colors.append('#4CAF50')
        elif s < 50: colors.append('#FF9800')
        elif s < 75: colors.append('#F44336')
        else: colors.append('#9C27B0')

    n, bins, patches = ax.hist(scores, bins=20, range=(0, 100),
                                edgecolor='black', linewidth=0.5)
    # Renklendirme
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 25: patch.set_facecolor('#4CAF50')
        elif left_edge < 50: patch.set_facecolor('#FF9800')
        elif left_edge < 75: patch.set_facecolor('#F44336')
        else: patch.set_facecolor('#9C27B0')

    ax.axvline(x=25, color='gray', linestyle='--', alpha=0.7, label='Seviye sınırları')
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=75, color='gray', linestyle='--', alpha=0.7)

    ax.set_xlabel('Şiddet Skoru', fontsize=12)
    ax.set_ylabel('İhlal Sayısı', fontsize=12)
    ax.set_title('İhlal Şiddet Skoru Dağılımı', fontsize=14)

    legend_patches = [
        mpatches.Patch(color='#4CAF50', label='DÜŞÜK (0-25)'),
        mpatches.Patch(color='#FF9800', label='ORTA (25-50)'),
        mpatches.Patch(color='#F44336', label='YÜKSEK (50-75)'),
        mpatches.Patch(color='#9C27B0', label='KRİTİK (75-100)'),
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()
    path = output_dir / "severity_histogram.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def fig_violation_types(predictions: list[dict], output_dir: Path):
    """2. İhlal tipi dağılımı pasta grafik."""
    types = {}
    for p in predictions:
        t = p.get("violation_type", "DİĞER")
        types[t] = types.get(t, 0) + 1

    if not types:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    colors_map = {
        'KAYNAK': '#F44336', 'SEYİR': '#9C27B0',
        'KENAR_TEMASI': '#4CAF50', 'DİĞER': '#607D8B'
    }
    labels = list(types.keys())
    sizes = list(types.values())
    colors = [colors_map.get(l, '#607D8B') for l in labels]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 11}
    )
    ax.set_title('İhlal Tipi Dağılımı', fontsize=14)

    plt.tight_layout()
    path = output_dir / "violation_types_pie.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def fig_tp_fp_boxplot(report: dict, output_dir: Path):
    """3. TP vs FP şiddet skoru karşılaştırması."""
    tp_scores = [d["pred"]["severity_score"] for d in report.get("tp_details", [])]
    fp_scores = [p["severity_score"] for p in report.get("fp_predictions", [])]

    if not tp_scores and not fp_scores:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    data = []
    labels = []
    colors = []
    if tp_scores:
        data.append(tp_scores)
        labels.append(f'True Positive\n(n={len(tp_scores)})')
        colors.append('#4CAF50')
    if fp_scores:
        data.append(fp_scores)
        labels.append(f'False Positive\n(n={len(fp_scores)})')
        colors.append('#F44336')

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Şiddet Skoru', fontsize=12)
    ax.set_title('TP vs FP Şiddet Skoru Karşılaştırması', fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = output_dir / "tp_fp_boxplot.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def fig_threshold_curve(report: dict, output_dir: Path):
    """4. Eşik değeri - Precision/Recall eğrisi."""
    tp_scores = [d["pred"]["severity_score"] for d in report.get("tp_details", [])]
    fp_scores = [p["severity_score"] for p in report.get("fp_predictions", [])]
    fn_count = len(report.get("fn_ground_truths", []))

    if not tp_scores and not fp_scores:
        return

    thresholds = list(range(0, 101, 5))
    precisions = []
    recalls = []

    for t in thresholds:
        tp_remaining = sum(1 for s in tp_scores if s >= t)
        fp_remaining = sum(1 for s in fp_scores if s >= t)
        fn_total = fn_count + sum(1 for s in tp_scores if s < t)

        p = tp_remaining / (tp_remaining + fp_remaining) if (tp_remaining + fp_remaining) > 0 else 0
        r = tp_remaining / (tp_remaining + fn_total) if (tp_remaining + fn_total) > 0 else 0
        precisions.append(p)
        recalls.append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, 'b-o', label='Precision', markersize=3)
    ax.plot(thresholds, recalls, 'r-s', label='Recall', markersize=3)

    # F1 eğrisi
    f1s = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(precisions, recalls)]
    ax.plot(thresholds, f1s, 'g--^', label='F1', markersize=3)

    # En iyi F1 noktası
    best_idx = np.argmax(f1s)
    ax.axvline(x=thresholds[best_idx], color='gray', linestyle=':', alpha=0.7)
    ax.annotate(f'En iyi F1={f1s[best_idx]:.2f}\n(eşik={thresholds[best_idx]})',
                xy=(thresholds[best_idx], f1s[best_idx]),
                xytext=(thresholds[best_idx]+10, f1s[best_idx]-0.1),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10)

    ax.set_xlabel('Minimum Şiddet Skoru Eşiği', fontsize=12)
    ax.set_ylabel('Değer', fontsize=12)
    ax.set_title('Şiddet Skoru Eşiği — Precision / Recall / F1', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / "threshold_precision_recall.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def fig_timeline(predictions: list[dict], output_dir: Path):
    """5. Zaman çizelgesi — ihlallerin videodaki konumu."""
    if not predictions:
        return

    fig, ax = plt.subplots(figsize=(12, 3))

    timestamps = [p["timestamp"] for p in predictions]
    scores = [p["severity_score"] for p in predictions]
    types = [p.get("violation_type", "DİĞER") for p in predictions]

    color_map = {
        'KAYNAK': '#F44336', 'SEYİR': '#9C27B0',
        'KENAR_TEMASI': '#4CAF50', 'DİĞER': '#607D8B'
    }
    colors = [color_map.get(t, '#607D8B') for t in types]

    ax.scatter(timestamps, scores, c=colors, s=80, edgecolors='black',
               linewidth=0.5, zorder=5)

    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.4)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.4)

    ax.set_xlabel('Zaman (saniye)', fontsize=12)
    ax.set_ylabel('Şiddet Skoru', fontsize=12)
    ax.set_title('İhlal Zaman Çizelgesi', fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)

    plt.tight_layout()
    path = output_dir / "violation_timeline.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def fig_component_breakdown(predictions: list[dict], output_dir: Path):
    """6. Bileşen katkı analizi — her metriğin ortalama katkısı."""
    if not predictions:
        return

    components = {"duration": [], "distance": [], "depth": [], "angle": []}
    for p in predictions:
        tm = p.get("trajectory_metrics", {})
        for key in components:
            if key in tm:
                weighted = tm[key]["normalized"] * tm[key]["weight"] * 100
                components[key].append(weighted)

    if not any(components.values()):
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    labels_tr = {
        "duration": "Süre", "distance": "Mesafe",
        "depth": "Derinlik", "angle": "Açı"
    }
    names = [labels_tr[k] for k in components]
    means = [np.mean(v) if v else 0 for v in components.values()]
    colors = ['#2196F3', '#FF9800', '#F44336', '#4CAF50']

    bars = ax.bar(names, means, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=11)

    ax.set_ylabel('Ortalama Katkı (skor puanı)', fontsize=12)
    ax.set_title('Şiddet Skoru Bileşen Katkıları', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = output_dir / "component_breakdown.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def main():
    parser = argparse.ArgumentParser(description="Tez grafikleri oluştur")
    parser.add_argument("--eval-report", help="Değerlendirme raporu JSON")
    parser.add_argument("--predictions", help="Ham tahmin JSON (eval-report yoksa)")
    parser.add_argument("--output", default="results/figures",
                        help="Grafik çıktı dizini")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Tez grafikleri oluşturuluyor...")

    if args.eval_report:
        report = load_report(args.eval_report)
        predictions = report.get("predictions", [])

        fig_severity_histogram(predictions, output_dir)
        fig_violation_types(predictions, output_dir)
        fig_tp_fp_boxplot(report, output_dir)
        fig_threshold_curve(report, output_dir)
        fig_timeline(predictions, output_dir)
        fig_component_breakdown(predictions, output_dir)

    elif args.predictions:
        with open(args.predictions) as f:
            predictions = json.load(f)

        fig_severity_histogram(predictions, output_dir)
        fig_violation_types(predictions, output_dir)
        fig_timeline(predictions, output_dir)
        fig_component_breakdown(predictions, output_dir)

    else:
        print("--eval-report veya --predictions gerekli")
        sys.exit(1)

    print(f"\nTüm grafikler: {output_dir}/")


if __name__ == "__main__":
    main()
