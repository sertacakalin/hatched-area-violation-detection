"""Şiddet skorlama ağırlık optimizasyonu — Grid Search.

Ground truth ile karşılaştırarak en yüksek F1 veren ağırlık
kombinasyonunu bulur. Tezde "ağırlıkları nasıl belirledin?"
sorusuna kanıtlı cevap verir.

Kullanım:
    python scripts/optimize_weights.py \
        --video data/videos/test/test_01.mp4 \
        --ground-truth data/ground_truth/test_01.json
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.core.config import Config
from src.pipeline.pipeline import Pipeline
from src.violation.severity import SeverityScorer
from src.violation.trajectory import TrajectoryAnalyzer


def evaluate_weights(predictions_raw: list[dict], ground_truths: list[dict],
                     weights: dict, threshold: float = 25.0,
                     tolerance_frames: int = 30) -> dict:
    """Verilen ağırlıklarla skorları yeniden hesapla ve F1 döndür."""
    scorer = SeverityScorer(**weights)

    # Yeniden skorla
    predictions = []
    for p in predictions_raw:
        from src.violation.trajectory import TrajectoryMetrics
        m = TrajectoryMetrics(
            track_id=p["track_id"],
            in_zone_frames=p["raw_duration"],
            in_zone_distance=p["raw_distance"],
            penetration_depth=p["raw_depth"],
            crossing_angle=p["raw_angle"],
        )
        result = scorer.score(m)
        if result.score >= threshold:
            predictions.append({
                "frame_number": p["frame_number"],
                "severity_score": result.score,
            })

    # Eşleştirme
    gt_matched = [False] * len(ground_truths)
    pred_matched = [False] * len(predictions)

    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truths):
            if gt_matched[gi]:
                continue
            gt_start = gt["start_frame"] - tolerance_frames
            gt_end = gt["end_frame"] + tolerance_frames
            if gt_start <= pred["frame_number"] <= gt_end:
                gt_matched[gi] = True
                pred_matched[pi] = True
                break

    tp = sum(gt_matched)
    fp = sum(1 for m in pred_matched if not m)
    fn = sum(1 for m in gt_matched if not m)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def main():
    parser = argparse.ArgumentParser(description="Ağırlık optimizasyonu")
    parser.add_argument("--video", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--zone", default="configs/zones/e5_avcilar.json")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="results/optimization")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ground truth
    with open(args.ground_truth) as f:
        gt_data = json.load(f)
    gt_violations = gt_data["violations"]

    # Pipeline çalıştır ve ham metrikleri topla
    print("Pipeline çalıştırılıyor (ham metrikler toplanıyor)...")
    config = Config(args.config)
    config._data["general"]["video_source"] = args.video
    config._data["zone"]["zone_file"] = args.zone
    config._data["general"]["save_video"] = False

    pipeline = Pipeline(config)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_predictions = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tracked = pipeline.tracker.update(None, frame)
        tracked, violations = pipeline.violation_detector.process_frame(
            tracked, frame, frame_num, fps
        )
        for v in violations:
            tm = v.trajectory_metrics
            raw_predictions.append({
                "frame_number": v.frame_number,
                "track_id": v.track_id,
                "raw_duration": tm["duration"]["raw"],
                "raw_distance": tm["distance"]["raw"],
                "raw_depth": tm["depth"]["raw"],
                "raw_angle": tm["angle"]["raw"],
            })
        frame_num += 1
        if frame_num % 500 == 0:
            print(f"  {frame_num}/{total}")

    cap.release()
    print(f"  {len(raw_predictions)} ihlal toplandı")

    # Grid Search
    print("\nGrid Search başlıyor...")
    step = 0.05
    weight_values = np.arange(0.05, 0.55, step)

    best_f1 = 0
    best_weights = {}
    results = []

    count = 0
    for w1 in weight_values:
        for w2 in weight_values:
            for w3 in weight_values:
                w4 = round(1.0 - w1 - w2 - w3, 2)
                if w4 < 0.05 or w4 > 0.50:
                    continue

                weights = {
                    "w_duration": round(w1, 2),
                    "w_distance": round(w2, 2),
                    "w_depth": round(w3, 2),
                    "w_angle": round(w4, 2),
                }
                metrics = evaluate_weights(
                    raw_predictions, gt_violations, weights
                )

                results.append({**weights, **metrics})

                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_weights = weights.copy()
                    best_metrics = metrics.copy()

                count += 1

    print(f"\n{count} kombinasyon test edildi")
    print(f"\nEN İYİ AĞIRLIKLAR:")
    print(f"  Süre     (w1): {best_weights['w_duration']}")
    print(f"  Mesafe   (w2): {best_weights['w_distance']}")
    print(f"  Derinlik (w3): {best_weights['w_depth']}")
    print(f"  Açı      (w4): {best_weights['w_angle']}")
    print(f"\n  F1: {best_f1:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")

    # Sonuçları kaydet
    report = {
        "best_weights": best_weights,
        "best_metrics": best_metrics,
        "total_combinations": count,
        "all_results": sorted(results, key=lambda x: x["f1"], reverse=True)[:20],
    }
    report_path = output_dir / "weight_optimization.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nRapor: {report_path}")


if __name__ == "__main__":
    main()
