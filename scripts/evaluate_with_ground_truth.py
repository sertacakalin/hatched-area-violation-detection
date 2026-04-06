"""Ground truth ile değerlendirme — Precision, Recall, F1 ve şiddet analizi.

Kullanım:
    python scripts/evaluate_with_ground_truth.py \
        --video data/videos/test/test_01.mp4 \
        --ground-truth data/ground_truth/test_01.json \
        --zone configs/zones/e5_avcilar.json

Ground truth formatı (JSON):
    {
        "video": "test_01.mp4",
        "fps": 30,
        "violations": [
            {
                "start_frame": 150,
                "end_frame": 200,
                "vehicle_class": "car",
                "type": "KAYNAK",
                "notes": "Beyaz araç sağdan sola şerit değiştirdi"
            },
            ...
        ]
    }
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.pipeline.pipeline_factory import create_pipeline

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_ground_truth(gt_path: str) -> dict:
    """Ground truth JSON dosyasını yükle."""
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_violations(predictions: list[dict], ground_truths: list[dict],
                     tolerance_frames: int = 30) -> dict:
    """Tahmin edilen ihlalleri ground truth ile eşleştir.

    Eşleştirme: Tahminin frame_number'ı, GT'nin [start_frame, end_frame]
    aralığına ±tolerance_frames ile düşüyorsa eşleşmiş sayılır.
    """
    gt_matched = [False] * len(ground_truths)
    pred_matched = [False] * len(predictions)

    tp_details = []

    for pi, pred in enumerate(predictions):
        pred_frame = pred["frame_number"]
        for gi, gt in enumerate(ground_truths):
            if gt_matched[gi]:
                continue
            gt_start = gt["start_frame"] - tolerance_frames
            gt_end = gt["end_frame"] + tolerance_frames
            if gt_start <= pred_frame <= gt_end:
                gt_matched[gi] = True
                pred_matched[pi] = True
                tp_details.append({
                    "pred": pred,
                    "gt": gt,
                    "frame_diff": pred_frame - gt["start_frame"],
                })
                break

    tp = sum(gt_matched)
    fp = sum(1 for m in pred_matched if not m)
    fn = sum(1 for m in gt_matched if not m)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp_details": tp_details,
        "fp_predictions": [p for p, m in zip(predictions, pred_matched) if not m],
        "fn_ground_truths": [g for g, m in zip(ground_truths, gt_matched) if not m],
    }


def severity_analysis(predictions: list[dict]) -> dict:
    """Şiddet skoru dağılım analizi."""
    if not predictions:
        return {}

    scores = [p["severity_score"] for p in predictions]
    types = [p.get("violation_type", "N/A") for p in predictions]
    levels = [p.get("severity_level", "N/A") for p in predictions]

    type_counts = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1

    level_counts = {}
    for l in levels:
        level_counts[l] = level_counts.get(l, 0) + 1

    return {
        "score_mean": round(float(np.mean(scores)), 1),
        "score_std": round(float(np.std(scores)), 1),
        "score_min": round(float(np.min(scores)), 1),
        "score_max": round(float(np.max(scores)), 1),
        "score_median": round(float(np.median(scores)), 1),
        "type_distribution": type_counts,
        "level_distribution": level_counts,
        "total_predictions": len(predictions),
    }


def false_positive_analysis(results: dict) -> dict:
    """False positive'lerin şiddet skoru analizi.

    Hipotez: Düşük şiddet skorlu ihlaller büyük olasılıkla FP'dir.
    """
    fp_preds = results.get("fp_predictions", [])
    tp_details = results.get("tp_details", [])

    if not fp_preds and not tp_details:
        return {}

    fp_scores = [p["severity_score"] for p in fp_preds]
    tp_scores = [d["pred"]["severity_score"] for d in tp_details]

    analysis = {}
    if fp_scores:
        analysis["fp_score_mean"] = round(float(np.mean(fp_scores)), 1)
        analysis["fp_score_median"] = round(float(np.median(fp_scores)), 1)
    if tp_scores:
        analysis["tp_score_mean"] = round(float(np.mean(tp_scores)), 1)
        analysis["tp_score_median"] = round(float(np.median(tp_scores)), 1)

    # Farklı eşik değerlerinde FP filtreleme etkisi
    if fp_scores or tp_scores:
        thresholds = [15, 25, 35, 50]
        analysis["threshold_impact"] = {}
        for t in thresholds:
            filtered_fp = sum(1 for s in fp_scores if s >= t)
            filtered_tp = sum(1 for s in tp_scores if s >= t)
            total_after = filtered_fp + filtered_tp
            new_precision = filtered_tp / total_after if total_after > 0 else 0
            analysis["threshold_impact"][f"min_score_{t}"] = {
                "remaining_fp": filtered_fp,
                "remaining_tp": filtered_tp,
                "new_precision": round(new_precision, 4),
            }

    return analysis


def run_pipeline_collect(config_path: str, video: str, zone: str,
                         tracker: str = "bytetrack") -> list[dict]:
    """Pipeline'ı çalıştır ve tüm ihlal olaylarını topla."""
    overrides = {
        "general.video_source": video,
        "zone.zone_file": zone,
        "tracking.tracker_type": tracker,
        "tracking.config_path": f"configs/{tracker}.yaml",
        "general.save_video": False,
        "general.show_display": False,
    }

    pipeline = create_pipeline(config_path, overrides)
    violations = []

    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked_objects = pipeline.tracker.update(None, frame)
        tracked_objects, new_violations = pipeline.violation_detector.process_frame(
            tracked_objects, frame, frame_num, fps
        )

        for v in new_violations:
            violations.append({
                "event_id": v.event_id,
                "track_id": v.track_id,
                "frame_number": v.frame_number,
                "timestamp": v.timestamp,
                "vehicle_class": v.vehicle_class,
                "vehicle_confidence": v.vehicle_confidence,
                "zone_id": v.zone_id,
                "frames_in_zone": v.frames_in_zone,
                "severity_score": v.severity_score,
                "severity_level": v.severity_level,
                "violation_type": v.violation_type,
                "trajectory_metrics": v.trajectory_metrics,
            })

        frame_num += 1
        if frame_num % 500 == 0:
            print(f"  Kare {frame_num}/{total} | İhlal: {len(violations)}")

    cap.release()
    return violations


def main():
    parser = argparse.ArgumentParser(description="Ground truth ile değerlendirme")
    parser.add_argument("--video", required=True, help="Test video dosyası")
    parser.add_argument("--ground-truth", required=True, help="Ground truth JSON")
    parser.add_argument("--zone", default="configs/zones/e5_avcilar.json")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tracker", default="bytetrack")
    parser.add_argument("--tolerance", type=int, default=30,
                        help="Eşleştirme toleransı (kare)")
    parser.add_argument("--output", default="results/evaluation",
                        help="Sonuç dizini")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ground truth yükle
    gt_data = load_ground_truth(args.ground_truth)
    gt_violations = gt_data["violations"]
    print(f"Ground truth: {len(gt_violations)} ihlal")

    # Pipeline çalıştır
    print(f"Pipeline çalıştırılıyor: {args.video}")
    t_start = time.time()
    predictions = run_pipeline_collect(
        args.config, args.video, args.zone, args.tracker
    )
    elapsed = time.time() - t_start
    print(f"Pipeline tamamlandı: {len(predictions)} ihlal, {elapsed:.1f} sn")

    # Eşleştirme + metrikler
    results = match_violations(predictions, gt_violations, args.tolerance)

    # Şiddet analizi
    sev_analysis = severity_analysis(predictions)

    # FP analizi
    fp_analysis = false_positive_analysis(results)

    # Sonuçları yazdır
    print("\n" + "=" * 60)
    print("DEĞERLENDIRME SONUÇLARI")
    print("=" * 60)
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1 Score  : {results['f1']:.4f}")
    print(f"  TP: {results['tp']}, FP: {results['fp']}, FN: {results['fn']}")

    print("\nŞiddet Skoru Dağılımı:")
    for k, v in sev_analysis.items():
        print(f"  {k}: {v}")

    if fp_analysis:
        print("\nFalse Positive Analizi:")
        for k, v in fp_analysis.items():
            print(f"  {k}: {v}")

    # JSON kaydet
    full_report = {
        "video": args.video,
        "tracker": args.tracker,
        "tolerance_frames": args.tolerance,
        "elapsed_sec": round(elapsed, 1),
        "metrics": {
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
            "tp": results["tp"],
            "fp": results["fp"],
            "fn": results["fn"],
        },
        "severity_analysis": sev_analysis,
        "fp_analysis": fp_analysis,
        "predictions": predictions,
        "ground_truth": gt_violations,
    }

    report_path = output_dir / f"eval_{Path(args.video).stem}_{args.tracker}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRapor kaydedildi: {report_path}")


if __name__ == "__main__":
    main()
