"""Tracker karşılaştırma deneyi — ByteTrack vs BoT-SORT vs DeepSORT.

Her tracker ile aynı videoyu işler, sonuçları karşılaştırır:
- Tespit edilen ihlal sayısı
- ID switch sayısı (takip tutarlılığı)
- FPS (hız)
- Sonuçların tutarlılığı

Kullanım:
    python scripts/compare_trackers.py --video data/videos/test/test_01.mp4
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from src.core.config import Config
from src.pipeline.pipeline import Pipeline


def run_with_tracker(video_path: str, tracker_type: str, zone_file: str,
                     config_path: str = "configs/config.yaml",
                     max_frames: int = 0) -> dict:
    """Tek bir tracker ile pipeline çalıştır."""
    config = Config(config_path)
    config._data["general"]["video_source"] = video_path
    config._data["tracking"]["tracker_type"] = tracker_type
    config._data["tracking"]["config_path"] = f"configs/{tracker_type}.yaml"
    config._data["zone"]["zone_file"] = zone_file
    config._data["general"]["save_video"] = False
    config._data["general"]["show_display"] = False
    config._data["vehicle_detection"]["half_precision"] = False

    pipeline = Pipeline(config)

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total = min(total, max_frames)

    violations = []
    track_ids_seen = set()
    frame_times = []

    for frame_num in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        tracked = pipeline.tracker.update(None, frame)
        tracked, new_violations = pipeline.violation_detector.process_frame(
            tracked, frame, frame_num, fps_video
        )
        frame_times.append(time.time() - t0)

        for obj in tracked:
            track_ids_seen.add(obj.track_id)

        for v in new_violations:
            violations.append({
                "frame_number": v.frame_number,
                "track_id": v.track_id,
                "vehicle_class": v.vehicle_class,
                "severity_score": v.severity_score,
                "violation_type": v.violation_type,
            })

    cap.release()

    avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

    return {
        "tracker": tracker_type,
        "frames_processed": len(frame_times),
        "total_violations": len(violations),
        "unique_tracks": len(track_ids_seen),
        "avg_fps": round(avg_fps, 2),
        "violations": violations,
    }


def main():
    parser = argparse.ArgumentParser(description="Tracker karşılaştırması")
    parser.add_argument("--video", required=True)
    parser.add_argument("--zone", default="configs/zones/e5_avcilar.json")
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Maksimum kare sayısı (0 = tümü)")
    parser.add_argument("--output", default="results/tracker_comparison")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    trackers = ["bytetrack", "botsort"]  # deepsort ek bağımlılık gerektirebilir

    all_results = []

    for tracker in trackers:
        print(f"\n{'='*50}")
        print(f"Tracker: {tracker}")
        print(f"{'='*50}")

        try:
            result = run_with_tracker(
                args.video, tracker, args.zone,
                max_frames=args.max_frames
            )
            all_results.append(result)
            print(f"  İhlal: {result['total_violations']}")
            print(f"  Benzersiz track: {result['unique_tracks']}")
            print(f"  FPS: {result['avg_fps']}")
        except Exception as e:
            print(f"  HATA: {e}")
            all_results.append({"tracker": tracker, "error": str(e)})

    # Karşılaştırma tablosu
    print(f"\n{'='*60}")
    print(f"{'Tracker':<12} {'İhlal':>8} {'Track':>8} {'FPS':>8}")
    print(f"{'='*60}")
    for r in all_results:
        if "error" in r:
            print(f"{r['tracker']:<12} {'HATA':>8}")
        else:
            print(f"{r['tracker']:<12} {r['total_violations']:>8} "
                  f"{r['unique_tracks']:>8} {r['avg_fps']:>8.1f}")

    # JSON kaydet
    report_path = output_dir / "tracker_comparison.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRapor: {report_path}")


if __name__ == "__main__":
    main()
