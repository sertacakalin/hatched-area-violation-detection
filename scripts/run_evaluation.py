"""Değerlendirme script'i — birden fazla konfigürasyonla deney çalıştır."""

import argparse
import json
import sys
import time
from pathlib import Path
from itertools import product

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.pipeline_factory import create_pipeline


def run_experiment(config_path: str, video: str, tracker: str,
                   model: str, output_dir: str) -> dict:
    """Tek bir deney çalıştır."""
    overrides = {
        "general.video_source": video,
        "tracking.tracker_type": tracker,
        "tracking.config_path": f"configs/{tracker}.yaml",
        "vehicle_detection.model_path": model,
        "general.output_dir": output_dir,
        "general.save_video": False,
        "general.show_display": False,
    }

    t_start = time.time()
    pipeline = create_pipeline(config_path, overrides)
    results = pipeline.run()
    elapsed = time.time() - t_start

    results["tracker"] = tracker
    results["model"] = model
    results["video"] = video
    results["elapsed_sec"] = elapsed

    return results


def main():
    parser = argparse.ArgumentParser(description="Deney seti çalıştır")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--videos", nargs="+", required=True,
                        help="Test video dosyaları")
    parser.add_argument("--trackers", nargs="+",
                        default=["bytetrack", "botsort", "deepsort"],
                        help="Test edilecek tracker'lar")
    parser.add_argument("--models", nargs="+",
                        default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                        help="Test edilecek modeller")
    parser.add_argument("--output", default="results/experiments",
                        help="Sonuç dizini")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total = len(args.videos) * len(args.trackers) * len(args.models)
    idx = 0

    for video, tracker, model in product(args.videos, args.trackers, args.models):
        idx += 1
        exp_name = f"{Path(video).stem}_{tracker}_{Path(model).stem}"
        exp_dir = str(output_dir / exp_name)
        Path(exp_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx}/{total}] Deney: {exp_name}")
        print(f"  Video: {video}")
        print(f"  Tracker: {tracker}")
        print(f"  Model: {model}")

        try:
            results = run_experiment(
                args.config, video, tracker, model, exp_dir
            )
            all_results.append(results)

            # Ara sonuçları kaydet
            with open(output_dir / f"{exp_name}.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"  Sonuç: {results.get('total_violations', 0)} ihlal, "
                  f"FPS: {results.get('average_fps', 0):.1f}")

        except Exception as e:
            print(f"  HATA: {e}")
            all_results.append({
                "video": video, "tracker": tracker, "model": model,
                "error": str(e),
            })

    # Genel karşılaştırma tablosu
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSonuçlar kaydedildi: {csv_path}")
    print("\n" + df.to_string())


if __name__ == "__main__":
    main()
