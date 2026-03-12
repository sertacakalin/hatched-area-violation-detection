"""Pipeline çalıştırma script'i."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.pipeline_factory import create_pipeline


def main():
    parser = argparse.ArgumentParser(description="İhlal tespit pipeline'ını çalıştır")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Config dosya yolu")
    parser.add_argument("--video", help="Video dosya yolu (config override)")
    parser.add_argument("--zone", help="Bölge JSON dosya yolu (config override)")
    parser.add_argument("--tracker", choices=["bytetrack", "botsort", "deepsort"],
                        help="Tracker tipi (config override)")
    parser.add_argument("--output", help="Çıktı dizini (config override)")
    parser.add_argument("--show", action="store_true", help="Canlı görüntü göster")
    parser.add_argument("--no-save", action="store_true", help="Video kaydetme")
    args = parser.parse_args()

    # Override'ları oluştur
    overrides = {}
    if args.video:
        overrides["general.video_source"] = args.video
    if args.zone:
        overrides["zone.zone_file"] = args.zone
    if args.tracker:
        overrides["tracking.tracker_type"] = args.tracker
        overrides["tracking.config_path"] = f"configs/{args.tracker}.yaml"
    if args.output:
        overrides["general.output_dir"] = args.output
    if args.show:
        overrides["general.show_display"] = True
    if args.no_save:
        overrides["general.save_video"] = False

    # Pipeline oluştur ve çalıştır
    pipeline = create_pipeline(args.config, overrides)
    results = pipeline.run()

    # Sonuçları yazdır
    print("\n" + "=" * 50)
    print("SONUÇLAR")
    print("=" * 50)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
