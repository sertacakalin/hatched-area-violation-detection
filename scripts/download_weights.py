"""Model ağırlıklarını indir."""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_yolov8_models():
    """YOLOv8 pretrained modellerini indir."""
    from ultralytics import YOLO

    models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    for model_name in models:
        logger.info(f"İndiriliyor: {model_name}")
        model = YOLO(model_name)
        logger.info(f"Hazır: {model_name}")


def download_plate_model():
    """Plaka tespit modelini indir (Roboflow'dan).

    Alternatif: Kendi eğittiğiniz modeli weights/ klasörüne kopyalayın.
    """
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    plate_model_path = weights_dir / "plate_detector.pt"

    if plate_model_path.exists():
        logger.info(f"Plaka modeli zaten mevcut: {plate_model_path}")
        return

    logger.info("Plaka modeli için seçenekler:")
    logger.info("  1. Roboflow'dan indirin: https://universe.roboflow.com/search?q=license+plate")
    logger.info("  2. Kendi modelinizi eğitin ve weights/plate_detector.pt olarak kaydedin")
    logger.info("  3. Aşağıdaki kodu kullanarak Roboflow API ile indirin:")
    logger.info("")
    logger.info("     from roboflow import Roboflow")
    logger.info("     rf = Roboflow(api_key='YOUR_API_KEY')")
    logger.info("     project = rf.workspace().project('license-plate-recognition-rxg4e')")
    logger.info("     model = project.version(4).model")


def main():
    parser = argparse.ArgumentParser(description="Model ağırlıklarını indir")
    parser.add_argument("--all", action="store_true", help="Tüm modelleri indir")
    parser.add_argument("--vehicle", action="store_true", help="Araç modellerini indir")
    parser.add_argument("--plate", action="store_true", help="Plaka modelini indir")
    args = parser.parse_args()

    if args.all or args.vehicle or not any(vars(args).values()):
        download_yolov8_models()

    if args.all or args.plate:
        download_plate_model()

    logger.info("İndirme tamamlandı")


if __name__ == "__main__":
    main()
