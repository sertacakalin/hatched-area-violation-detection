"""YOLOv8 araç tespiti wrapper."""

import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.core.data_models import Detection

logger = logging.getLogger(__name__)

# COCO sınıf isimleri (araç ile ilgili olanlar)
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    """YOLOv8 tabanlı araç tespit modülü."""

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None,
        img_size: int = 640,
        half: bool = True,
        device: str = "auto",
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes or list(COCO_VEHICLE_CLASSES.keys())
        self.img_size = img_size
        self.half = half

        # Model yükle
        self.model = YOLO(model_path)
        if device != "auto":
            self.model.to(device)

        logger.info(f"Araç tespit modeli yüklendi: {model_path}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Tek bir karedeki araçları tespit et."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.img_size,
            half=self.half,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)

                for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
                    class_name = COCO_VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
                    det = Detection(
                        bbox=bbox,
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=class_name,
                    )
                    detections.append(det)

        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Birden fazla kareyi toplu tespit et."""
        return [self.detect(frame) for frame in frames]

    def get_raw_results(self, frame: np.ndarray):
        """Ultralytics Results objesini doğrudan döndür (tracking için)."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.img_size,
            half=self.half,
            verbose=False,
        )
        return results[0] if results else None
