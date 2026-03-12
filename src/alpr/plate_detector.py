"""YOLOv8n plaka tespiti — araç kırpmasından plaka bölgesini bulur."""

import logging

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class PlateDetector:
    """YOLOv8 tabanlı plaka tespit modülü."""

    def __init__(
        self,
        model_path: str = "weights/plate_detector.pt",
        confidence_threshold: float = 0.5,
        img_size: int = 640,
        device: str = "auto",
    ):
        self.confidence_threshold = confidence_threshold
        self.img_size = img_size

        try:
            self.model = YOLO(model_path)
            if device != "auto":
                self.model.to(device)
            logger.info(f"Plaka tespit modeli yüklendi: {model_path}")
        except Exception as e:
            logger.error(f"Plaka modeli yüklenemedi: {e}")
            self.model = None

    def detect(self, vehicle_crop: np.ndarray) -> list[dict]:
        """Araç kırpmasından plaka bölgelerini tespit et.

        Returns:
            [{"bbox": [x1,y1,x2,y2], "confidence": float, "crop": ndarray}, ...]
        """
        if self.model is None or vehicle_crop.size == 0:
            return []

        results = self.model.predict(
            source=vehicle_crop,
            conf=self.confidence_threshold,
            imgsz=self.img_size,
            verbose=False,
        )

        plates = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for bbox, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = bbox.astype(int)
                    h, w = vehicle_crop.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    plate_crop = vehicle_crop[y1:y2, x1:x2]

                    if plate_crop.size > 0:
                        plates.append({
                            "bbox": bbox,
                            "confidence": float(conf),
                            "crop": plate_crop,
                        })

        # En yüksek güvenli plaka
        plates.sort(key=lambda p: p["confidence"], reverse=True)
        return plates

    def detect_from_frame(self, frame: np.ndarray,
                          vehicle_bbox: np.ndarray) -> list[dict]:
        """Tam kareden araç bölgesini kırp ve plaka tespit et."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = vehicle_bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        vehicle_crop = frame[y1:y2, x1:x2]

        plates = self.detect(vehicle_crop)

        # Global koordinatlara dönüştür
        for plate in plates:
            plate["bbox"][0] += x1
            plate["bbox"][1] += y1
            plate["bbox"][2] += x1
            plate["bbox"][3] += y1

        return plates
