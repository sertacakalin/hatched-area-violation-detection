"""YOLOv8 tabanlı plaka tespit wrapper.

Vehicle detector pattern'ı ile birebir simetrik. Aracın kırpılmış
görüntüsü içinde plaka bbox'ı bulur (genelde tek plaka, nadiren çoklu).
"""

import logging

import numpy as np
from ultralytics import YOLO

from src.core.data_models import Detection

logger = logging.getLogger(__name__)


class PlateDetector:
    """YOLOv8 tabanlı plaka tespit modülü."""

    def __init__(
        self,
        model_path: str = "weights/plate.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        half: bool = False,
        device: str = "auto",
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.half = half

        self.model = YOLO(model_path)
        if device != "auto":
            self.model.to(device)

        logger.info(f"Plaka tespit modeli yüklendi: {model_path}")

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Tek bir görüntüde plakaları tespit et."""
        if image is None or image.size == 0:
            return []
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            half=self.half,
            verbose=False,
        )
        return self._parse_result(results[0]) if results else []

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Birden fazla görüntüyü tek seferde işle (recognizer ring buffer için)."""
        valid_with_idx = [
            (i, img) for i, img in enumerate(images)
            if img is not None and img.size > 0
        ]
        if not valid_with_idx:
            return [[] for _ in images]

        valid_imgs = [img for _, img in valid_with_idx]
        results = self.model.predict(
            source=valid_imgs,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            half=self.half,
            verbose=False,
        )

        out: list[list[Detection]] = [[] for _ in images]
        for (orig_idx, _), result in zip(valid_with_idx, results):
            out[orig_idx] = self._parse_result(result)
        return out

    @staticmethod
    def _parse_result(result) -> list[Detection]:
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        return [
            Detection(
                bbox=bbox,
                confidence=float(conf),
                class_id=int(cls_id),
                class_name="plate",
            )
            for bbox, conf, cls_id in zip(boxes, confs, cls_ids)
        ]
