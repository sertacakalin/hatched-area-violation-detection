"""BoT-SORT wrapper — Ultralytics entegre tracker."""

import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.core.data_models import Detection, TrackedObject
from src.tracking.tracker import BaseTracker

logger = logging.getLogger(__name__)


class BoTSORTWrapper(BaseTracker):
    """Ultralytics üzerinden BoT-SORT takip."""

    def __init__(self, config_path: str | None = None,
                 model_path: str = "yolov8s.pt",
                 conf: float = 0.35, iou: float = 0.45,
                 classes: list[int] | None = None,
                 img_size: int = 640, half: bool = True, **kwargs):
        self.config_path = config_path or str(
            Path(__file__).parent.parent.parent / "configs" / "botsort.yaml"
        )
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.classes = classes or [2, 3, 5, 7]
        self.img_size = img_size
        self.half = half
        logger.info(f"BoT-SORT başlatıldı: config={self.config_path}")

    def update(self, detections: list[Detection] | None,
               frame: np.ndarray) -> list[TrackedObject]:
        from src.detection.vehicle_detector import COCO_VEHICLE_CLASSES

        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            imgsz=self.img_size,
            half=self.half,
            tracker=self.config_path,
            persist=True,
            verbose=False,
        )

        tracked_objects = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)

                for bbox, tid, conf, cls_id in zip(boxes, track_ids, confs, cls_ids):
                    class_name = COCO_VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
                    det = Detection(
                        bbox=bbox,
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=class_name,
                    )
                    obj = TrackedObject(track_id=int(tid), detection=det)
                    tracked_objects.append(obj)

        return tracked_objects

    def reset(self) -> None:
        self.model = YOLO(self.model.model_name)
