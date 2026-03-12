"""DeepSORT wrapper — Ultralytics entegre tracker (karşılaştırma için)."""

import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.core.data_models import Detection, TrackedObject
from src.tracking.tracker import BaseTracker

logger = logging.getLogger(__name__)


class DeepSORTWrapper(BaseTracker):
    """Ultralytics üzerinden DeepSORT takip.

    Not: DeepSORT, Ultralytics'te doğrudan desteklenmez.
    Bunun yerine BoT-SORT'u ReID ile kullanırız (benzer davranış).
    Alternatif olarak deep-sort-realtime paketi kullanılabilir.
    """

    def __init__(self, config_path: str | None = None,
                 model_path: str = "yolov8s.pt",
                 conf: float = 0.35, iou: float = 0.45,
                 classes: list[int] | None = None,
                 img_size: int = 640, half: bool = True, **kwargs):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.classes = classes or [2, 3, 5, 7]
        self.img_size = img_size
        self.half = half
        self._tracker = None

        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._tracker = DeepSort(
                max_age=45,
                n_init=3,
                max_cosine_distance=0.3,
                nn_budget=100,
            )
            self._use_native = True
            logger.info("DeepSORT (deep-sort-realtime) başlatıldı")
        except ImportError:
            logger.warning("deep-sort-realtime bulunamadı, BoT-SORT+ReID ile simüle edilecek")
            self._use_native = False
            self._botsort_config = config_path or str(
                Path(__file__).parent.parent.parent / "configs" / "botsort.yaml"
            )
            self.model = YOLO(model_path)

    def update(self, detections: list[Detection] | None,
               frame: np.ndarray) -> list[TrackedObject]:
        if self._use_native:
            return self._update_native(detections, frame)
        return self._update_fallback(frame)

    def _update_native(self, detections: list[Detection],
                       frame: np.ndarray) -> list[TrackedObject]:
        """deep-sort-realtime ile takip."""
        if not detections:
            detections = []

        # DeepSort formatına dönüştür: [x1, y1, w, h], conf, class_name
        ds_detections = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w, h = x2 - x1, y2 - y1
            ds_detections.append(([x1, y1, w, h], det.confidence, det.class_name))

        tracks = self._tracker.update_tracks(ds_detections, frame=frame)

        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            det = Detection(
                bbox=np.array(ltrb),
                confidence=track.det_conf if track.det_conf else 0.0,
                class_id=0,
                class_name=track.det_class if track.det_class else "vehicle",
            )
            obj = TrackedObject(track_id=track.track_id, detection=det)
            tracked_objects.append(obj)

        return tracked_objects

    def _update_fallback(self, frame: np.ndarray) -> list[TrackedObject]:
        """BoT-SORT fallback."""
        from src.detection.vehicle_detector import COCO_VEHICLE_CLASSES

        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            imgsz=self.img_size,
            half=self.half,
            tracker=self._botsort_config,
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
                        bbox=bbox, confidence=float(conf),
                        class_id=int(cls_id), class_name=class_name,
                    )
                    obj = TrackedObject(track_id=int(tid), detection=det)
                    tracked_objects.append(obj)

        return tracked_objects

    def reset(self) -> None:
        if self._use_native and self._tracker:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._tracker = DeepSort(
                max_age=45, n_init=3,
                max_cosine_distance=0.3, nn_budget=100,
            )
        elif not self._use_native:
            self.model = YOLO(self.model_path)
