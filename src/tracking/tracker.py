"""Birleşik tracker arayüzü — ByteTrack, BoT-SORT, DeepSORT."""

from abc import ABC, abstractmethod

import numpy as np

from src.core.data_models import Detection, TrackedObject


class BaseTracker(ABC):
    """Tüm tracker implementasyonları için temel sınıf."""

    @abstractmethod
    def update(self, detections: list[Detection], frame: np.ndarray) -> list[TrackedObject]:
        """Tespitleri takipçiye gönder ve takip edilen nesneleri döndür."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Tracker durumunu sıfırla."""
        ...


def create_tracker(tracker_type: str, config_path: str | None = None,
                   model_path: str = "yolov8s.pt", **kwargs) -> BaseTracker:
    """Factory fonksiyonu — tracker tipine göre uygun wrapper'ı oluştur."""
    tracker_type = tracker_type.lower()

    if tracker_type == "bytetrack":
        from src.tracking.bytetrack_wrapper import ByteTrackWrapper
        return ByteTrackWrapper(config_path=config_path, model_path=model_path, **kwargs)
    elif tracker_type == "botsort":
        from src.tracking.botsort_wrapper import BoTSORTWrapper
        return BoTSORTWrapper(config_path=config_path, model_path=model_path, **kwargs)
    elif tracker_type == "deepsort":
        from src.tracking.deepsort_wrapper import DeepSORTWrapper
        return DeepSORTWrapper(config_path=config_path, model_path=model_path, **kwargs)
    else:
        raise ValueError(f"Bilinmeyen tracker tipi: {tracker_type}. "
                         f"Desteklenen: bytetrack, botsort, deepsort")
