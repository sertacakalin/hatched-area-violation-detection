"""Tracker arayüzü ve factory (ByteTrack)."""

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


def create_tracker(tracker_type: str = "bytetrack", config_path: str | None = None,
                   model_path: str = "yolov8s.pt", **kwargs) -> BaseTracker:
    """Factory fonksiyonu — ByteTrack wrapper döndürür.

    Bitirme projesi kapsamında yalnızca ByteTrack kullanılır.
    """
    tracker_type = tracker_type.lower()

    if tracker_type != "bytetrack":
        raise ValueError(
            f"Desteklenmeyen tracker: {tracker_type}. "
            f"Bu projede sadece 'bytetrack' kullanılır."
        )

    from src.tracking.bytetrack_wrapper import ByteTrackWrapper
    return ByteTrackWrapper(config_path=config_path, model_path=model_path, **kwargs)
