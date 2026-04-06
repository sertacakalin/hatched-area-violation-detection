"""Veri modelleri — tüm pipeline bileşenleri arasında paylaşılan dataclass'lar."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class VehicleState(Enum):
    """Aracın taralı alan bölgesine göre durumu."""
    OUTSIDE = "OUTSIDE"
    ENTERING = "ENTERING"
    INSIDE = "INSIDE"
    VIOLATION = "VIOLATION"


@dataclass
class Detection:
    """Tek bir nesne tespiti."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        return (cx, cy)

    @property
    def bottom_center(self) -> tuple[float, float]:
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = self.bbox[3]
        return (cx, cy)

    @property
    def area(self) -> float:
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return max(0, w) * max(0, h)


@dataclass
class TrackedObject:
    """Takip edilen nesne (araç)."""
    track_id: int
    detection: Detection
    frames_tracked: int = 0
    state: VehicleState = VehicleState.OUTSIDE
    frames_in_zone: int = 0
    is_violation: bool = False

    @property
    def bbox(self) -> np.ndarray:
        return self.detection.bbox

    @property
    def center(self) -> tuple[float, float]:
        return self.detection.center

    @property
    def bottom_center(self) -> tuple[float, float]:
        return self.detection.bottom_center


@dataclass
class PlateResult:
    """Plaka okuma sonucu."""
    plate_text: str
    raw_text: str              # OCR ham çıktı (temizlenmemiş)
    confidence: float
    plate_bbox: Optional[np.ndarray] = None  # Plaka bbox'ı
    plate_image: Optional[np.ndarray] = None
    is_valid: bool = False     # Türk plaka formatına uygun mu


@dataclass
class ViolationEvent:
    """Tespit edilen bir ihlal olayı."""
    event_id: str
    track_id: int
    frame_number: int
    timestamp: float           # Videonun kaç. saniyesi
    vehicle_bbox: np.ndarray
    vehicle_class: str
    vehicle_confidence: float
    zone_id: str
    frames_in_zone: int
    plate: Optional[PlateResult] = None
    vehicle_crop: Optional[np.ndarray] = None
    frame_image: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
    # Yörünge + şiddet skorlama alanları
    severity_score: float = 0.0
    severity_level: str = ""
    violation_type: str = ""
    trajectory_metrics: dict = field(default_factory=dict)
