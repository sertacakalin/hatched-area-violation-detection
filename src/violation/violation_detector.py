"""İhlal tespit mantığı — zone kontrolü + temporal filtre + yörünge + şiddet skorlama."""

import logging
import uuid
from typing import Optional

import numpy as np

from src.core.data_models import (
    TrackedObject, VehicleState, ViolationEvent,
)
from src.violation.state_machine import VehicleStateMachine
from src.violation.trajectory import TrajectoryAnalyzer
from src.violation.severity import SeverityScorer
from src.zones.zone_manager import ZoneManager

logger = logging.getLogger(__name__)


class ViolationDetector:
    """Taralı alan ihlallerini tespit eden ana modül.

    Pipeline'daki rolü:
    1. Takip edilen araçları al
    2. Her aracın bölge içinde olup olmadığını kontrol et
    3. State machine ile durum geçişlerini yönet
    4. İhlal tetiklendiğinde ViolationEvent oluştur
    """

    def __init__(
        self,
        zone_manager: ZoneManager,
        min_frames_in_zone: int = 5,
        cooldown_frames: int = 90,
        min_overlap_ratio: float = 0.3,
        use_bottom_center: bool = True,
        severity_weights: dict | None = None,
    ):
        self.zone_manager = zone_manager
        self.min_overlap_ratio = min_overlap_ratio
        self.use_bottom_center = use_bottom_center
        self.state_machine = VehicleStateMachine(
            min_frames_in_zone=min_frames_in_zone,
            cooldown_frames=cooldown_frames,
        )
        self.trajectory_analyzer = TrajectoryAnalyzer()
        weights = severity_weights or {}
        self.severity_scorer = SeverityScorer(**weights)
        self._violation_count = 0
        self._severity_scores: list[float] = []
        self._violation_types: dict[str, int] = {}
        self._severity_levels: dict[str, int] = {}

    def process_frame(
        self,
        tracked_objects: list[TrackedObject],
        frame: np.ndarray,
        frame_number: int,
        fps: float = 30.0,
    ) -> tuple[list[TrackedObject], list[ViolationEvent]]:
        """Bir karedeki tüm takip edilen araçları işle.

        Returns:
            (güncellenen tracked_objects, yeni ihlal olayları)
        """
        new_violations = []
        active_ids = set()

        for obj in tracked_objects:
            active_ids.add(obj.track_id)

            # Bölge kontrolü — iki yöntem
            is_in_zone = False
            zone_id = None

            if self.use_bottom_center:
                # Yöntem 1: Alt merkez noktası polygon içinde mi
                is_in_zone, zone_id = self.zone_manager.is_point_in_zone(
                    obj.bottom_center
                )
            else:
                # Yöntem 2: Bbox örtüşme oranı eşiği
                ratio, zone_id = self.zone_manager.get_bbox_overlap_ratio(obj.bbox)
                is_in_zone = ratio >= self.min_overlap_ratio

            # Yörünge takibi güncelle
            self.trajectory_analyzer.update(
                obj.track_id, obj.bottom_center, is_in_zone
            )

            # State machine güncelle
            new_state, is_new_violation = self.state_machine.update(
                obj.track_id, is_in_zone, zone_id
            )

            # TrackedObject durumunu güncelle
            obj.state = new_state
            track_state = self.state_machine.get_state(obj.track_id)
            obj.frames_in_zone = track_state.frames_in_zone
            obj.is_violation = new_state == VehicleState.VIOLATION

            # Yeni ihlal olayı oluştur
            if is_new_violation:
                self._violation_count += 1
                event = self._create_violation_event(
                    obj, frame, frame_number, fps, zone_id
                )
                new_violations.append(event)
                # İstatistik takibi
                self._severity_scores.append(event.severity_score)
                vtype = event.violation_type
                self._violation_types[vtype] = self._violation_types.get(vtype, 0) + 1
                slevel = event.severity_level
                self._severity_levels[slevel] = self._severity_levels.get(slevel, 0) + 1
                logger.info(
                    f"IHLAL #{self._violation_count}: "
                    f"Track {obj.track_id}, {obj.detection.class_name}, "
                    f"Zone: {zone_id}, Kare: {frame_number}, "
                    f"Skor: {event.severity_score}, "
                    f"Tip: {event.violation_type}"
                )

        # Eski track'leri temizle
        self.state_machine.cleanup_stale_tracks(active_ids)
        self.trajectory_analyzer.cleanup_stale_tracks(active_ids)

        return tracked_objects, new_violations

    def _create_violation_event(
        self,
        obj: TrackedObject,
        frame: np.ndarray,
        frame_number: int,
        fps: float,
        zone_id: Optional[str],
    ) -> ViolationEvent:
        """İhlal olayı oluştur (araç kırpması + yörünge + şiddet dahil)."""
        # Araç kırpması
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = obj.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        vehicle_crop = frame[y1:y2, x1:x2].copy()

        # Yörünge metrikleri hesapla
        zone_polygon = self._get_zone_polygon(zone_id)
        traj_metrics = self.trajectory_analyzer.compute_metrics(
            obj.track_id, zone_polygon
        )

        # Şiddet skoru hesapla
        severity_result = self.severity_scorer.score(traj_metrics)

        event = ViolationEvent(
            event_id=str(uuid.uuid4())[:8],
            track_id=obj.track_id,
            frame_number=frame_number,
            timestamp=frame_number / fps if fps > 0 else 0,
            vehicle_bbox=obj.bbox.copy(),
            vehicle_class=obj.detection.class_name,
            vehicle_confidence=obj.detection.confidence,
            zone_id=zone_id or "unknown",
            frames_in_zone=obj.frames_in_zone,
            vehicle_crop=vehicle_crop,
            frame_image=frame.copy(),
            severity_score=severity_result.score,
            severity_level=severity_result.level.value,
            violation_type=severity_result.violation_type.value,
            trajectory_metrics=severity_result.components,
        )
        return event

    def _get_zone_polygon(self, zone_id: Optional[str]):
        """Zone polygon'unu bul (yörünge hesaplaması için)."""
        from shapely.geometry import Polygon as ShapelyPolygon
        for zone in self.zone_manager.zones:
            if zone.zone_id == zone_id:
                return zone.polygon
        # Fallback: ilk zone
        if self.zone_manager.zones:
            return self.zone_manager.zones[0].polygon
        # Son çare: dummy polygon
        return ShapelyPolygon([(0, 0), (100, 0), (100, 100), (0, 100)])

    @property
    def violation_count(self) -> int:
        return self._violation_count

    def get_severity_statistics(self) -> dict:
        """Tüm ihlallerin şiddet istatistikleri (tez tablosu için)."""
        scores = self._severity_scores
        return {
            "total": self._violation_count,
            "score_mean": round(sum(scores) / len(scores), 1) if scores else 0,
            "score_min": round(min(scores), 1) if scores else 0,
            "score_max": round(max(scores), 1) if scores else 0,
            "score_std": round(
                (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5, 1
            ) if scores else 0,
            "by_type": dict(self._violation_types),
            "by_level": dict(self._severity_levels),
        }

    def reset(self) -> None:
        self.state_machine.reset()
        self.trajectory_analyzer.reset()
        self._violation_count = 0
        self._severity_scores.clear()
        self._violation_types.clear()
        self._severity_levels.clear()
