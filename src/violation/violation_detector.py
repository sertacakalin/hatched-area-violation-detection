"""İhlal tespit mantığı — zone kontrolü + temporal filtre + olay oluşturma."""

import logging
import uuid
from typing import Optional

import numpy as np

from src.core.data_models import (
    TrackedObject, VehicleState, ViolationEvent,
)
from src.violation.state_machine import VehicleStateMachine
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
    ):
        self.zone_manager = zone_manager
        self.min_overlap_ratio = min_overlap_ratio
        self.use_bottom_center = use_bottom_center
        self.state_machine = VehicleStateMachine(
            min_frames_in_zone=min_frames_in_zone,
            cooldown_frames=cooldown_frames,
        )
        self._violation_count = 0

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
                logger.info(
                    f"IHLAL #{self._violation_count}: "
                    f"Track {obj.track_id}, {obj.detection.class_name}, "
                    f"Zone: {zone_id}, Kare: {frame_number}"
                )

        # Eski track'leri temizle
        self.state_machine.cleanup_stale_tracks(active_ids)

        return tracked_objects, new_violations

    def _create_violation_event(
        self,
        obj: TrackedObject,
        frame: np.ndarray,
        frame_number: int,
        fps: float,
        zone_id: Optional[str],
    ) -> ViolationEvent:
        """İhlal olayı oluştur (araç kırpması dahil)."""
        # Araç kırpması
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = obj.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        vehicle_crop = frame[y1:y2, x1:x2].copy()

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
        )
        return event

    @property
    def violation_count(self) -> int:
        return self._violation_count

    def reset(self) -> None:
        self.state_machine.reset()
        self._violation_count = 0
