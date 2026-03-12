"""Görselleştirme — bbox, bölge, etiket çizimi."""

import cv2
import numpy as np

from src.core.data_models import TrackedObject, VehicleState, ViolationEvent


class Visualizer:
    """Video kareleri üzerine çizim yapan sınıf."""

    # Varsayılan renkler (BGR)
    COLOR_VEHICLE = (0, 255, 0)       # Yeşil
    COLOR_VIOLATION = (0, 0, 255)     # Kırmızı
    COLOR_ZONE = (255, 165, 0)        # Turuncu
    COLOR_PLATE = (255, 255, 0)       # Cyan
    COLOR_TEXT_BG = (0, 0, 0)         # Siyah

    def __init__(self, zone_alpha: float = 0.3, font_scale: float = 0.6,
                 thickness: int = 2):
        self.zone_alpha = zone_alpha
        self.font_scale = font_scale
        self.thickness = thickness

    def draw_zone(self, frame: np.ndarray, polygon: np.ndarray,
                  color: tuple = None, label: str = None) -> np.ndarray:
        """Yarı şeffaf polygon çiz."""
        color = color or self.COLOR_ZONE
        overlay = frame.copy()
        pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], True, color, self.thickness)
        frame = cv2.addWeighted(overlay, self.zone_alpha, frame,
                                1 - self.zone_alpha, 0)
        if label:
            x, y = pts[0][0]
            self._put_label(frame, label, (x, y - 10), color)
        return frame

    def draw_tracked_object(self, frame: np.ndarray, obj: TrackedObject) -> np.ndarray:
        """Takip edilen aracı çiz (bbox + ID + sınıf)."""
        is_violation = obj.state == VehicleState.VIOLATION
        color = self.COLOR_VIOLATION if is_violation else self.COLOR_VEHICLE
        x1, y1, x2, y2 = obj.bbox.astype(int)

        # Bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

        # Etiket
        label = f"ID:{obj.track_id} {obj.detection.class_name}"
        if is_violation:
            label += " IHLAL"
        label += f" {obj.detection.confidence:.2f}"
        self._put_label(frame, label, (x1, y1 - 5), color)

        # Alt merkez noktası (polygon kontrolü için)
        cx, cy = obj.bottom_center
        cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)

        return frame

    def draw_violation_event(self, frame: np.ndarray,
                             event: ViolationEvent) -> np.ndarray:
        """İhlal olayını çiz (bbox + plaka bilgisi)."""
        x1, y1, x2, y2 = event.vehicle_bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_VIOLATION, 3)

        # Plaka bilgisi
        if event.plate and event.plate.plate_text:
            plate_label = f"PLAKA: {event.plate.plate_text} ({event.plate.confidence:.2f})"
            self._put_label(frame, plate_label, (x1, y2 + 20), self.COLOR_PLATE)

        return frame

    def draw_info_panel(self, frame: np.ndarray, frame_num: int,
                        fps: float, violation_count: int,
                        tracked_count: int) -> np.ndarray:
        """Sol üst köşeye bilgi paneli çiz."""
        info_lines = [
            f"Kare: {frame_num}",
            f"FPS: {fps:.1f}",
            f"Takip: {tracked_count}",
            f"Ihlal: {violation_count}",
        ]
        y_offset = 25
        for line in info_lines:
            self._put_label(frame, line, (10, y_offset), self.COLOR_VEHICLE)
            y_offset += 25
        return frame

    def _put_label(self, frame: np.ndarray, text: str,
                   position: tuple, color: tuple) -> None:
        """Arka planlı metin yaz."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, self.font_scale, 1)
        x, y = int(position[0]), int(position[1])
        cv2.rectangle(frame, (x, y - th - baseline), (x + tw, y + baseline),
                      self.COLOR_TEXT_BG, -1)
        cv2.putText(frame, text, (x, y), font, self.font_scale, color, 1,
                    cv2.LINE_AA)
