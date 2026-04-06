"""İhlal kayıt sistemi — veritabanı + dosya sistemi."""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.core.data_models import ViolationEvent
from src.alpr.plate_validator import PlateValidator
from src.storage.database import ViolationDatabase

logger = logging.getLogger(__name__)


class ViolationLogger:
    """İhlal olaylarını veritabanına ve dosya sistemine kaydeder."""

    def __init__(self, db_path: str = "results/violations.db",
                 output_dir: str = "results",
                 video_source: str = ""):
        self.db = ViolationDatabase(db_path)
        self.output_dir = Path(output_dir)
        self.crops_dir = self.output_dir / "crops"
        self.frames_dir = self.output_dir / "frames"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.video_source = video_source
        self.validator = PlateValidator()

    def log_violation(self, event: ViolationEvent) -> None:
        """İhlal olayını kaydet."""
        # Araç kırpmasını kaydet
        crop_path = None
        if event.vehicle_crop is not None and event.vehicle_crop.size > 0:
            crop_filename = f"violation_{event.event_id}_track{event.track_id}.jpg"
            crop_path = str(self.crops_dir / crop_filename)
            cv2.imwrite(crop_path, event.vehicle_crop)

        # Kare görüntüsünü kaydet
        frame_path = None
        if event.frame_image is not None:
            frame_filename = f"frame_{event.event_id}_f{event.frame_number}.jpg"
            frame_path = str(self.frames_dir / frame_filename)
            cv2.imwrite(frame_path, event.frame_image)

        # Plaka bilgisi
        plate_text = ""
        plate_raw = ""
        plate_conf = 0.0
        plate_valid = 0
        city_code = None
        city_name = None

        if event.plate:
            plate_text = event.plate.plate_text
            plate_raw = event.plate.raw_text
            plate_conf = event.plate.confidence
            plate_valid = 1 if event.plate.is_valid else 0

            if event.plate.is_valid:
                detail = self.validator.validate_detailed(plate_text)
                city_code = detail.get("city_code")
                city_name = detail.get("city_name")

        # Bbox'ı string olarak sakla
        bbox_str = ",".join(map(str, event.vehicle_bbox.astype(int).tolist()))

        # Trajectory metrikleri JSON
        traj_json = json.dumps(event.trajectory_metrics, ensure_ascii=False) if event.trajectory_metrics else None

        # Veritabanına kaydet
        self.db.insert_violation({
            "event_id": event.event_id,
            "track_id": event.track_id,
            "frame_number": event.frame_number,
            "timestamp_sec": event.timestamp,
            "vehicle_class": event.vehicle_class,
            "vehicle_confidence": event.vehicle_confidence,
            "vehicle_bbox": bbox_str,
            "zone_id": event.zone_id,
            "frames_in_zone": event.frames_in_zone,
            "plate_text": plate_text,
            "plate_raw": plate_raw,
            "plate_confidence": plate_conf,
            "plate_valid": plate_valid,
            "city_code": city_code,
            "city_name": city_name,
            "severity_score": event.severity_score,
            "severity_level": event.severity_level,
            "violation_type": event.violation_type,
            "trajectory_metrics": traj_json,
            "vehicle_crop_path": crop_path,
            "frame_image_path": frame_path,
            "video_source": self.video_source,
        })

        logger.info(
            f"İhlal kaydedildi: {event.event_id} | "
            f"Track: {event.track_id} | "
            f"Skor: {event.severity_score} ({event.severity_level}) | "
            f"Tip: {event.violation_type} | "
            f"Plaka: {plate_text or 'okunamadı'}"
        )

    def get_statistics(self) -> dict:
        return self.db.get_statistics()

    def close(self) -> None:
        self.db.close()
