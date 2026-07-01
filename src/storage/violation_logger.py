"""İhlal kayıt sistemi — veritabanı + dosya sistemi."""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.core.data_models import ViolationEvent
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
        self.plates_dir = self.output_dir / "plates"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.plates_dir.mkdir(parents=True, exist_ok=True)
        self.video_source = video_source

    def log_violation(self, event: ViolationEvent) -> None:
        """İhlal olayını kaydet (araç + plaka + kare + DB)."""
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

        # Plaka kırpmasını kaydet (varsa) — yolu DB'ye plate_crop_path olarak yazıyoruz
        plate = event.plate
        plate_crop_path: str | None = None
        if plate is not None and plate.plate_image is not None and plate.plate_image.size > 0:
            plate_filename = f"plate_{event.event_id}_track{event.track_id}.jpg"
            plate_crop_path = str(self.plates_dir / plate_filename)
            cv2.imwrite(plate_crop_path, plate.plate_image)

        # Tam kare artık diskte — in-memory kopyayı bırak. Aksi halde bu event
        # pipeline.events / all_violations listelerinde tutulduğu için her ihlal
        # bir tam kare kadar RAM'i (1080p'de ~6 MB) kalıcı işgal eder. Küçük olan
        # vehicle_crop/plate_image'a dokunmuyoruz: on_violation callback'leri
        # (ör. live_alert kanıt fotosu) log_violation'dan SONRA bunları kullanıyor.
        event.frame_image = None

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
            "plate_text":      plate.plate_text if plate else None,
            "plate_raw":       plate.raw_text if plate else None,
            "plate_confidence": plate.confidence if plate else None,
            "plate_valid":     1 if (plate and plate.is_valid) else 0,
            "city_code":       plate.city_code if plate else None,
            "city_name":       plate.city_name if plate else None,
            "severity_score": event.severity_score,
            "severity_level": event.severity_level,
            "violation_type": event.violation_type,
            "trajectory_metrics": traj_json,
            "vehicle_crop_path": crop_path,
            "plate_crop_path": plate_crop_path,
            "frame_image_path": frame_path,
            "video_source": self.video_source,
        })

        plate_info = ""
        if plate is not None:
            valid_mark = "✓" if plate.is_valid else "✗"
            plate_info = (
                f" | Plaka: {plate.plate_text or '?'} "
                f"({plate.confidence:.2f}, {valid_mark})"
            )
            if plate.city_name:
                plate_info += f" [{plate.city_name}]"

        logger.info(
            f"İhlal kaydedildi: {event.event_id} | "
            f"Track: {event.track_id} | "
            f"Skor: {event.severity_score} ({event.severity_level}) | "
            f"Tip: {event.violation_type}"
            f"{plate_info}"
        )

    def get_statistics(self) -> dict:
        return self.db.get_statistics()

    def close(self) -> None:
        self.db.close()
