"""Ana orkestratör — tüm bileşenleri birleştiren pipeline."""

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.core.config import Config
from src.core.frame_provider import FrameProvider
from src.core.data_models import ViolationEvent
from src.core.visualizer import Visualizer
from src.detection.vehicle_detector import VehicleDetector
from src.tracking.tracker import create_tracker
from src.zones.zone_manager import ZoneManager
from src.violation.violation_detector import ViolationDetector
from src.alpr.plate_detector import PlateDetector
from src.alpr.plate_reader import PlateReader
from src.storage.violation_logger import ViolationLogger

logger = logging.getLogger(__name__)


class Pipeline:
    """Uçtan uca ihlal tespit pipeline'ı.

    Akış:
    Video → Kare → Araç Tespiti → Takip → ROI Kontrolü → İhlal Tespiti
    → Plaka Tespiti → OCR → Doğrulama → Kayıt → Görselleştirme
    """

    def __init__(self, config: Config):
        self.config = config

        # Bileşenleri başlat
        logger.info("Pipeline bileşenleri başlatılıyor...")

        # Video
        self.frame_provider = FrameProvider(
            config.get("general.video_source")
        )

        # Araç Tespiti + Takip
        self.tracker = create_tracker(
            tracker_type=config.get("tracking.tracker_type", "bytetrack"),
            config_path=config.get("tracking.config_path"),
            model_path=config.get("vehicle_detection.model_path", "yolov8s.pt"),
            conf=config.get("vehicle_detection.confidence_threshold", 0.35),
            iou=config.get("vehicle_detection.iou_threshold", 0.45),
            classes=config.get("vehicle_detection.classes", [2, 3, 5, 7]),
            img_size=config.get("vehicle_detection.img_size", 640),
            half=config.get("vehicle_detection.half_precision", True),
        )

        # Bölge Yönetimi
        self.zone_manager = ZoneManager(
            zone_file=config.get("zone.zone_file"),
            polygon_buffer=config.get("zone.polygon_buffer", -10),
        )

        # İhlal Tespiti
        self.violation_detector = ViolationDetector(
            zone_manager=self.zone_manager,
            min_frames_in_zone=config.get("violation.min_frames_in_zone", 5),
            cooldown_frames=config.get("violation.cooldown_frames", 90),
            min_overlap_ratio=config.get("zone.min_overlap_ratio", 0.3),
        )

        # Plaka Tespiti + OCR
        self.plate_detector = PlateDetector(
            model_path=config.get("plate_detection.model_path", "weights/plate_detector.pt"),
            confidence_threshold=config.get("plate_detection.confidence_threshold", 0.5),
        )

        ocr_engine = config.get("plate_ocr.engine", "paddleocr")
        self.plate_reader = PlateReader(
            engine=ocr_engine,
            min_confidence=config.get("plate_validation.min_confidence", 0.6),
        )

        # Kayıt
        self.violation_logger = ViolationLogger(
            db_path=config.get("database.path", "results/violations.db"),
            output_dir=config.get("general.output_dir", "results"),
            video_source=str(config.get("general.video_source", "")),
        )

        # Görselleştirme
        self.visualizer = Visualizer(
            zone_alpha=config.get("visualization.zone_alpha", 0.3),
            font_scale=config.get("visualization.font_scale", 0.6),
            thickness=config.get("visualization.thickness", 2),
        )

        # Video writer
        self._video_writer: cv2.VideoWriter | None = None
        self._total_violations = 0

        logger.info("Pipeline hazır")

    def run(self) -> dict:
        """Pipeline'ı çalıştır ve sonuçları döndür."""
        save_video = self.config.get("general.save_video", True)
        show_display = self.config.get("general.show_display", False)

        with self.frame_provider as fp:
            fps = fp.fps
            total = fp.total_frames

            # Video writer
            if save_video:
                output_path = Path(self.config.get("general.output_dir", "results")) / "output.mp4"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._video_writer = cv2.VideoWriter(
                    str(output_path), fourcc, fps,
                    (fp.width, fp.height),
                )

            frame_times = []
            logger.info(f"İşlem başlıyor: {total} kare, {fps:.1f} FPS")

            for frame_num, frame in fp:
                t_start = time.time()

                # 1. Takip (tespit + takip birleşik)
                tracked_objects = self.tracker.update(None, frame)

                # 2. İhlal tespiti
                tracked_objects, new_violations = self.violation_detector.process_frame(
                    tracked_objects, frame, frame_num, fps
                )

                # 3. Plaka tespiti + OCR (sadece yeni ihlaller için)
                for event in new_violations:
                    self._process_plate(event, frame)
                    self.violation_logger.log_violation(event)
                    self._total_violations += 1

                # 4. Görselleştirme
                display_frame = self._visualize(
                    frame, tracked_objects, new_violations,
                    frame_num, fps
                )

                # Video kaydet
                if self._video_writer is not None:
                    self._video_writer.write(display_frame)

                # Ekranda göster
                if show_display:
                    cv2.imshow("Pipeline", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Kullanıcı tarafından durduruldu")
                        break

                # FPS hesapla
                t_elapsed = time.time() - t_start
                frame_times.append(t_elapsed)

                # İlerleme raporu
                if frame_num % 100 == 0:
                    avg_fps = 1.0 / (sum(frame_times[-100:]) / len(frame_times[-100:]))
                    logger.info(
                        f"Kare {frame_num}/{total} | "
                        f"FPS: {avg_fps:.1f} | "
                        f"İhlal: {self._total_violations}"
                    )

        # Temizlik
        if self._video_writer is not None:
            self._video_writer.release()
        if show_display:
            cv2.destroyAllWindows()

        # Sonuç raporu
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
        stats = self.violation_logger.get_statistics()
        stats["average_fps"] = avg_fps
        stats["total_frames_processed"] = len(frame_times)

        logger.info(f"İşlem tamamlandı: {self._total_violations} ihlal tespit edildi")
        self.violation_logger.close()

        return stats

    def _process_plate(self, event: ViolationEvent,
                       frame: np.ndarray) -> None:
        """İhlal olayı için plaka tespiti ve OCR."""
        # Plaka tespiti
        plates = self.plate_detector.detect_from_frame(frame, event.vehicle_bbox)

        if plates:
            best_plate = plates[0]
            plate_crop = best_plate["crop"]

            # OCR
            plate_result = self.plate_reader.read_with_retry(plate_crop)
            plate_result.plate_bbox = best_plate["bbox"]
            plate_result.plate_image = plate_crop

            event.plate = plate_result

    def _visualize(self, frame: np.ndarray, tracked_objects, violations,
                   frame_num: int, fps: float) -> np.ndarray:
        """Kare üzerine tüm bilgileri çiz."""
        display = frame.copy()

        # Bölgeleri çiz
        for name, polygon in self.zone_manager.get_zone_polygons_for_drawing():
            display = self.visualizer.draw_zone(display, polygon, label=name)

        # Takip edilen araçları çiz
        for obj in tracked_objects:
            display = self.visualizer.draw_tracked_object(display, obj)

        # İhlal olaylarını çiz
        for event in violations:
            display = self.visualizer.draw_violation_event(display, event)

        # Bilgi paneli
        display = self.visualizer.draw_info_panel(
            display, frame_num, fps,
            self._total_violations, len(tracked_objects)
        )

        return display
