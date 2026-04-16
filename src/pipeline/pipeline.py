"""Ana orkestratör — tüm bileşenleri birleştiren pipeline.

Basitleştirilmiş akış:
    Video → Kare → Tespit+Takip → Bölge Kontrolü → İhlal Onayı
    → Kayıt → Görselleştirme → Çıktı Video
"""

import logging
import time
from pathlib import Path

import cv2

from src.core.config import Config
from src.core.frame_provider import FrameProvider
from src.core.visualizer import Visualizer
from src.tracking.tracker import create_tracker
from src.zones.zone_manager import ZoneManager
from src.violation.violation_detector import ViolationDetector
from src.storage.violation_logger import ViolationLogger

logger = logging.getLogger(__name__)


class Pipeline:
    """Uçtan uca ihlal tespit pipeline'ı."""

    def __init__(self, config: Config):
        self.config = config

        logger.info("Pipeline bileşenleri başlatılıyor...")

        # Video
        self.frame_provider = FrameProvider(
            config.get("general.video_source")
        )

        # Araç Tespiti + Takip (ByteTrack içinde YOLOv8 inference var)
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

        # Bölge Yönetimi (taralı alan polygon'u)
        self.zone_manager = ZoneManager(
            zone_file=config.get("zone.zone_file"),
            polygon_buffer=config.get("zone.polygon_buffer", -10),
        )

        # İhlal Tespiti (state machine + trajectory + severity)
        self.violation_detector = ViolationDetector(
            zone_manager=self.zone_manager,
            min_frames_in_zone=config.get("violation.min_frames_in_zone", 5),
            cooldown_frames=config.get("violation.cooldown_frames", 90),
            min_overlap_ratio=config.get("zone.min_overlap_ratio", 0.3),
        )

        # Kayıt (SQLite + JSON)
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

                # 1. Tespit + Takip
                tracked_objects = self.tracker.update(None, frame)

                # 2. İhlal tespiti (state machine + trajectory + severity)
                tracked_objects, new_violations = self.violation_detector.process_frame(
                    tracked_objects, frame, frame_num, fps
                )

                # 3. Yeni ihlaller → kayıt
                for event in new_violations:
                    self.violation_logger.log_violation(event)
                    self._total_violations += 1

                # 4. Görselleştirme
                display_frame = self._visualize(
                    frame, tracked_objects, new_violations,
                    frame_num, fps
                )

                if self._video_writer is not None:
                    self._video_writer.write(display_frame)

                if show_display:
                    cv2.imshow("Pipeline", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Kullanıcı tarafından durduruldu")
                        break

                t_elapsed = time.time() - t_start
                frame_times.append(t_elapsed)

                if frame_num % 100 == 0:
                    avg_fps = 1.0 / (sum(frame_times[-100:]) / len(frame_times[-100:]))
                    logger.info(
                        f"Kare {frame_num}/{total} | "
                        f"FPS: {avg_fps:.1f} | "
                        f"İhlal: {self._total_violations}"
                    )

        if self._video_writer is not None:
            self._video_writer.release()
        if show_display:
            cv2.destroyAllWindows()

        # Sonuç raporu
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
        stats = self.violation_logger.get_statistics()
        stats["average_fps"] = avg_fps
        stats["total_frames_processed"] = len(frame_times)
        stats["severity_statistics"] = self.violation_detector.get_severity_statistics()

        logger.info(f"İşlem tamamlandı: {self._total_violations} ihlal tespit edildi")
        self.violation_logger.close()

        return stats

    def _visualize(self, frame, tracked_objects, violations,
                   frame_num: int, fps: float):
        """Kare üzerine bbox + bölge + ihlal bilgisi çiz."""
        display = frame.copy()

        for name, polygon in self.zone_manager.get_zone_polygons_for_drawing():
            display = self.visualizer.draw_zone(display, polygon, label=name)

        for obj in tracked_objects:
            display = self.visualizer.draw_tracked_object(display, obj)

        for event in violations:
            display = self.visualizer.draw_violation_event(display, event)

        display = self.visualizer.draw_info_panel(
            display, frame_num, fps,
            self._total_violations, len(tracked_objects)
        )

        return display
