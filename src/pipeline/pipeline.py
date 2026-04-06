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
from src.core.heatmap import ViolationHeatmap
from src.core.traffic_density import TrafficDensityAnalyzer
from src.detection.vehicle_detector import VehicleDetector
from src.tracking.tracker import create_tracker
from src.zones.zone_manager import ZoneManager
from src.zones.dynamic_zone_tracker import DynamicZoneTracker
from src.violation.violation_detector import ViolationDetector
from src.violation.clip_extractor import ViolationClipExtractor
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

        # Plaka Tespiti + OCR (opsiyonel — model/kütüphane yoksa atla)
        self.plate_detector = None
        self.plate_reader = None
        try:
            plate_model_path = config.get("plate_detection.model_path", "weights/plate_detector.pt")
            from pathlib import Path as _P
            if _P(plate_model_path).exists():
                self.plate_detector = PlateDetector(
                    model_path=plate_model_path,
                    confidence_threshold=config.get("plate_detection.confidence_threshold", 0.5),
                )
                ocr_engine = config.get("plate_ocr.engine", "paddleocr")
                self.plate_reader = PlateReader(
                    engine=ocr_engine,
                    min_confidence=config.get("plate_validation.min_confidence", 0.6),
                )
                logger.info("Plaka tespiti + OCR aktif")
            else:
                logger.warning(f"Plaka modeli bulunamadı: {plate_model_path} — plaka tespiti devre dışı")
        except Exception as e:
            logger.warning(f"Plaka modülü başlatılamadı: {e} — plaka tespiti devre dışı")

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

        # Klip çıkarıcı (ihlal anı video klipleri)
        self.clip_extractor = ViolationClipExtractor(
            buffer_seconds=config.get("clip.buffer_seconds", 2.0),
            after_seconds=config.get("clip.after_seconds", 2.0),
            output_dir=str(Path(config.get("general.output_dir", "results")) / "clips"),
            fps=30.0,  # frame_provider açılınca güncellenir
        )

        # Isı haritası (tez figürü için)
        self.heatmap = ViolationHeatmap()

        # Trafik yoğunluğu (ihlal bağlamı için)
        self.density_analyzer = TrafficDensityAnalyzer(
            window_size=config.get("density.window_size", 90),
        )

        # Dinamik zone takibi (kamera hareket ediyorsa)
        self.dynamic_zone = None
        if config.get("zone.dynamic_tracking", False):
            self.dynamic_zone = DynamicZoneTracker(
                method=config.get("zone.tracking_method", "orb"),
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

            # Klip çıkarıcı FPS'ini güncelle
            self.clip_extractor.fps = fps
            self.clip_extractor.buffer_size = int(2.0 * fps)
            self.clip_extractor.after_frames = int(2.0 * fps)
            self.clip_extractor._buffer = __import__('collections').deque(
                maxlen=self.clip_extractor.buffer_size
            )

            # Isı haritası boyutunu ayarla
            self.heatmap = ViolationHeatmap(width=fp.width, height=fp.height)

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

                # 1.5 Dinamik zone güncelleme (kamera kayması telafisi)
                if self.dynamic_zone is not None:
                    if not self.dynamic_zone.is_initialized:
                        # İlk frame → referans olarak kaydet
                        initial_coords = []
                        for zone in self.zone_manager.zones:
                            initial_coords = list(zone.polygon.exterior.coords[:-1])
                            break
                        if initial_coords:
                            self.dynamic_zone.set_reference(
                                frame, [[int(x), int(y)] for x, y in initial_coords]
                            )
                    else:
                        # Polygon'u güncelle
                        new_coords = self.dynamic_zone.update(frame)
                        if new_coords and self.zone_manager.zones:
                            from shapely.geometry import Polygon as ShapelyPolygon
                            self.zone_manager.zones[0].polygon = ShapelyPolygon(new_coords)

                # 2. Trafik yoğunluğu güncelle
                self.density_analyzer.update(len(tracked_objects))

                # 3. Klip buffer'a kare ekle
                self.clip_extractor.feed_frame(frame)

                # 4. İhlal tespiti
                tracked_objects, new_violations = self.violation_detector.process_frame(
                    tracked_objects, frame, frame_num, fps
                )

                # 5. Yeni ihlaller → plaka + klip + heatmap + kayıt
                for event in new_violations:
                    # Yoğunluk bağlamını ekle
                    event.metadata["traffic_density"] = self.density_analyzer.get_violation_context()

                    self._process_plate(event, frame)
                    self.violation_logger.log_violation(event)
                    self._total_violations += 1

                    # Isı haritasına ekle
                    cx = (event.vehicle_bbox[0] + event.vehicle_bbox[2]) / 2
                    cy = event.vehicle_bbox[3]  # alt merkez
                    self.heatmap.add_violation((cx, cy), event.severity_score)

                    # Klip kaydını başlat
                    zone_polygons = list(self.zone_manager.get_zone_polygons_for_drawing())
                    self.clip_extractor.on_violation(
                        event_id=event.event_id,
                        track_id=event.track_id,
                        severity_score=event.severity_score,
                        violation_type=event.violation_type,
                        vehicle_bbox=event.vehicle_bbox,
                        zone_polygons=zone_polygons,
                    )

                # 6. Görselleştirme
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
        self.clip_extractor.flush()

        if self._video_writer is not None:
            self._video_writer.release()
        if show_display:
            cv2.destroyAllWindows()

        # Isı haritasını kaydet
        heatmap_path = str(Path(self.config.get("general.output_dir", "results")) / "heatmap.png")
        self.heatmap.save(heatmap_path)
        logger.info(f"Isı haritası kaydedildi: {heatmap_path}")

        # Sonuç raporu
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
        stats = self.violation_logger.get_statistics()
        stats["average_fps"] = avg_fps
        stats["total_frames_processed"] = len(frame_times)
        stats["severity_statistics"] = self.violation_detector.get_severity_statistics()
        stats["traffic_density"] = {
            "average": self.density_analyzer.current_density,
            "level": self.density_analyzer.density_level,
        }

        logger.info(f"İşlem tamamlandı: {self._total_violations} ihlal tespit edildi")
        self.violation_logger.close()

        return stats

    def _process_plate(self, event: ViolationEvent,
                       frame: np.ndarray) -> None:
        """İhlal olayı için plaka tespiti ve OCR."""
        if self.plate_detector is None or self.plate_reader is None:
            return

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
