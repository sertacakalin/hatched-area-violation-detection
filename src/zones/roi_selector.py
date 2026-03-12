"""İnteraktif ROI (Region of Interest) seçim aracı — OpenCV mouse callback."""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ROISelector:
    """OpenCV penceresi ile interaktif polygon seçimi.

    Kullanım:
        selector = ROISelector()
        polygon = selector.select_from_video("video.mp4")
        selector.save_zone("configs/zones/camera_01.json", polygon)
    """

    WINDOW_NAME = "ROI Secimi - Sol tik: nokta ekle | Sag tik: geri al | 'q': bitir"

    def __init__(self):
        self.points: list[list[int]] = []
        self._frame: np.ndarray | None = None
        self._display: np.ndarray | None = None

    def _mouse_callback(self, event: int, x: int, y: int,
                        flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self._redraw()

    def _redraw(self) -> None:
        self._display = self._frame.copy()
        if len(self.points) > 0:
            pts = np.array(self.points, dtype=np.int32)

            # Noktaları çiz
            for i, pt in enumerate(self.points):
                cv2.circle(self._display, tuple(pt), 5, (0, 255, 0), -1)
                cv2.putText(self._display, str(i + 1), (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Çizgileri çiz
            if len(self.points) > 1:
                cv2.polylines(self._display, [pts], False, (0, 255, 0), 2)

            # Polygon'u yarı şeffaf göster (3+ nokta)
            if len(self.points) > 2:
                overlay = self._display.copy()
                cv2.fillPoly(overlay, [pts], (255, 165, 0))
                self._display = cv2.addWeighted(overlay, 0.3, self._display, 0.7, 0)
                cv2.polylines(self._display, [pts], True, (255, 165, 0), 2)

        # Bilgi metni
        info = f"Noktalar: {len(self.points)} | Sol tik: ekle | Sag tik: geri | 'q': bitir | 'r': sifirla"
        cv2.putText(self._display, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(self.WINDOW_NAME, self._display)

    def select_from_frame(self, frame: np.ndarray) -> list[list[int]]:
        """Tek bir kare üzerinden polygon seç."""
        self.points = []
        self._frame = frame.copy()
        self._display = frame.copy()

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1280, 720)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
        cv2.imshow(self.WINDOW_NAME, self._display)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.points = []
                self._redraw()

        cv2.destroyAllWindows()

        if len(self.points) < 3:
            logger.warning("En az 3 nokta gerekli, polygon oluşturulamadı")
            return []

        logger.info(f"Polygon seçildi: {len(self.points)} nokta")
        return self.points

    def select_from_video(self, video_path: str,
                          frame_number: int = 0) -> list[list[int]]:
        """Video dosyasının belirli bir karesinden polygon seç."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Video açılamadı: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise IOError(f"Kare okunamadı: frame={frame_number}")

        return self.select_from_frame(frame)

    def save_zone(self, output_path: str | Path, polygon: list[list[int]],
                  zone_id: str = "zone_01", name: str = "Taralı Alan",
                  frame_width: int = 1920, frame_height: int = 1080) -> None:
        """Seçilen polygon'u JSON dosyasına kaydet."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "camera_id": output_path.stem,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "zones": [
                {
                    "zone_id": zone_id,
                    "name": name,
                    "polygon": polygon,
                    "type": "hatched_area",
                }
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Bölge kaydedildi: {output_path}")
