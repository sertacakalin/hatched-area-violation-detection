"""Perspektif dönüşümü — piksel koordinatlarından gerçek dünya koordinatlarına.

Kamera görüntüsündeki piksel konumlarını, homografi matrisi kullanarak
metre cinsinden gerçek dünya koordinatlarına dönüştürür. Bu sayede:
    - Araç hızı: km/h (piksel/kare değil)
    - Taralı alandaki mesafe: metre (piksel değil)
    - Nüfuz derinliği: metre (oran değil)

Kullanım:
    1. Referans noktaları belirleme (yol çizgileri, kaldırım kenarları)
    2. Gerçek dünya karşılıklarını girme (ölçüm veya Google Maps)
    3. Homografi matrisi hesaplama
    4. Pipeline'da her frame'de araç konumlarını dönüştürme

Yazar: Sertaç Akalın — İstanbul Arel Üniversitesi (220303053)
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HomographyTransformer:
    """Piksel ↔ gerçek dünya koordinat dönüştürücü.

    Bird's eye view (kuş bakışı) perspektif dönüşümü uygular.
    En az 4 referans noktası gerektirir.
    """

    def __init__(self):
        self._H: np.ndarray | None = None  # piksel → dünya
        self._H_inv: np.ndarray | None = None  # dünya → piksel
        self._src_pts: np.ndarray | None = None
        self._dst_pts: np.ndarray | None = None
        self._reprojection_error: float = 0.0

    def calibrate(self, pixel_points: list[list[float]],
                  world_points: list[list[float]]) -> float:
        """Referans noktalarından homografi matrisi hesapla.

        Args:
            pixel_points: Görüntüdeki piksel koordinatları [[x1,y1], [x2,y2], ...]
            world_points: Gerçek dünya koordinatları (metre) [[X1,Y1], [X2,Y2], ...]

        Returns:
            Reprojection error (piksel cinsinden, düşük = iyi)
        """
        if len(pixel_points) < 4 or len(world_points) < 4:
            raise ValueError("En az 4 referans noktası gerekli")
        if len(pixel_points) != len(world_points):
            raise ValueError("Piksel ve dünya nokta sayıları eşit olmalı")

        src = np.float32(pixel_points)
        dst = np.float32(world_points)

        if len(pixel_points) == 4:
            self._H = cv2.getPerspectiveTransform(src, dst)
        else:
            self._H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        self._H_inv = np.linalg.inv(self._H)
        self._src_pts = src
        self._dst_pts = dst

        # Reprojection error hesapla
        self._reprojection_error = self._compute_error(src, dst)
        logger.info(
            f"Homografi kalibrasyonu tamamlandı. "
            f"Reprojection error: {self._reprojection_error:.2f} px"
        )
        return self._reprojection_error

    def pixel_to_world(self, pixel_point: tuple[float, float]) -> tuple[float, float]:
        """Piksel koordinatını gerçek dünya koordinatına dönüştür (metre)."""
        if self._H is None:
            raise RuntimeError("Kalibrasyon yapılmamış — calibrate() çağırın")

        pt = np.float32([[[pixel_point[0], pixel_point[1]]]])
        result = cv2.perspectiveTransform(pt, self._H)
        return (float(result[0][0][0]), float(result[0][0][1]))

    def world_to_pixel(self, world_point: tuple[float, float]) -> tuple[float, float]:
        """Gerçek dünya koordinatını piksel koordinatına dönüştür."""
        if self._H_inv is None:
            raise RuntimeError("Kalibrasyon yapılmamış — calibrate() çağırın")

        pt = np.float32([[[world_point[0], world_point[1]]]])
        result = cv2.perspectiveTransform(pt, self._H_inv)
        return (float(result[0][0][0]), float(result[0][0][1]))

    def pixel_distance_to_meters(self, pt1: tuple[float, float],
                                  pt2: tuple[float, float]) -> float:
        """İki piksel noktası arasındaki gerçek mesafeyi hesapla (metre)."""
        w1 = self.pixel_to_world(pt1)
        w2 = self.pixel_to_world(pt2)
        return float(np.sqrt((w2[0] - w1[0])**2 + (w2[1] - w1[1])**2))

    def estimate_speed(self, positions: list[tuple[float, float]],
                       fps: float) -> float:
        """Ardışık piksel konumlarından hız tahmin et (km/h).

        Args:
            positions: Ardışık karelerdeki piksel konumları
            fps: Video kare hızı

        Returns:
            Ortalama hız (km/h)
        """
        if len(positions) < 2 or self._H is None:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(positions)):
            total_distance += self.pixel_distance_to_meters(
                positions[i-1], positions[i]
            )

        total_time = (len(positions) - 1) / fps  # saniye
        if total_time <= 0:
            return 0.0

        speed_ms = total_distance / total_time  # m/s
        return speed_ms * 3.6  # km/h

    def get_birds_eye_view(self, frame: np.ndarray,
                           output_size: tuple[int, int] = (800, 600)) -> np.ndarray:
        """Bird's eye view (kuş bakışı) görüntü oluştur.

        Args:
            frame: Orijinal kamera görüntüsü
            output_size: Çıktı boyutu (genişlik, yüksek)
        """
        if self._H is None:
            raise RuntimeError("Kalibrasyon yapılmamış")

        # Dünya koordinatlarından çıktı aralığını belirle
        dst = self._dst_pts
        x_min, y_min = dst.min(axis=0)
        x_max, y_max = dst.max(axis=0)

        # Dünya → normalize çıktı dönüşümü
        scale_x = output_size[0] / (x_max - x_min)
        scale_y = output_size[1] / (y_max - y_min)
        scale = min(scale_x, scale_y) * 0.9  # %10 margin

        offset_x = (output_size[0] - (x_max - x_min) * scale) / 2
        offset_y = (output_size[1] - (y_max - y_min) * scale) / 2

        T = np.float32([
            [scale, 0, offset_x - x_min * scale],
            [0, scale, offset_y - y_min * scale],
            [0, 0, 1],
        ])

        H_bev = T @ self._H
        return cv2.warpPerspective(frame, H_bev, output_size)

    def save(self, filepath: str) -> None:
        """Kalibrasyon verilerini JSON olarak kaydet."""
        if self._H is None:
            raise RuntimeError("Kalibrasyon yapılmamış")

        data = {
            "homography_matrix": self._H.tolist(),
            "pixel_points": self._src_pts.tolist(),
            "world_points": self._dst_pts.tolist(),
            "reprojection_error": self._reprojection_error,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Kalibrasyon kaydedildi: {filepath}")

    def load(self, filepath: str) -> None:
        """Kalibrasyon verilerini JSON'dan yükle."""
        with open(filepath) as f:
            data = json.load(f)

        self._H = np.float64(data["homography_matrix"])
        self._H_inv = np.linalg.inv(self._H)
        self._src_pts = np.float32(data["pixel_points"])
        self._dst_pts = np.float32(data["world_points"])
        self._reprojection_error = data.get("reprojection_error", 0.0)
        logger.info(f"Kalibrasyon yüklendi: {filepath}")

    @property
    def is_calibrated(self) -> bool:
        return self._H is not None

    @property
    def reprojection_error(self) -> float:
        return self._reprojection_error

    def _compute_error(self, src: np.ndarray, dst: np.ndarray) -> float:
        """Reprojection error hesapla."""
        projected = cv2.perspectiveTransform(
            src.reshape(-1, 1, 2), self._H
        ).reshape(-1, 2)
        errors = np.sqrt(np.sum((projected - dst)**2, axis=1))
        return float(np.mean(errors))
