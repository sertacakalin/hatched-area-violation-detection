"""Dinamik zone takibi — kamera hareket ettiğinde polygon'u günceller.

Kamera kayması (pan, titreme, el kamerası) durumunda sabit polygon
yanlış bölgeye işaret eder. Bu modül her frame'de optik akış veya
ORB feature matching ile kamera hareketini hesaplar ve polygon
koordinatlarını buna göre günceller.

Algoritma:
    Frame 0: Referans frame + başlangıç polygon kaydedilir
    Frame N: ORB feature matching ile Frame 0 → Frame N homografi hesaplanır
             Polygon noktaları bu homografi ile dönüştürülür
             ZoneManager'daki polygon güncellenir

Yazar: Sertaç Akalın — İstanbul Arel Üniversitesi (220303053)
"""

import logging

import cv2
import numpy as np
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


class DynamicZoneTracker:
    """Kamera hareketiyle birlikte zone polygon'unu takip eder.

    İki mod destekler:
    1. ORB + homografi: Genel kamera hareketi (döndürme, yakınlaştırma dahil)
    2. Optical Flow: Hızlı, küçük kayma için (sadece öteleme)
    """

    def __init__(self, method: str = "orb", min_matches: int = 20,
                 ransac_threshold: float = 5.0):
        """
        Args:
            method: "orb" veya "optical_flow"
            min_matches: ORB eşleştirme için minimum özellik sayısı
            ransac_threshold: RANSAC outlier eşiği (piksel)
        """
        self.method = method
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold

        # Referans durumu
        self._ref_frame_gray: np.ndarray | None = None
        self._ref_polygon: np.ndarray | None = None  # [[x1,y1], [x2,y2], ...]
        self._ref_keypoints = None
        self._ref_descriptors = None

        # ORB dedektör
        self._orb = cv2.ORB_create(nfeatures=1000)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Optical flow için önceki frame
        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None

        # İstatistikler
        self._total_frames = 0
        self._failed_frames = 0
        self._cumulative_shift = np.float64([0.0, 0.0])

    def set_reference(self, frame: np.ndarray,
                      polygon_points: list[list[int]]) -> None:
        """Referans frame ve başlangıç polygon'u kaydet.

        Pipeline başlangıcında bir kere çağrılır.

        Args:
            frame: İlk video karesi (BGR)
            polygon_points: Zone köşe koordinatları [[x1,y1], ...]
        """
        self._ref_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._ref_polygon = np.float32(polygon_points)

        if self.method == "orb":
            self._ref_keypoints, self._ref_descriptors = self._orb.detectAndCompute(
                self._ref_frame_gray, None
            )
            if self._ref_descriptors is None or len(self._ref_keypoints) < self.min_matches:
                logger.warning(
                    f"Referans frame'de yeterli özellik bulunamadı "
                    f"({len(self._ref_keypoints) if self._ref_keypoints else 0}). "
                    f"Stabilizasyon zayıf olabilir."
                )

        self._prev_gray = self._ref_frame_gray.copy()
        self._prev_points = self._ref_polygon.copy().reshape(-1, 1, 2)
        self._total_frames = 0
        self._failed_frames = 0

        logger.info(
            f"Dinamik zone takibi başlatıldı: method={self.method}, "
            f"polygon={len(polygon_points)} nokta"
        )

    def update(self, frame: np.ndarray) -> list[list[int]]:
        """Yeni frame ile polygon konumunu güncelle.

        Args:
            frame: Güncel video karesi (BGR)

        Returns:
            Güncellenmiş polygon koordinatları [[x1,y1], ...]
        """
        if self._ref_polygon is None:
            raise RuntimeError("set_reference() çağrılmamış")

        self._total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.method == "orb":
            new_polygon = self._update_orb(gray)
        else:
            new_polygon = self._update_optical_flow(gray)

        self._prev_gray = gray

        if new_polygon is not None:
            return [[int(p[0]), int(p[1])] for p in new_polygon]

        # Başarısız → orijinal polygon'u döndür
        self._failed_frames += 1
        return [[int(p[0]), int(p[1])] for p in self._ref_polygon]

    def _update_orb(self, gray: np.ndarray) -> np.ndarray | None:
        """ORB feature matching ile homografi hesapla."""
        if self._ref_descriptors is None:
            return None

        kp, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(kp) < self.min_matches:
            return None

        # KNN eşleştirme + Lowe's ratio test
        matches = self._matcher.knnMatch(self._ref_descriptors, desc, k=2)
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.min_matches:
            logger.debug(
                f"Frame {self._total_frames}: Yetersiz eşleşme "
                f"({len(good_matches)}/{self.min_matches})"
            )
            return None

        # Homografi hesapla (referans → güncel frame)
        src_pts = np.float32(
            [self._ref_keypoints[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold
        )

        if H is None:
            return None

        # Polygon'u dönüştür
        polygon_pts = self._ref_polygon.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(polygon_pts, H)

        return transformed.reshape(-1, 2)

    def _update_optical_flow(self, gray: np.ndarray) -> np.ndarray | None:
        """Optical flow ile polygon kaydırma."""
        if self._prev_gray is None:
            return None

        # Sparse optical flow (Lucas-Kanade)
        # Referans polygon noktalarını takip et
        prev_pts = self._ref_polygon.copy().reshape(-1, 1, 2)

        # Referans frame'den güncel frame'e optical flow
        new_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self._ref_frame_gray, gray, prev_pts, None,
            winSize=(51, 51),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        if new_pts is None or status is None:
            return None

        # Başarılı noktalar
        good_mask = status.flatten() == 1
        if np.sum(good_mask) < 3:
            return None

        # Ortalama kayma hesapla (tüm noktalar aynı yöne kaymalı)
        shifts = new_pts[good_mask] - prev_pts[good_mask]
        median_shift = np.median(shifts, axis=0)

        # Polygon'u kaydır
        new_polygon = self._ref_polygon + median_shift.flatten()

        self._cumulative_shift = median_shift.flatten()

        return new_polygon

    def get_updated_zone_data(self, frame: np.ndarray) -> dict:
        """ZoneManager'a verilecek formatta güncel zone verisi döndür.

        Args:
            frame: Güncel video karesi

        Returns:
            ZoneManager JSON formatında zone verisi
        """
        polygon = self.update(frame)
        return {
            "zones": [{
                "zone_id": "hatched_area_1",
                "name": "tarali alan (dinamik)",
                "polygon": polygon,
                "type": "hatched_area",
            }]
        }

    @property
    def is_initialized(self) -> bool:
        return self._ref_polygon is not None

    @property
    def tracking_quality(self) -> float:
        """Takip kalitesi (0-1). Yüksek = iyi."""
        if self._total_frames == 0:
            return 1.0
        return 1.0 - (self._failed_frames / self._total_frames)

    @property
    def stats(self) -> dict:
        return {
            "total_frames": self._total_frames,
            "failed_frames": self._failed_frames,
            "quality": round(self.tracking_quality, 3),
            "cumulative_shift_px": [
                round(float(self._cumulative_shift[0]), 1),
                round(float(self._cumulative_shift[1]), 1),
            ],
        }
