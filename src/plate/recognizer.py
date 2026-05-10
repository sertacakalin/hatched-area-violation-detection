"""Plaka tanıma orkestratörü — ring buffer + best-frame voting.

Pipeline her frame'de ``update_buffer()`` çağırır; bu yöntem aktif
track_id'lerin vehicle crop'larını track başına bir deque'e yazar
(maxlen = buffer_size). Violation onaylandığında ``recognize(track_id)``
çağrılır:

    1. Buffer'daki tüm crop'lara plate_detector batch'le çalıştırılır
    2. Detect skoru × √alan ile crop'lar sıralanır (büyük + güvenli plaka)
    3. Top-K (default 3) crop'a OCR uygulanır
    4. OCR confidence × (TR formatına uyuyorsa 1.5×) en yüksek olan seçilir

Bu strateji tek frame'lik OCR'ın gürültüsünü, çok-frame'li toplu işlemin
maliyetini dengeleyen pratik bir orta yoldur.
"""

import logging
from collections import deque

import numpy as np

from src.core.data_models import PlateResult, TrackedObject
from src.plate.detector import PlateDetector
from src.plate.ocr import PlateOCR

logger = logging.getLogger(__name__)


class PlateRecognizer:
    """Per-track ring buffer + best-frame voting."""

    def __init__(
        self,
        detector: PlateDetector,
        ocr: PlateOCR,
        buffer_size: int = 10,
        min_plate_conf: float = 0.25,
        topk_for_ocr: int = 3,
        valid_format_bonus: float = 1.5,
    ):
        self.detector = detector
        self.ocr = ocr
        self.buffer_size = buffer_size
        self.min_plate_conf = min_plate_conf
        self.topk_for_ocr = topk_for_ocr
        self.valid_format_bonus = valid_format_bonus
        self._buffers: dict[int, deque] = {}

    # ---- Ring buffer ---------------------------------------------------

    def update_buffer(
        self,
        tracked_objects: list[TrackedObject],
        frame: np.ndarray,
        frame_num: int,
    ) -> None:
        """Aktif track'lerin vehicle crop'larını buffer'a ekle, eskileri temizle."""
        h, w = frame.shape[:2]
        active_ids: set[int] = set()

        for obj in tracked_objects:
            tid = obj.track_id
            active_ids.add(tid)

            x1, y1, x2, y2 = obj.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            if tid not in self._buffers:
                self._buffers[tid] = deque(maxlen=self.buffer_size)
            self._buffers[tid].append((frame_num, crop))

        # Artık takip edilmeyen track'lerin buffer'ını sil
        stale = set(self._buffers.keys()) - active_ids
        for tid in stale:
            del self._buffers[tid]

    # ---- Recognition ---------------------------------------------------

    def recognize(self, track_id: int) -> PlateResult | None:
        """Track buffer'ı üzerinden best-frame voting ile plaka oku."""
        if track_id not in self._buffers or not self._buffers[track_id]:
            return None

        crops = [c for _, c in self._buffers[track_id]]
        all_dets = self.detector.detect_batch(crops)

        # Aday plaka kırpmaları + skoru
        candidates = []  # (score, plate_crop, plate_bbox_in_crop)
        for crop, dets in zip(crops, all_dets):
            for det in dets:
                if det.confidence < self.min_plate_conf:
                    continue
                px1, py1, px2, py2 = det.bbox.astype(int)
                ph, pw = crop.shape[:2]
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(pw, px2), min(ph, py2)
                if px2 <= px1 or py2 <= py1:
                    continue
                plate_crop = crop[py1:py2, px1:px2]
                area = float((px2 - px1) * (py2 - py1))
                score = det.confidence * np.sqrt(area)
                candidates.append((score, plate_crop, det.bbox.copy()))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)

        # Top-K aday üzerinde OCR — en iyi sonucu seç
        best_combined = -1.0
        best: tuple | None = None
        for _score, plate_crop, plate_bbox in candidates[: self.topk_for_ocr]:
            ocr_result = self.ocr.recognize(plate_crop)
            bonus = self.valid_format_bonus if ocr_result["is_valid"] else 1.0
            combined = ocr_result["confidence"] * bonus
            if combined > best_combined:
                best_combined = combined
                best = (ocr_result, plate_crop, plate_bbox)

        if best is None:
            return None

        ocr_result, plate_img, plate_bb = best
        return PlateResult(
            plate_text=ocr_result["plate_text"],
            raw_text=ocr_result["raw_text"],
            confidence=ocr_result["confidence"],
            plate_bbox=plate_bb,
            plate_image=plate_img,
            is_valid=ocr_result["is_valid"],
            city_code=ocr_result["city_code"],
            city_name=ocr_result["city_name"],
        )

    def cleanup_track(self, track_id: int) -> None:
        if track_id in self._buffers:
            del self._buffers[track_id]

    def reset(self) -> None:
        self._buffers.clear()
