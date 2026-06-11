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

import cv2
import numpy as np

from src.core.data_models import PlateResult, TrackedObject
from src.plate.detector import PlateDetector
from src.plate.ocr import PlateOCR
from src.plate.tr_plate import repair_tr_plate, validate_tr_plate

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
        vehicle_padding_ratio: float = 0.03,
        plate_padding_ratio_x: float = 0.12,
        plate_padding_ratio_y: float = 0.25,
        min_plate_width: int = 32,
        min_plate_height: int = 8,
        min_ocr_confidence: float = 0.15,
        return_invalid_results: bool = False,
    ):
        self.detector = detector
        self.ocr = ocr
        self.buffer_size = buffer_size
        self.min_plate_conf = min_plate_conf
        self.topk_for_ocr = topk_for_ocr
        self.valid_format_bonus = valid_format_bonus
        self.vehicle_padding_ratio = vehicle_padding_ratio
        self.plate_padding_ratio_x = plate_padding_ratio_x
        self.plate_padding_ratio_y = plate_padding_ratio_y
        self.min_plate_width = min_plate_width
        self.min_plate_height = min_plate_height
        self.min_ocr_confidence = min_ocr_confidence
        self.return_invalid_results = return_invalid_results
        self._buffers: dict[int, deque] = {}

    @staticmethod
    def _expand_bbox(
        bbox: np.ndarray,
        image_w: int,
        image_h: int,
        pad_x_ratio: float,
        pad_y_ratio: float,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox.astype(int)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(round(bw * pad_x_ratio))
        pad_y = int(round(bh * pad_y_ratio))
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(image_w, x2 + pad_x),
            min(image_h, y2 + pad_y),
        )

    def _plate_quality(self, plate_crop: np.ndarray) -> float:
        """Boyut, oran, kontrast ve keskinlikten basit kalite skoru üret."""
        h, w = plate_crop.shape[:2]
        if w < self.min_plate_width or h < self.min_plate_height:
            return 0.0

        aspect = w / max(h, 1)
        if aspect < 1.6 or aspect > 8.5:
            aspect_score = 0.35
        elif 2.0 <= aspect <= 6.5:
            aspect_score = 1.0
        else:
            aspect_score = 0.7

        gray = (
            cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            if plate_crop.ndim == 3
            else plate_crop
        )
        contrast = float(gray.std())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        size_score = min(w / 90.0, 1.5) * min(h / 24.0, 1.5)
        contrast_score = min(max(contrast / 35.0, 0.45), 1.5)
        sharpness_score = min(max(sharpness / 80.0, 0.45), 1.8)
        return aspect_score * size_score * contrast_score * sharpness_score

    # ---- Ring buffer ---------------------------------------------------

    def update_buffer(
        self,
        tracked_objects: list[TrackedObject],
        frame: np.ndarray,
        frame_num: int,
        keep_track_ids: set[int] | None = None,
    ) -> None:
        """Aktif track'lerin vehicle crop'larını buffer'a ekle, eskileri temizle."""
        h, w = frame.shape[:2]
        active_ids: set[int] = set()
        keep_track_ids = keep_track_ids or set()

        for obj in tracked_objects:
            tid = obj.track_id
            active_ids.add(tid)

            x1, y1, x2, y2 = self._expand_bbox(
                obj.bbox,
                w,
                h,
                self.vehicle_padding_ratio,
                self.vehicle_padding_ratio,
            )
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            if tid not in self._buffers:
                self._buffers[tid] = deque(maxlen=self.buffer_size)
            self._buffers[tid].append((frame_num, crop))

        # Artık takip edilmeyen track'lerin buffer'ını sil
        stale = set(self._buffers.keys()) - active_ids - keep_track_ids
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
        candidates = []  # (score, quality, plate_crop, plate_bbox_in_crop)
        for crop, dets in zip(crops, all_dets):
            for det in dets:
                if det.confidence < self.min_plate_conf:
                    continue
                ph, pw = crop.shape[:2]
                px1, py1, px2, py2 = self._expand_bbox(
                    det.bbox,
                    pw,
                    ph,
                    self.plate_padding_ratio_x,
                    self.plate_padding_ratio_y,
                )
                if px2 <= px1 or py2 <= py1:
                    continue
                plate_crop = crop[py1:py2, px1:px2].copy()
                quality = self._plate_quality(plate_crop)
                if quality <= 0:
                    continue
                area = float((px2 - px1) * (py2 - py1))
                score = det.confidence * np.sqrt(area) * quality
                plate_bbox = np.array([px1, py1, px2, py2], dtype=float)
                candidates.append((score, quality, plate_crop, plate_bbox))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)

        # Top-K aday üzerinde OCR. İki ayrı oylama yapılır:
        #   1. Tüm-metin oyu (votes): aynı string birden çok frame'de gelirse
        #      ağırlıkları toplanır.
        #   2. Karakter-bazlı consensus: tek bir frame'in karakter hatasını
        #      (örn. 4↔6, D↔O) elemek için aynı uzunluktaki okumalarda her
        #      pozisyon ağırlıklı çoğunlukla seçilir. Video LPR'da tek frame
        #      gürültüsüne karşı en etkili düzeltme.
        votes: dict[str, float] = {}
        # reads: (plate_text, weight, ocr_result, plate_crop, plate_bbox)
        reads: list[tuple[str, float, dict, np.ndarray, np.ndarray]] = []

        for _score, quality, plate_crop, plate_bbox in candidates[
            : self.topk_for_ocr
        ]:
            ocr_result = self.ocr.recognize(plate_crop)
            plate_text = ocr_result["plate_text"]
            if not plate_text:
                continue
            bonus = self.valid_format_bonus if ocr_result["is_valid"] else 1.0
            quality_bonus = 0.75 + min(quality, 1.5) * 0.25
            combined = ocr_result["confidence"] * bonus * quality_bonus

            votes[plate_text] = votes.get(plate_text, 0.0) + combined
            reads.append(
                (plate_text, combined, ocr_result, plate_crop, plate_bbox)
            )

        if not reads:
            return None

        # İki finalist üret, geçerli-format öncelikli seç. Consensus tek başına
        # bazen geçersiz string üretebilir; o durumda tüm-metin oyunun geçerli
        # kazananına düşmek recall'ı korur (yalnız-PaddleOCR davranışının altına
        # asla inmeyiz, üstüne karakter düzeltmesi ekleriz).
        finalists: list[dict] = []

        # Finalist 1: karakter-bazlı consensus (≥2 okuma aynı uzunlukta)
        consensus = self._char_consensus([(t, w) for t, w, *_ in reads])
        if consensus is not None:
            ctext = repair_tr_plate(consensus)
            cvalid, ccity, cname = validate_tr_plate(ctext)
            same_len = [r for r in reads if len(r[0]) == len(consensus)]
            csrc = max(same_len or reads, key=lambda r: r[1])
            cweight = sum(r[1] for r in same_len) if same_len else csrc[1]
            finalists.append({
                "text": ctext, "valid": cvalid, "city_code": ccity,
                "city_name": cname, "weight": cweight, "src": csrc,
                "prefer": 1,  # eşitlikte consensus tercih edilir
            })

        # Finalist 2: tüm-metin oyu kazananı
        best_text = max(votes, key=votes.get)
        wsrc = max((r for r in reads if r[0] == best_text), key=lambda r: r[1])
        w_ocr = wsrc[2]
        finalists.append({
            "text": w_ocr["plate_text"], "valid": w_ocr["is_valid"],
            "city_code": w_ocr["city_code"], "city_name": w_ocr["city_name"],
            "weight": votes[best_text], "src": wsrc, "prefer": 0,
        })

        # Geçerli format > ağırlık > consensus-tercihi
        finalists.sort(
            key=lambda f: (f["valid"], f["weight"], f["prefer"]),
            reverse=True,
        )
        choice = finalists[0]
        src = choice["src"]
        ocr_src = src[2]
        plate_text = choice["text"]
        is_valid = choice["valid"]
        city_code = choice["city_code"]
        city_name = choice["city_name"]
        confidence = ocr_src["confidence"]
        raw_text = ocr_src["raw_text"]
        plate_img, plate_bb = src[3], src[4]

        if confidence < self.min_ocr_confidence:
            return None
        if not is_valid and not self.return_invalid_results:
            return None

        return PlateResult(
            plate_text=plate_text,
            raw_text=raw_text,
            confidence=confidence,
            plate_bbox=plate_bb,
            plate_image=plate_img,
            is_valid=is_valid,
            city_code=city_code,
            city_name=city_name,
        )

    @staticmethod
    def _char_consensus(reads: list[tuple[str, float]]) -> str | None:
        """Aynı uzunluktaki okumalarda pozisyon başına ağırlıklı çoğunluk.

        En az 2 okuma aynı (modal) uzunlukta değilse None döner; bu durumda
        çağıran taraf tüm-metin oyuna düşer.
        """
        if len(reads) < 2:
            return None
        # Modal uzunluk: toplam ağırlığı en yüksek uzunluk
        len_weight: dict[int, float] = {}
        for text, w in reads:
            len_weight[len(text)] = len_weight.get(len(text), 0.0) + w
        target_len = max(len_weight, key=len_weight.get)
        same_len = [(t, w) for t, w in reads if len(t) == target_len]
        if len(same_len) < 2:
            return None

        chars: list[str] = []
        for pos in range(target_len):
            pos_w: dict[str, float] = {}
            for text, w in same_len:
                pos_w[text[pos]] = pos_w.get(text[pos], 0.0) + w
            chars.append(max(pos_w, key=pos_w.get))
        return "".join(chars)

    def cleanup_track(self, track_id: int) -> None:
        if track_id in self._buffers:
            del self._buffers[track_id]

    def reset(self) -> None:
        self._buffers.clear()
