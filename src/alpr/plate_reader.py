"""Plaka OCR — PaddleOCR (birincil) ve EasyOCR (karşılaştırma) wrapper."""

import logging
import re

import numpy as np

from src.core.data_models import PlateResult
from src.alpr.plate_preprocessor import PlatePreprocessor
from src.alpr.plate_validator import PlateValidator

logger = logging.getLogger(__name__)


class PlateReader:
    """Plaka görüntüsünden metin okuma modülü."""

    def __init__(
        self,
        engine: str = "paddleocr",
        lang: str = "en",
        use_angle_cls: bool = True,
        gpu: bool = True,
        min_confidence: float = 0.6,
    ):
        self.engine_name = engine
        self.min_confidence = min_confidence
        self.preprocessor = PlatePreprocessor()
        self.validator = PlateValidator()
        self._engine = None

        if engine == "paddleocr":
            self._init_paddleocr(lang, use_angle_cls, gpu)
        elif engine == "easyocr":
            self._init_easyocr(gpu)
        else:
            raise ValueError(f"Bilinmeyen OCR motoru: {engine}")

    def _init_paddleocr(self, lang: str, use_angle_cls: bool,
                        gpu: bool) -> None:
        try:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(
                lang=lang,
                det=False,       # Plaka zaten kırpılmış, sadece tanıma
                rec=True,
                use_angle_cls=use_angle_cls,
                use_gpu=gpu,
                show_log=False,
                rec_algorithm="SVTR_LCNet",
            )
            logger.info("PaddleOCR başlatıldı")
        except ImportError:
            logger.error("PaddleOCR yüklü değil: pip install paddleocr paddlepaddle")
            raise

    def _init_easyocr(self, gpu: bool) -> None:
        try:
            import easyocr
            self._engine = easyocr.Reader(
                ["en"],
                gpu=gpu,
                verbose=False,
            )
            logger.info("EasyOCR başlatıldı")
        except ImportError:
            logger.error("EasyOCR yüklü değil: pip install easyocr")
            raise

    def read(self, plate_image: np.ndarray) -> PlateResult:
        """Plaka görüntüsünden metin oku."""
        if plate_image.size == 0:
            return PlateResult(plate_text="", raw_text="", confidence=0.0)

        if self.engine_name == "paddleocr":
            return self._read_paddleocr(plate_image)
        else:
            return self._read_easyocr(plate_image)

    def _read_paddleocr(self, plate_image: np.ndarray) -> PlateResult:
        """PaddleOCR ile plaka oku."""
        # Ön-işleme
        processed = self.preprocessor.preprocess_for_paddleocr(plate_image)

        results = self._engine.ocr(processed, det=False, rec=True, cls=True)

        best_text = ""
        best_conf = 0.0

        if results:
            for line in results:
                if line:
                    for item in line:
                        text, conf = item[0], item[1]
                        if conf > best_conf:
                            best_text = text
                            best_conf = conf

        # Metin temizleme
        cleaned = self._clean_plate_text(best_text)

        # Doğrulama
        is_valid = self.validator.validate(cleaned)

        return PlateResult(
            plate_text=cleaned,
            raw_text=best_text,
            confidence=best_conf,
            plate_image=plate_image,
            is_valid=is_valid,
        )

    def _read_easyocr(self, plate_image: np.ndarray) -> PlateResult:
        """EasyOCR ile plaka oku."""
        processed = self.preprocessor.preprocess_for_easyocr(plate_image)

        results = self._engine.readtext(processed, detail=1)

        best_text = ""
        best_conf = 0.0

        if results:
            for bbox, text, conf in results:
                if conf > best_conf:
                    best_text = text
                    best_conf = conf

        cleaned = self._clean_plate_text(best_text)
        is_valid = self.validator.validate(cleaned)

        return PlateResult(
            plate_text=cleaned,
            raw_text=best_text,
            confidence=best_conf,
            plate_image=plate_image,
            is_valid=is_valid,
        )

    def read_with_retry(self, plate_image: np.ndarray) -> PlateResult:
        """Birden fazla ön-işleme varyantıyla deneme yaparak en iyi sonucu döndür."""
        variants = self.preprocessor.get_all_variants(plate_image)
        best_result = PlateResult(plate_text="", raw_text="", confidence=0.0)

        for variant in variants:
            result = self.read(variant)
            if result.is_valid and result.confidence > best_result.confidence:
                best_result = result
            elif (not best_result.is_valid
                  and result.confidence > best_result.confidence):
                best_result = result

        return best_result

    @staticmethod
    def _clean_plate_text(text: str) -> str:
        """OCR çıktısını temizle — Türk plaka formatına uygun hale getir."""
        # Büyük harfe çevir
        text = text.upper().strip()

        # Yaygın OCR hatalarını düzelt
        replacements = {
            "O": "0", "I": "1", "S": "5", "B": "8",
            "G": "6", "Z": "2", "T": "7",
        }

        # Sadece harf ve rakam
        text = re.sub(r"[^A-Z0-9]", "", text)

        if len(text) < 4:
            return text

        # İlk 2 karakter il kodu (rakam olmalı)
        result = list(text)
        for i in range(min(2, len(result))):
            if result[i] in replacements:
                result[i] = replacements[result[i]]

        return "".join(result)
