"""EasyOCR tabanlı plaka karakter okuma + TR format validasyonu.

EasyOCR seçimi: PyTorch tabanlı, Mac (MPS/CPU) ve Linux (CUDA) üzerinde
sorunsuz çalışır, Latin karakter setinde tatmin edici accuracy. PaddleOCR
ile karşılaştırma tezde tartışılabilir.

OCR öncesi upscale: kaynak frame'de uzak/küçük plakalar 30-50 piksel
genişlikte olabilir, EasyOCR bu boyutta düşük accuracy gösterir. Plaka
crop'unu en az ~150 piksel genişliğe Lanczos ile büyütüp hafif unsharp
mask uygulamak okunabilirlik oranını anlamlı artırır.
"""

import logging
from typing import TypedDict

import cv2
import numpy as np

from src.plate.tr_plate import normalize_tr_plate, validate_tr_plate

MIN_OCR_WIDTH = 150  # Daha küçük plakalar Lanczos ile bu genişliğe çıkarılır

logger = logging.getLogger(__name__)


class OCRResult(TypedDict):
    raw_text: str
    plate_text: str
    confidence: float
    is_valid: bool
    city_code: str | None
    city_name: str | None


SUPPORTED_OCR_BACKENDS = ("easyocr",)


class PlateOCR:
    """EasyOCR sarmalayıcı + TR plaka format normalizasyonu."""

    def __init__(
        self,
        languages: list[str] | None = None,
        use_gpu: bool = False,
        backend: str = "easyocr",
    ):
        backend = (backend or "easyocr").lower()
        if backend not in SUPPORTED_OCR_BACKENDS:
            raise ValueError(
                f"Desteklenmeyen OCR backend: {backend!r}. "
                f"Desteklenenler: {SUPPORTED_OCR_BACKENDS}"
            )
        self.backend = backend

        try:
            import easyocr
        except ImportError as exc:
            raise ImportError(
                "easyocr yüklü değil. `pip install easyocr` ile kurun."
            ) from exc

        self._reader = easyocr.Reader(
            languages or ["en"],
            gpu=use_gpu,
            verbose=False,
        )
        logger.info(
            f"PlateOCR ({backend}) yüklendi: "
            f"lang={languages or ['en']}, gpu={use_gpu}"
        )

    @staticmethod
    def _preprocess(plate_image: np.ndarray) -> np.ndarray:
        """Küçük plakaları Lanczos ile büyüt + hafif unsharp mask uygula."""
        h, w = plate_image.shape[:2]
        if w < MIN_OCR_WIDTH:
            scale = max(2, int(np.ceil(MIN_OCR_WIDTH / max(w, 1))))
            plate_image = cv2.resize(
                plate_image,
                (w * scale, h * scale),
                interpolation=cv2.INTER_LANCZOS4,
            )
            blur = cv2.GaussianBlur(plate_image, (0, 0), sigmaX=1.0)
            plate_image = cv2.addWeighted(plate_image, 1.5, blur, -0.5, 0)
        return plate_image

    def read(self, plate_image: np.ndarray) -> tuple[str, float]:
        """Plaka kırpmasından metin oku.

        Returns:
            (raw_text, mean_confidence)
        """
        if plate_image is None or plate_image.size == 0:
            return "", 0.0
        prepped = self._preprocess(plate_image)
        try:
            results = self._reader.readtext(prepped, detail=1)
        except Exception as exc:
            logger.warning(f"OCR çağrısı hatası: {exc}")
            return "", 0.0
        if not results:
            return "", 0.0

        # Birden fazla text region varsa hepsini birleştir, ortalama conf al
        texts = [r[1] for r in results]
        confs = [float(r[2]) for r in results]
        return " ".join(texts), float(np.mean(confs))

    def recognize(self, plate_image: np.ndarray) -> OCRResult:
        """Plaka oku ve TR formatına göre valide et."""
        raw_text, confidence = self.read(plate_image)
        plate_text = normalize_tr_plate(raw_text)
        is_valid, city_code, city_name = validate_tr_plate(plate_text)
        return OCRResult(
            raw_text=raw_text,
            plate_text=plate_text,
            confidence=confidence,
            is_valid=is_valid,
            city_code=city_code,
            city_name=city_name,
        )
