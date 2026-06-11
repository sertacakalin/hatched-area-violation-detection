"""Plaka karakter okuma (EasyOCR / PaddleOCR) + TR format validasyonu.

İki backend desteklenir:

- ``easyocr``: PyTorch tabanlı, Mac (MPS/CPU) ve Linux (CUDA) üzerinde
  sorunsuz çalışır, ``allowlist`` ile karakter setini kısıtlayabilir.
- ``paddleocr``: PP-OCRv5 tanıyıcı; küçük/orta boy (≈20-40 px yükseklik)
  plakalarda EasyOCR'a göre belirgin biçimde daha yüksek okuma oranı verir
  (ampirik karşılaştırma tezde Ch7'de raporlanır). allowlist'i doğrudan
  desteklemediği için çıktı son-işlemede filtrelenir.

OCR öncesi upscale: MOBESE geniş açılı kaynakta uzak araçların plakaları
30-50 piksel genişlikte olabilir, OCR bu boyutta düşük accuracy gösterir.
Plaka crop'unu Lanczos ile büyütüp hafif unsharp mask uygulamak okunabilirlik
oranını anlamlı artırır. Yine de ~16 px yükseklik altındaki plakalar fiziksel
olarak okunamaz (yeterli piksel bilgisi yok); bu sınır graceful degradation
olarak kabul edilir.
"""

import logging
import os
import re
from typing import TypedDict

import cv2
import numpy as np

from src.plate.tr_plate import repair_tr_plate, validate_tr_plate

MIN_OCR_WIDTH = 180  # Daha küçük plakalar Lanczos ile bu genişliğe çıkarılır
MIN_OCR_HEIGHT = 48
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

logger = logging.getLogger(__name__)


class OCRResult(TypedDict):
    raw_text: str
    plate_text: str
    confidence: float
    is_valid: bool
    city_code: str | None
    city_name: str | None


SUPPORTED_OCR_BACKENDS = ("easyocr", "paddleocr")


class PlateOCR:
    """EasyOCR / PaddleOCR sarmalayıcı + TR plaka format normalizasyonu."""

    def __init__(
        self,
        languages: list[str] | None = None,
        use_gpu: bool = False,
        backend: str = "easyocr",
        allowlist: str = OCR_ALLOWLIST,
    ):
        backend = (backend or "easyocr").lower()
        if backend not in SUPPORTED_OCR_BACKENDS:
            raise ValueError(
                f"Desteklenmeyen OCR backend: {backend!r}. "
                f"Desteklenenler: {SUPPORTED_OCR_BACKENDS}"
            )
        self.backend = backend
        self.allowlist = allowlist or OCR_ALLOWLIST
        self._allow_re = re.compile(f"[^{re.escape(self.allowlist)}]")
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu

        self._reader = None  # easyocr.Reader
        self._paddle = None  # paddleocr.PaddleOCR

        if backend == "easyocr":
            self._init_easyocr()
        else:
            self._init_paddleocr()

        logger.info(
            f"PlateOCR ({self.backend}) yüklendi: "
            f"lang={self.languages}, gpu={use_gpu}"
        )

    def _init_easyocr(self) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError(
                "easyocr yüklü değil. `pip install easyocr` ile kurun."
            ) from exc
        self._reader = easyocr.Reader(
            self.languages,
            gpu=self.use_gpu,
            verbose=False,
        )

    def _init_paddleocr(self) -> None:
        # Mac'te libomp çift-yükleme çökmesini ve yavaş model-kaynak
        # bağlantı kontrolünü önle (import'tan ÖNCE ayarlanmalı).
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            logger.warning(
                "paddleocr yüklü değil (%s) → easyocr backend'ine düşülüyor. "
                "Paddle için: pip install paddleocr paddlepaddle",
                exc,
            )
            self._fallback_to_easyocr()
            return
        # PP-OCRv5: textline orientation küçük plakalarda hafif faydalı.
        # Belge ön-işleme (unwarp/orientation) modülleri plaka için gereksiz,
        # kapatarak hızlandırırız.
        lang = "en"
        if self.languages and self.languages[0] not in ("en", "tr"):
            lang = self.languages[0]
        try:
            self._paddle = PaddleOCR(
                lang=lang,
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        except Exception as exc:
            logger.warning(
                "PaddleOCR kurulamadı (%s) → easyocr backend'ine düşülüyor.",
                exc,
            )
            self._fallback_to_easyocr()

    def _fallback_to_easyocr(self) -> None:
        """Paddle kullanılamadığında easyocr'a sessizce geç (plaka ölmesin).

        Config paddle-tercihli kalabilir; paddle kurulu değilse veya kurulum
        çökerse plaka tanıma tamamen kapanmak yerine easyocr ile çalışır.
        """
        self.backend = "easyocr"
        self._paddle = None
        self._init_easyocr()

    @staticmethod
    def _resize_for_ocr(plate_image: np.ndarray) -> np.ndarray:
        """Küçük plakaları min genişlik/yüksekliğe Lanczos ile büyüt."""
        h, w = plate_image.shape[:2]
        scale = max(
            1.0,
            MIN_OCR_WIDTH / max(w, 1),
            MIN_OCR_HEIGHT / max(h, 1),
        )
        if scale <= 1.0:
            return plate_image
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(
            plate_image,
            (new_w, new_h),
            interpolation=cv2.INTER_LANCZOS4,
        )

    @staticmethod
    def _sharpen(image: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(image, 1.5, blur, -0.5, 0)

    def _preprocess_variants(self, plate_image: np.ndarray) -> list[np.ndarray]:
        """OCR için birkaç düşük maliyetli görüntü varyantı üret.

        EasyOCR ikili (otsu/adaptive) görüntülerde iyi çalışır; PaddleOCR
        bunlarda accuracy kaybeder ve tek kanallı girişi sevmez. Bu yüzden
        backend'e göre varyant seti değişir ve PaddleOCR için her şey
        3-kanallı BGR'a çevrilir (hız + uyumluluk).
        """
        h, w = plate_image.shape[:2]
        pad_x = max(3, int(round(w * 0.08)))
        pad_y = max(3, int(round(h * 0.20)))
        padded = cv2.copyMakeBorder(
            plate_image,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            cv2.BORDER_REPLICATE,
        )
        resized = self._resize_for_ocr(padded)
        sharp = self._sharpen(resized)

        gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY) if sharp.ndim == 3 else sharp
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        if self.backend == "paddleocr":
            # PaddleOCR per-call maliyetli; renkli + CLAHE iki varyant yeter.
            clahe_bgr = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
            return [sharp, clahe_bgr]

        _, otsu = cv2.threshold(
            clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        adaptive = cv2.adaptiveThreshold(
            clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        return [sharp, gray, clahe, otsu, adaptive]

    def _read_variant(self, prepped: np.ndarray) -> tuple[str, float]:
        """Tek bir ön-işlenmiş görüntüden (raw_text, mean_conf) oku."""
        if self.backend == "paddleocr":
            return self._read_variant_paddle(prepped)
        return self._read_variant_easyocr(prepped)

    def _read_variant_easyocr(self, prepped: np.ndarray) -> tuple[str, float]:
        try:
            results = self._reader.readtext(
                prepped,
                detail=1,
                allowlist=self.allowlist,
            )
        except Exception as exc:
            logger.warning(f"EasyOCR çağrısı hatası: {exc}")
            return "", 0.0
        if not results:
            return "", 0.0
        texts = [r[1] for r in results]
        confs = [float(r[2]) for r in results]
        return " ".join(texts), float(np.mean(confs))

    def _read_variant_paddle(self, prepped: np.ndarray) -> tuple[str, float]:
        try:
            results = self._paddle.predict(prepped)
        except Exception as exc:
            logger.warning(f"PaddleOCR çağrısı hatası: {exc}")
            return "", 0.0
        texts: list[str] = []
        confs: list[float] = []
        for res in results or []:
            # PP-OCRv5 predict() çıktısı dict-benzeri: rec_texts / rec_scores
            rec_texts = res.get("rec_texts", []) if hasattr(res, "get") else []
            rec_scores = res.get("rec_scores", []) if hasattr(res, "get") else []
            for txt, score in zip(rec_texts, rec_scores):
                # PaddleOCR allowlist desteklemez → istenmeyen karakterleri at
                cleaned = self._allow_re.sub("", str(txt).upper())
                if not cleaned:
                    continue
                texts.append(cleaned)
                confs.append(float(score))
        if not texts:
            return "", 0.0
        return " ".join(texts), float(np.mean(confs))

    @staticmethod
    def _score_text(raw_text: str, confidence: float) -> float:
        plate_text = repair_tr_plate(raw_text)
        is_valid, _, _ = validate_tr_plate(plate_text)
        length_bonus = 1.0 if 5 <= len(plate_text) <= 9 else 0.65
        valid_bonus = 1.6 if is_valid else 1.0
        return confidence * length_bonus * valid_bonus

    def read(self, plate_image: np.ndarray) -> tuple[str, float]:
        """Plaka kırpmasından metin oku.

        Returns:
            (raw_text, mean_confidence)
        """
        if plate_image is None or plate_image.size == 0:
            return "", 0.0
        best_text = ""
        best_conf = 0.0
        best_score = -1.0

        for prepped in self._preprocess_variants(plate_image):
            raw_text, confidence = self._read_variant(prepped)
            if not raw_text:
                continue
            score = self._score_text(raw_text, confidence)
            if score > best_score:
                best_score = score
                best_text = raw_text
                best_conf = confidence

        return best_text, best_conf

    def recognize(self, plate_image: np.ndarray) -> OCRResult:
        """Plaka oku ve TR formatına göre valide et."""
        raw_text, confidence = self.read(plate_image)
        plate_text = repair_tr_plate(raw_text)
        is_valid, city_code, city_name = validate_tr_plate(plate_text)
        return OCRResult(
            raw_text=raw_text,
            plate_text=plate_text,
            confidence=confidence,
            is_valid=is_valid,
            city_code=city_code,
            city_name=city_name,
        )
