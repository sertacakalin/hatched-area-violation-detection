"""Plaka görüntüsü ön-işleme — OCR doğruluğunu artırmak için."""

import cv2
import numpy as np


class PlatePreprocessor:
    """Plaka kırpmasını OCR için optimize eden ön-işleme pipeline'ı.

    Pipeline:
    1. Boyut büyütme (upscale)
    2. Gri tonlamaya dönüştürme
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    4. Gaussian bulanıklaştırma
    5. Adaptive threshold / Otsu threshold
    """

    def __init__(
        self,
        target_height: int = 80,
        clahe_clip: float = 2.0,
        clahe_grid: int = 8,
        blur_kernel: int = 3,
        use_adaptive_threshold: bool = False,
    ):
        self.target_height = target_height
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid),
        )
        self.blur_kernel = blur_kernel
        self.use_adaptive_threshold = use_adaptive_threshold

    def preprocess(self, plate_image: np.ndarray) -> np.ndarray:
        """Tam ön-işleme pipeline'ı uygula."""
        if plate_image.size == 0:
            return plate_image

        img = plate_image.copy()

        # 1. Boyut büyütme
        img = self._upscale(img)

        # 2. Gri tonlama
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. CLAHE
        img = self.clahe.apply(img)

        # 4. Gaussian blur
        if self.blur_kernel > 0:
            img = cv2.GaussianBlur(img, (self.blur_kernel, self.blur_kernel), 0)

        # 5. Threshold (opsiyonel)
        if self.use_adaptive_threshold:
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )

        return img

    def preprocess_for_paddleocr(self, plate_image: np.ndarray) -> np.ndarray:
        """PaddleOCR için optimize edilmiş ön-işleme.
        PaddleOCR kendi ön-işlemesini yapar, minimal müdahale yeterli.
        """
        if plate_image.size == 0:
            return plate_image

        img = self._upscale(plate_image)

        # CLAHE sadece düşük kontrastlı görüntülerde
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        std = gray.std()
        if std < 40:  # Düşük kontrast
            if len(img.shape) == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                img = self.clahe.apply(img)

        return img

    def preprocess_for_easyocr(self, plate_image: np.ndarray) -> np.ndarray:
        """EasyOCR için ön-işleme — gri tonlama + CLAHE."""
        if plate_image.size == 0:
            return plate_image

        img = self._upscale(plate_image)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = self.clahe.apply(img)
        return img

    def _upscale(self, img: np.ndarray) -> np.ndarray:
        """Hedef yüksekliğe ölçekle."""
        h, w = img.shape[:2]
        if h < self.target_height:
            scale = self.target_height / h
            new_w = int(w * scale)
            img = cv2.resize(img, (new_w, self.target_height),
                             interpolation=cv2.INTER_CUBIC)
        return img

    def get_all_variants(self, plate_image: np.ndarray) -> list[np.ndarray]:
        """OCR denemesi için birden fazla ön-işleme varyantı döndür."""
        variants = []

        # Orijinal (upscale)
        variants.append(self._upscale(plate_image))

        # Gri + CLAHE
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image
        gray = self._upscale(gray.reshape(gray.shape[0], gray.shape[1]))
        variants.append(self.clahe.apply(gray))

        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(thresh)

        return variants
