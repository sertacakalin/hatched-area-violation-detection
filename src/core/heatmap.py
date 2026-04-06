"""İhlal ısı haritası — ihlallerin yoğunlaştığı bölgeleri gösterir.

Tüm ihlal noktalarını Gaussian blur ile birleştirerek
sıcaklık haritası oluşturur. Tez figürü ve dashboard için.
"""

import cv2
import numpy as np
from pathlib import Path


class ViolationHeatmap:
    """Yol üzerinde ihlal yoğunluk haritası."""

    def __init__(self, width: int = 1080, height: int = 1920):
        self.width = width
        self.height = height
        self._accumulator = np.zeros((height, width), dtype=np.float32)

    def add_violation(self, position: tuple[float, float],
                      severity_score: float = 50.0) -> None:
        """İhlal noktası ekle. Skor yüksekse daha sıcak."""
        x, y = int(position[0]), int(position[1])
        if 0 <= x < self.width and 0 <= y < self.height:
            # Severity'ye göre ağırlık (yüksek skor = daha sıcak)
            weight = severity_score / 100.0
            cv2.circle(self._accumulator, (x, y), 25, weight, -1)

    def add_trajectory(self, positions: list[tuple[float, float]],
                       severity_score: float = 50.0) -> None:
        """İhlal yörüngesini ekle — çizgi boyunca ısı."""
        weight = severity_score / 100.0
        for i in range(1, len(positions)):
            pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
            pt2 = (int(positions[i][0]), int(positions[i][1]))
            cv2.line(self._accumulator, pt1, pt2, weight, 8)

    def render(self, background: np.ndarray | None = None,
               alpha: float = 0.6) -> np.ndarray:
        """Isı haritasını oluştur.

        Args:
            background: Arka plan görüntüsü (None = siyah)
            alpha: Heatmap saydamlığı (0-1)
        """
        # Gaussian blur ile yumuşat
        blurred = cv2.GaussianBlur(self._accumulator, (51, 51), 0)

        # Normalize (0-255)
        if blurred.max() > 0:
            normalized = (blurred / blurred.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(blurred, dtype=np.uint8)

        # Renk haritası (COLORMAP_JET: mavi=soğuk, kırmızı=sıcak)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # Sıfır olan yerleri şeffaf yap
        mask = normalized > 5
        mask_3ch = np.stack([mask] * 3, axis=-1)

        if background is not None:
            bg = background.copy()
            if bg.shape[:2] != (self.height, self.width):
                bg = cv2.resize(bg, (self.width, self.height))
        else:
            bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        result = bg.copy()
        result[mask_3ch] = cv2.addWeighted(
            heatmap, alpha, bg, 1 - alpha, 0
        )[mask_3ch]

        return result

    def save(self, output_path: str,
             background: np.ndarray | None = None) -> None:
        """Isı haritasını dosyaya kaydet."""
        result = self.render(background)

        # Başlık ekle
        cv2.putText(result, "IHLAL ISI HARITASI",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)

        # Renk skalası (legend)
        legend_x = self.width - 60
        for i in range(256):
            y = self.height - 50 - i
            color_bar = cv2.applyColorMap(
                np.array([[i]], dtype=np.uint8), cv2.COLORMAP_JET
            )[0][0].tolist()
            cv2.line(result, (legend_x, y), (legend_x + 30, y), color_bar, 1)

        cv2.putText(result, "Yuksek", (legend_x - 10, self.height - 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(result, "Dusuk", (legend_x - 10, self.height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, result)

    def reset(self) -> None:
        self._accumulator = np.zeros(
            (self.height, self.width), dtype=np.float32
        )
