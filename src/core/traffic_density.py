"""Trafik yoğunluk analizi — ihlallerle yoğunluk korelasyonu.

Her karede takip edilen araç sayısını kaydeder.
İhlal anındaki yoğunluğu ihlal kaydına ekler.

Tezdeki argüman:
    "Yoğun trafikte araçlar mecburen taralı alana giriyor,
    bu durum false positive üretiyor. Şiddet skoru bu tür
    ihlallere düşük skor vererek filtrelemeyi mümkün kılıyor."
"""

import logging
from collections import deque

logger = logging.getLogger(__name__)


class TrafficDensityAnalyzer:
    """Kare bazında trafik yoğunluğu takibi."""

    def __init__(self, window_size: int = 90):
        self.window_size = window_size  # ~3 saniye @30fps
        self._counts: deque[int] = deque(maxlen=window_size)
        self._total_frames = 0

    def update(self, vehicle_count: int) -> None:
        """Her karede çağrılır."""
        self._counts.append(vehicle_count)
        self._total_frames += 1

    @property
    def current_density(self) -> float:
        """Son penceredeki ortalama araç sayısı."""
        if not self._counts:
            return 0.0
        return sum(self._counts) / len(self._counts)

    @property
    def density_level(self) -> str:
        """Yoğunluk seviyesi."""
        d = self.current_density
        if d < 5:
            return "SEYREK"
        if d < 15:
            return "NORMAL"
        if d < 25:
            return "YOGUN"
        return "SIKISIK"

    def get_violation_context(self) -> dict:
        """İhlal anında yoğunluk bağlamı."""
        return {
            "density": round(self.current_density, 1),
            "level": self.density_level,
            "min_in_window": min(self._counts) if self._counts else 0,
            "max_in_window": max(self._counts) if self._counts else 0,
        }
