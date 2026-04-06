"""Şiddet (severity) skorlama ve ihlal tipi sınıflandırma.

Yörünge metriklerini birleştirerek 0-100 arası şiddet skoru hesaplar
ve ihlali KAYNAK / SEYİR / KENAR_TEMASI olarak sınıflandırır.
"""

import logging
from dataclasses import dataclass
from enum import Enum

from src.violation.trajectory import TrajectoryMetrics

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """İhlal tipi sınıflandırması."""
    KAYNAK = "KAYNAK"              # Çapraz geçiş, şerit değiştirme
    SEYIR = "SEYİR"                # Bölge içinde ilerleme
    KENAR_TEMASI = "KENAR_TEMASI"  # Kenardan sürtüp geçmiş
    DIGER = "DİĞER"


class SeverityLevel(Enum):
    """Şiddet seviyesi."""
    DUSUK = "DÜŞÜK"        # 0-25: muhtemelen false positive
    ORTA = "ORTA"          # 25-50: hafif ihlal
    YUKSEK = "YÜKSEK"      # 50-75: net ihlal
    KRITIK = "KRİTİK"      # 75-100: uzun süre taralı alanda seyir


@dataclass
class SeverityResult:
    """Şiddet skorlama sonucu."""
    score: float                    # 0-100
    level: SeverityLevel
    violation_type: ViolationType
    components: dict                # Bileşen detayları (debug/tez için)


class SeverityScorer:
    """Çok boyutlu şiddet skorlama motoru.

    Formül:
        skor = w1 * norm(süre) + w2 * norm(mesafe) +
               w3 * norm(derinlik) + w4 * norm(açı)

    Ağırlıklar ve normalizasyon referans değerleri deneylerle
    kalibre edilir. Varsayılan değerler makul başlangıç noktasıdır.
    """

    def __init__(
        self,
        w_duration: float = 0.30,
        w_distance: float = 0.25,
        w_depth: float = 0.30,
        w_angle: float = 0.15,
        ref_duration_frames: float = 30.0,   # ~1 sn @30fps → tam skor
        ref_distance_px: float = 200.0,      # 200px → tam skor
    ):
        self.w_duration = w_duration
        self.w_distance = w_distance
        self.w_depth = w_depth
        self.w_angle = w_angle
        self.ref_duration_frames = ref_duration_frames
        self.ref_distance_px = ref_distance_px

        total = w_duration + w_distance + w_depth + w_angle
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Ağırlık toplamı 1.0 değil: {total:.2f}")

    def score(self, metrics: TrajectoryMetrics) -> SeverityResult:
        """Yörünge metriklerinden şiddet skoru hesapla."""
        # Normalize (0-1 arası, clamp)
        norm_duration = min(1.0, metrics.in_zone_frames / self.ref_duration_frames)
        norm_distance = min(1.0, metrics.in_zone_distance / self.ref_distance_px)
        norm_depth = metrics.penetration_depth  # zaten 0-1
        norm_angle = min(1.0, metrics.crossing_angle / 90.0)

        raw_score = (
            self.w_duration * norm_duration
            + self.w_distance * norm_distance
            + self.w_depth * norm_depth
            + self.w_angle * norm_angle
        )
        score = round(raw_score * 100, 1)
        score = max(0.0, min(100.0, score))

        level = self._classify_level(score)
        violation_type = self._classify_type(metrics)

        components = {
            "duration": {"raw": metrics.in_zone_frames,
                         "normalized": round(norm_duration, 3),
                         "weight": self.w_duration},
            "distance": {"raw": round(metrics.in_zone_distance, 1),
                         "normalized": round(norm_distance, 3),
                         "weight": self.w_distance},
            "depth":    {"raw": round(metrics.penetration_depth, 3),
                         "normalized": round(norm_depth, 3),
                         "weight": self.w_depth},
            "angle":    {"raw": round(metrics.crossing_angle, 1),
                         "normalized": round(norm_angle, 3),
                         "weight": self.w_angle},
        }

        result = SeverityResult(
            score=score,
            level=level,
            violation_type=violation_type,
            components=components,
        )

        logger.info(
            f"Track {metrics.track_id}: skor={score}, "
            f"seviye={level.value}, tip={violation_type.value}"
        )
        return result

    @staticmethod
    def _classify_level(score: float) -> SeverityLevel:
        if score < 25:
            return SeverityLevel.DUSUK
        if score < 50:
            return SeverityLevel.ORTA
        if score < 75:
            return SeverityLevel.YUKSEK
        return SeverityLevel.KRITIK

    @staticmethod
    def _classify_type(metrics: TrajectoryMetrics) -> ViolationType:
        """Yörünge metriklerine göre ihlal tipini belirle.

        Karar ağacı (öncelik sırasıyla):
            1. Sığ nüfuz (derinlik < 0.30) → KENAR_TEMASI
               Araç bölgenin kenarından sürtüp geçmiş.
            2. Kısa süre (< 15 kare) + derinlik ≥ 0.30 → KAYNAK
               Hızlıca bölgeyi kesip geçmiş (şerit değiştirme).
            3. Uzun süre (≥ 15 kare) → SEYİR
               Bölge içinde ilerlemiş (taralı alanda seyir).
            4. Diğer durumlar → DİĞER
        """
        duration = metrics.in_zone_frames
        depth = metrics.penetration_depth

        if depth < 0.30:
            return ViolationType.KENAR_TEMASI
        if duration < 15:
            return ViolationType.KAYNAK
        if duration >= 15:
            return ViolationType.SEYIR
        return ViolationType.DIGER
