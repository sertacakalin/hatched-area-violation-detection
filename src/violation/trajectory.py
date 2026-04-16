"""Yörünge (trajectory) analizi — aracın bölge ile etkileşim metriklerini hesaplar.

Her takip edilen araç için konum geçmişi tutulur ve ihlal anında
giriş/çıkış noktaları, bölge içi mesafe, nüfuz derinliği ve geçiş açısı
hesaplanır. Bu metrikler severity modülüne girdi olarak verilir.
"""

import logging
import math
from dataclasses import dataclass, field

from shapely.geometry import Point, LineString, Polygon

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryMetrics:
    """Bir aracın bölge ile etkileşim metrikleri."""
    track_id: int
    entry_point: tuple[float, float] | None = None
    exit_point: tuple[float, float] | None = None
    in_zone_distance: float = 0.0        # Bölge içi toplam mesafe (px)
    in_zone_frames: int = 0              # Bölge içi kare sayısı
    crossing_angle: float = 0.0          # Hareket yönü ile kenar arası açı (derece)
    penetration_depth: float = 0.0       # Bölge merkezine yaklaşma oranı (0-1)
    positions_in_zone: list[tuple[float, float]] = field(default_factory=list)


class TrajectoryAnalyzer:
    """Her aracın pozisyon geçmişini tutarak yörünge metrikleri hesaplar.

    Pipeline akışında ViolationDetector tarafından çağrılır.
    İhlal tetiklendiğinde compute_metrics() ile metrikler döndürülür.

    Pozisyon smoothing (EMA) ile bbox titremesi azaltılır.
    """

    def __init__(self, max_history: int = 300, ema_alpha: float = 0.3):
        self.max_history = max_history
        self.ema_alpha = ema_alpha  # 0 = tam smooth, 1 = smoothing yok
        # track_id → konum listesi [(x, y), ...]
        self._histories: dict[int, list[tuple[float, float]]] = {}
        # track_id → bölge-içi konumlar
        self._zone_positions: dict[int, list[tuple[float, float]]] = {}
        # track_id → ilk giriş noktası
        self._entry_points: dict[int, tuple[float, float] | None] = {}
        # track_id → bölgede mi (önceki kare)
        self._was_in_zone: dict[int, bool] = {}
        # track_id → smoothed pozisyon (EMA)
        self._smoothed: dict[int, tuple[float, float]] = {}

    def update(self, track_id: int, position: tuple[float, float],
               is_in_zone: bool) -> None:
        """Her karede aracın konumunu kaydet (EMA smoothing ile)."""
        # İlk kare
        if track_id not in self._histories:
            self._histories[track_id] = []
            self._zone_positions[track_id] = []
            self._entry_points[track_id] = None
            self._was_in_zone[track_id] = False
            self._smoothed[track_id] = position

        # EMA smoothing — bbox titremesini azalt
        prev = self._smoothed[track_id]
        a = self.ema_alpha
        smoothed = (a * position[0] + (1 - a) * prev[0],
                    a * position[1] + (1 - a) * prev[1])
        self._smoothed[track_id] = smoothed
        position = smoothed

        history = self._histories[track_id]
        history.append(position)
        if len(history) > self.max_history:
            history.pop(0)

        was_in = self._was_in_zone.get(track_id, False)

        # Bölgeye giriş anını yakala
        if is_in_zone and not was_in:
            self._entry_points[track_id] = position
            self._zone_positions[track_id] = []

        # Bölge içi konumları biriktir
        if is_in_zone:
            self._zone_positions[track_id].append(position)

        self._was_in_zone[track_id] = is_in_zone

    def compute_metrics(self, track_id: int,
                        zone_polygon: Polygon) -> TrajectoryMetrics:
        """İhlal anında yörünge metriklerini hesapla."""
        zone_positions = self._zone_positions.get(track_id, [])
        entry_point = self._entry_points.get(track_id)

        metrics = TrajectoryMetrics(
            track_id=track_id,
            entry_point=entry_point,
            in_zone_frames=len(zone_positions),
            positions_in_zone=list(zone_positions),
        )

        if len(zone_positions) < 2:
            return metrics

        # Çıkış noktası = bölge-içi son konum
        metrics.exit_point = zone_positions[-1]

        # Bölge içi mesafe (ardışık noktalar arası toplam)
        metrics.in_zone_distance = self._calc_total_distance(zone_positions)

        # Nüfuz derinliği (0-1 arası, merkeze ne kadar yaklaştı)
        metrics.penetration_depth = self._calc_penetration_depth(
            zone_positions, zone_polygon
        )

        # Geçiş açısı (hareket yönü ile bölge kenarı arası)
        if entry_point is not None:
            metrics.crossing_angle = self._calc_crossing_angle(
                entry_point, zone_positions, zone_polygon
            )

        return metrics

    def cleanup_stale_tracks(self, active_track_ids: set[int]) -> None:
        """Artık takip edilmeyen araçların geçmişini temizle."""
        stale = set(self._histories.keys()) - active_track_ids
        for tid in stale:
            self._histories.pop(tid, None)
            self._zone_positions.pop(tid, None)
            self._entry_points.pop(tid, None)
            self._was_in_zone.pop(tid, None)
            self._smoothed.pop(tid, None)

    def reset(self) -> None:
        self._histories.clear()
        self._zone_positions.clear()
        self._entry_points.clear()
        self._was_in_zone.clear()
        self._smoothed.clear()

    # ── Hesaplama yardımcıları ──────────────────────────────────────

    @staticmethod
    def _calc_total_distance(points: list[tuple[float, float]]) -> float:
        """Ardışık noktalar arası öklidyen mesafe toplamı."""
        total = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            total += math.hypot(dx, dy)
        return total

    @staticmethod
    def _calc_penetration_depth(points: list[tuple[float, float]],
                                polygon: Polygon) -> float:
        """Aracın bölge merkezine en fazla ne kadar yaklaştığını hesapla.

        Returns:
            0.0 = kenarında kaldı, 1.0 = merkeze ulaştı
        """
        centroid = polygon.centroid
        # Bölge kenarından merkeze olan maksimum mesafe
        boundary = polygon.exterior
        max_possible = centroid.distance(boundary)
        if max_possible == 0:
            return 0.0

        # Her bölge-içi noktanın merkeze uzaklığını hesapla
        min_dist_to_center = float("inf")
        for px, py in points:
            dist = math.hypot(px - centroid.x, py - centroid.y)
            if dist < min_dist_to_center:
                min_dist_to_center = dist

        # Merkeze olan en yakın mesafeyi 0-1 oranına çevir
        depth = 1.0 - (min_dist_to_center / max_possible)
        return max(0.0, min(1.0, depth))

    @staticmethod
    def _calc_crossing_angle(entry_point: tuple[float, float],
                             zone_positions: list[tuple[float, float]],
                             polygon: Polygon) -> float:
        """Aracın hareket yönü ile bölge kenarı arasındaki açıyı hesapla.

        Returns:
            0° = kenar boyunca sürtme, 90° = dik geçiş
        """
        if len(zone_positions) < 2:
            return 0.0

        # Hareket vektörü: giriş → ilk birkaç bölge-içi noktanın ortalaması
        n = min(5, len(zone_positions))
        avg_x = sum(p[0] for p in zone_positions[:n]) / n
        avg_y = sum(p[1] for p in zone_positions[:n]) / n
        move_dx = avg_x - entry_point[0]
        move_dy = avg_y - entry_point[1]
        move_len = math.hypot(move_dx, move_dy)
        if move_len < 1e-6:
            return 0.0

        # Giriş noktasına en yakın kenar segmentini bul
        boundary = polygon.exterior
        entry_pt = Point(entry_point)
        nearest_dist = boundary.project(entry_pt)
        # Kenar üzerinde yakın iki nokta → kenar yönü
        d = 5.0  # küçük ofset
        p1 = boundary.interpolate(max(0, nearest_dist - d))
        p2 = boundary.interpolate(min(boundary.length, nearest_dist + d))
        edge_dx = p2.x - p1.x
        edge_dy = p2.y - p1.y
        edge_len = math.hypot(edge_dx, edge_dy)
        if edge_len < 1e-6:
            return 90.0

        # İki vektör arası açı (dot product)
        cos_angle = (move_dx * edge_dx + move_dy * edge_dy) / (move_len * edge_len)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_rad = math.acos(abs(cos_angle))
        return math.degrees(angle_rad)
