"""Polygon tabanlı bölge yönetimi — JSON'dan yükleme ve point-in-polygon kontrolü."""

import json
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon, box

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """Tek bir taralı alan bölgesi."""
    zone_id: str
    name: str
    polygon: Polygon
    zone_type: str = "hatched_area"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.polygon.bounds

    @property
    def area(self) -> float:
        return self.polygon.area

    @property
    def exterior_coords(self) -> np.ndarray:
        return np.array(self.polygon.exterior.coords[:-1], dtype=np.int32)


class ZoneManager:
    """Birden fazla bölgeyi yöneten sınıf."""

    def __init__(self, zone_file: str | Path | None = None,
                 polygon_buffer: float = -10):
        self.zones: list[Zone] = []
        self.polygon_buffer = polygon_buffer

        if zone_file is not None:
            self.load_zones(zone_file)

    def load_zones(self, zone_file: str | Path) -> None:
        """JSON dosyasından bölgeleri yükle."""
        path = Path(zone_file)
        if not path.exists():
            raise FileNotFoundError(f"Bölge dosyası bulunamadı: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.zones = []
        for zone_data in data.get("zones", []):
            coords = zone_data["polygon"]
            polygon = Polygon(coords)

            # Negatif buffer: polygon'u içe doğru küçült (gürültü azaltma)
            if self.polygon_buffer != 0:
                buffered = polygon.buffer(self.polygon_buffer)
                if not buffered.is_empty and buffered.area > 0:
                    polygon = buffered

            zone = Zone(
                zone_id=zone_data["zone_id"],
                name=zone_data.get("name", zone_data["zone_id"]),
                polygon=polygon,
                zone_type=zone_data.get("type", "hatched_area"),
            )
            self.zones.append(zone)

        logger.info(f"{len(self.zones)} bölge yüklendi: {path.name}")

    def set_zone_from_points(self, zone_id: str, points: list[list[int]],
                             name: str = "") -> None:
        """Programatik olarak bölge ekle."""
        polygon = Polygon(points)
        if self.polygon_buffer != 0:
            buffered = polygon.buffer(self.polygon_buffer)
            if not buffered.is_empty and buffered.area > 0:
                polygon = buffered

        zone = Zone(
            zone_id=zone_id,
            name=name or zone_id,
            polygon=polygon,
        )
        self.zones.append(zone)

    def is_point_in_zone(self, point: tuple[float, float],
                         zone_id: str | None = None) -> tuple[bool, str | None]:
        """Noktanın herhangi bir bölgede olup olmadığını kontrol et."""
        pt = Point(point)
        zones_to_check = self.zones
        if zone_id:
            zones_to_check = [z for z in self.zones if z.zone_id == zone_id]

        for zone in zones_to_check:
            if zone.polygon.contains(pt):
                return True, zone.zone_id
        return False, None

    def get_bbox_overlap_ratio(self, bbox: np.ndarray,
                               zone_id: str | None = None) -> tuple[float, str | None]:
        """Bbox ile bölge arasındaki örtüşme oranını hesapla."""
        x1, y1, x2, y2 = bbox
        bbox_poly = box(x1, y1, x2, y2)
        bbox_area = bbox_poly.area

        if bbox_area == 0:
            return 0.0, None

        zones_to_check = self.zones
        if zone_id:
            zones_to_check = [z for z in self.zones if z.zone_id == zone_id]

        max_ratio = 0.0
        max_zone_id = None
        for zone in zones_to_check:
            intersection = bbox_poly.intersection(zone.polygon)
            ratio = intersection.area / bbox_area
            if ratio > max_ratio:
                max_ratio = ratio
                max_zone_id = zone.zone_id

        return max_ratio, max_zone_id

    def save_zones(self, output_path: str | Path,
                   frame_width: int = 1920, frame_height: int = 1080) -> None:
        """Bölgeleri JSON dosyasına kaydet."""
        data = {
            "camera_id": "camera_01",
            "frame_width": frame_width,
            "frame_height": frame_height,
            "zones": [],
        }
        for zone in self.zones:
            coords = list(zone.polygon.exterior.coords[:-1])
            data["zones"].append({
                "zone_id": zone.zone_id,
                "name": zone.name,
                "polygon": [[int(x), int(y)] for x, y in coords],
                "type": zone.zone_type,
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Bölgeler kaydedildi: {output_path}")

    def get_zone_polygons_for_drawing(self) -> list[tuple[str, np.ndarray]]:
        """Görselleştirme için bölge ismi ve koordinatlarını döndür."""
        result = []
        for zone in self.zones:
            coords = np.array(zone.polygon.exterior.coords[:-1], dtype=np.int32)
            result.append((zone.name, coords))
        return result
