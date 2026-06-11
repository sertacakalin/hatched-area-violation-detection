"""Plaka tespit + okuma modülü.

Pipeline akışı:
    Pipeline her frame'de update_buffer() çağırır → aktif track_id'lerin
    vehicle crop'ları ring buffer'a yazılır.
    Violation onaylandığında recognize(track_id) çağrılır → buffer'daki
    crop'lar üzerinde plate_detector + ocr çalıştırılır, en yüksek
    skorlu sonuç PlateResult olarak döner.
"""

from src.plate.detector import PlateDetector
from src.plate.ocr import PlateOCR
from src.plate.recognizer import PlateRecognizer
from src.plate.tr_plate import (
    TR_CITY_CODES,
    normalize_tr_plate,
    repair_tr_plate,
    validate_tr_plate,
)

__all__ = [
    "PlateDetector",
    "PlateOCR",
    "PlateRecognizer",
    "TR_CITY_CODES",
    "normalize_tr_plate",
    "repair_tr_plate",
    "validate_tr_plate",
]
