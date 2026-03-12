"""Türk plaka format doğrulama — regex tabanlı."""

import re
import logging

logger = logging.getLogger(__name__)

# Türk plaka formatları:
# İl kodu (01-81) + Harf(ler) (1-3) + Rakam(lar) (2-4)
# Örnekler: 34ABC1234, 06AB123, 35A1234, 01AAA12
TURKISH_PLATE_PATTERNS = [
    # Format 1: İl(2) + Harf(1) + Rakam(4)  → 34A1234
    r"^(0[1-9]|[1-7][0-9]|8[01])\s?[A-Z]\s?\d{4}$",
    # Format 2: İl(2) + Harf(1) + Rakam(3)  → 34A123
    r"^(0[1-9]|[1-7][0-9]|8[01])\s?[A-Z]\s?\d{3}$",
    # Format 3: İl(2) + Harf(2) + Rakam(3)  → 34AB123
    r"^(0[1-9]|[1-7][0-9]|8[01])\s?[A-Z]{2}\s?\d{3}$",
    # Format 4: İl(2) + Harf(2) + Rakam(4)  → 34AB1234
    r"^(0[1-9]|[1-7][0-9]|8[01])\s?[A-Z]{2}\s?\d{4}$",
    # Format 5: İl(2) + Harf(3) + Rakam(2)  → 34ABC12
    r"^(0[1-9]|[1-7][0-9]|8[01])\s?[A-Z]{3}\s?\d{2}$",
    # Format 6: İl(2) + Harf(3) + Rakam(3)  → 34ABC123
    r"^(0[1-9]|[1-7][0-9]|8[01])\s?[A-Z]{3}\s?\d{3}$",
    # Format 7: İl(2) + Harf(3) + Rakam(4)  → 34ABC1234
    r"^(0[1-9]|[1-7][0-9]|8[01])\s?[A-Z]{3}\s?\d{4}$",
]

# Tüm formatları tek regex'e birleştir
COMBINED_PATTERN = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])\s?[A-Z]{1,3}\s?\d{2,4}$"
)

# İl kodları tablosu
CITY_CODES = {
    1: "Adana", 2: "Adıyaman", 3: "Afyonkarahisar", 4: "Ağrı",
    5: "Amasya", 6: "Ankara", 7: "Antalya", 8: "Artvin",
    9: "Aydın", 10: "Balıkesir", 11: "Bilecik", 12: "Bingöl",
    13: "Bitlis", 14: "Bolu", 15: "Burdur", 16: "Bursa",
    17: "Çanakkale", 18: "Çankırı", 19: "Çorum", 20: "Denizli",
    21: "Diyarbakır", 22: "Edirne", 23: "Elazığ", 24: "Erzincan",
    25: "Erzurum", 26: "Eskişehir", 27: "Gaziantep", 28: "Giresun",
    29: "Gümüşhane", 30: "Hakkari", 31: "Hatay", 32: "Isparta",
    33: "Mersin", 34: "İstanbul", 35: "İzmir", 36: "Kars",
    37: "Kastamonu", 38: "Kayseri", 39: "Kırklareli", 40: "Kırşehir",
    41: "Kocaeli", 42: "Konya", 43: "Kütahya", 44: "Malatya",
    45: "Manisa", 46: "Kahramanmaraş", 47: "Mardin", 48: "Muğla",
    49: "Muş", 50: "Nevşehir", 51: "Niğde", 52: "Ordu",
    53: "Rize", 54: "Sakarya", 55: "Samsun", 56: "Siirt",
    57: "Sinop", 58: "Sivas", 59: "Tekirdağ", 60: "Tokat",
    61: "Trabzon", 62: "Tunceli", 63: "Şanlıurfa", 64: "Uşak",
    65: "Van", 66: "Yozgat", 67: "Zonguldak", 68: "Aksaray",
    69: "Bayburt", 70: "Karaman", 71: "Kırıkkale", 72: "Batman",
    73: "Şırnak", 74: "Bartın", 75: "Ardahan", 76: "Iğdır",
    77: "Yalova", 78: "Karabük", 79: "Kilis", 80: "Osmaniye",
    81: "Düzce",
}


class PlateValidator:
    """Türk plaka formatını doğrulayan sınıf."""

    def __init__(self, pattern: str | None = None):
        if pattern:
            self._pattern = re.compile(pattern)
        else:
            self._pattern = COMBINED_PATTERN

    def validate(self, plate_text: str) -> bool:
        """Plaka metninin Türk formatına uyup uymadığını kontrol et."""
        if not plate_text:
            return False
        # Boşlukları kaldır
        cleaned = plate_text.replace(" ", "").upper()
        return bool(self._pattern.match(cleaned))

    def get_city(self, plate_text: str) -> str | None:
        """Plaka metninden il ismini döndür."""
        cleaned = plate_text.replace(" ", "").upper()
        match = re.match(r"^(\d{2})", cleaned)
        if match:
            code = int(match.group(1))
            return CITY_CODES.get(code)
        return None

    def format_plate(self, plate_text: str) -> str:
        """Plaka metnini standart formata çevir: '34 ABC 1234'."""
        cleaned = plate_text.replace(" ", "").upper()
        match = re.match(r"^(\d{2})([A-Z]{1,3})(\d{2,4})$", cleaned)
        if match:
            return f"{match.group(1)} {match.group(2)} {match.group(3)}"
        return cleaned

    def validate_detailed(self, plate_text: str) -> dict:
        """Detaylı doğrulama sonucu döndür."""
        cleaned = plate_text.replace(" ", "").upper()
        result = {
            "input": plate_text,
            "cleaned": cleaned,
            "is_valid": False,
            "city_code": None,
            "city_name": None,
            "letters": None,
            "numbers": None,
            "formatted": None,
        }

        match = re.match(r"^(\d{2})([A-Z]{1,3})(\d{2,4})$", cleaned)
        if match and self.validate(cleaned):
            code = int(match.group(1))
            result.update({
                "is_valid": True,
                "city_code": match.group(1),
                "city_name": CITY_CODES.get(code),
                "letters": match.group(2),
                "numbers": match.group(3),
                "formatted": f"{match.group(1)} {match.group(2)} {match.group(3)}",
            })

        return result
