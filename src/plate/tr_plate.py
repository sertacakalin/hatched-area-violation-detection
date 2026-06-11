"""Türk plaka formatı kuralları + il kodu mapping.

TR plaka formatı: ``[01-81] [1-3 harf] [2-4 rakam]``
    Örnek: ``34 ABC 1234``, ``06 BBB 12``, ``35 K 1234``

İl kodları 01-81 arasında. Geçerli kod listesi resmi olarak bilinen
şehir kodlarıyla sınırlı tutuluyor.
"""

import re

TR_CITY_CODES: dict[str, str] = {
    "01": "Adana", "02": "Adıyaman", "03": "Afyonkarahisar", "04": "Ağrı",
    "05": "Amasya", "06": "Ankara", "07": "Antalya", "08": "Artvin",
    "09": "Aydın", "10": "Balıkesir", "11": "Bilecik", "12": "Bingöl",
    "13": "Bitlis", "14": "Bolu", "15": "Burdur", "16": "Bursa",
    "17": "Çanakkale", "18": "Çankırı", "19": "Çorum", "20": "Denizli",
    "21": "Diyarbakır", "22": "Edirne", "23": "Elazığ", "24": "Erzincan",
    "25": "Erzurum", "26": "Eskişehir", "27": "Gaziantep", "28": "Giresun",
    "29": "Gümüşhane", "30": "Hakkari", "31": "Hatay", "32": "Isparta",
    "33": "Mersin", "34": "İstanbul", "35": "İzmir", "36": "Kars",
    "37": "Kastamonu", "38": "Kayseri", "39": "Kırklareli", "40": "Kırşehir",
    "41": "Kocaeli", "42": "Konya", "43": "Kütahya", "44": "Malatya",
    "45": "Manisa", "46": "Kahramanmaraş", "47": "Mardin", "48": "Muğla",
    "49": "Muş", "50": "Nevşehir", "51": "Niğde", "52": "Ordu",
    "53": "Rize", "54": "Sakarya", "55": "Samsun", "56": "Siirt",
    "57": "Sinop", "58": "Sivas", "59": "Tekirdağ", "60": "Tokat",
    "61": "Trabzon", "62": "Tunceli", "63": "Şanlıurfa", "64": "Uşak",
    "65": "Van", "66": "Yozgat", "67": "Zonguldak", "68": "Aksaray",
    "69": "Bayburt", "70": "Karaman", "71": "Kırıkkale", "72": "Batman",
    "73": "Şırnak", "74": "Bartın", "75": "Ardahan", "76": "Iğdır",
    "77": "Yalova", "78": "Karabük", "79": "Kilis", "80": "Osmaniye",
    "81": "Düzce",
}

_TR_PLATE_BODY = re.compile(r"^([A-Z]{1,3})(\d{2,4})$")

# OCR'in plakada en sık karıştırdığı karakterler. Bu dönüşümler yalnızca
# pozisyon belli olduğunda kullanılır; normalize_tr_plate ham metni agresif
# biçimde değiştirmez.
_TO_DIGIT = str.maketrans({
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "T": "7",
})
_TO_LETTER = str.maketrans({
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
})


def normalize_tr_plate(raw: str) -> str:
    """OCR ham çıktısını TR plaka karakter setine indirger.

    - Boşluk/özel karakter siler
    - Küçük → büyük harf
    - Karıştırılan karakterleri **ölçülü** düzeltmez (false correction
      riski yüksek); sadece basit temizlik yapar.
    """
    if not raw:
        return ""
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def _translate_with_change_count(text: str, table: dict[int, str]) -> tuple[str, int]:
    out = []
    changes = 0
    for ch in text:
        mapped = table.get(ord(ch), ch)
        out.append(mapped)
        if mapped != ch:
            changes += 1
    return "".join(out), changes


def repair_tr_plate(raw: str) -> str:
    """OCR çıktısını TR plaka şablonuna göre kontrollü düzelt.

    Düzeltme kuralı pozisyon bazlıdır:
    - İlk iki karakter il kodu olduğu için rakam gibi değerlendirilir.
    - Orta bölüm harf, son bölüm rakam olarak denenir.
    - Yalnızca 01-81 il kodu ve TR gövde şablonu oluşuyorsa sonuç seçilir.

    Geçerli bir plaka zaten geldiyse aynen korunur. Geçerli aday yoksa
    normalize edilmiş ham metin döner.
    """
    cleaned = normalize_tr_plate(raw)
    is_valid, _, _ = validate_tr_plate(cleaned)
    if is_valid:
        return cleaned
    if len(cleaned) < 5 or len(cleaned) > 9:
        return cleaned

    city, city_changes = _translate_with_change_count(cleaned[:2], _TO_DIGIT)
    if not city.isdigit() or city not in TR_CITY_CODES:
        return cleaned

    rest = cleaned[2:]
    candidates: list[tuple[int, str]] = []
    for letter_count in range(1, 4):
        digit_count = len(rest) - letter_count
        if digit_count < 2 or digit_count > 4:
            continue

        letters, letter_changes = _translate_with_change_count(
            rest[:letter_count], _TO_LETTER
        )
        digits, digit_changes = _translate_with_change_count(
            rest[letter_count:], _TO_DIGIT
        )
        candidate = f"{city}{letters}{digits}"
        if _TR_PLATE_BODY.match(f"{letters}{digits}"):
            candidates.append((
                city_changes + letter_changes + digit_changes,
                candidate,
            ))

    if not candidates:
        return cleaned

    # En az müdahale edilen adayı seç; eşitlikte daha uzun harf bloğu yerine
    # daha az toplam değişiklik zaten belirleyici olur.
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def validate_tr_plate(text: str) -> tuple[bool, str | None, str | None]:
    """TR plaka formatına uygunluk kontrolü.

    Returns:
        (is_valid, city_code, city_name) — geçerli değilse city alanları
        kısmen dolu olabilir (örn. il kodu doğru ama kalan kısım hatalı).
    """
    cleaned = normalize_tr_plate(text)
    # TR plaka uzunluk aralığı: 2 + 1 + 2 = 5  →  2 + 3 + 4 = 9
    if len(cleaned) < 5 or len(cleaned) > 9:
        return False, None, None

    city_code = cleaned[:2]
    if not city_code.isdigit() or city_code not in TR_CITY_CODES:
        return False, None, None

    city_name = TR_CITY_CODES[city_code]
    rest = cleaned[2:]
    if _TR_PLATE_BODY.match(rest):
        return True, city_code, city_name

    return False, city_code, city_name
