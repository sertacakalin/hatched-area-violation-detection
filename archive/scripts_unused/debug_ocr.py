"""Mevcut plaka crop'ları üzerinde OCR debug — upscale stratejilerini karşılaştırır.

Pipeline tekrar koşturmaya gerek yok; results/plates/ altındaki dosyaları
doğrudan EasyOCR'a verip farklı upscale faktörlerinin sonucunu yazdırır.
"""

import sys
from pathlib import Path

import cv2
import easyocr
import numpy as np


def upscale_lanczos(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_LANCZOS4)


def upscale_with_sharpen(img: np.ndarray, factor: int) -> np.ndarray:
    big = upscale_lanczos(img, factor)
    # Hafif unsharp mask — kontrastı artırır
    blur = cv2.GaussianBlur(big, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(big, 1.5, blur, -0.5, 0)


def main():
    plates_dir = Path("results/plates")
    if not plates_dir.exists():
        print(f"Klasör yok: {plates_dir}")
        sys.exit(1)

    plate_files = sorted(plates_dir.glob("*.jpg"))
    if not plate_files:
        print("Plaka dosyası yok")
        sys.exit(1)

    print(f"{len(plate_files)} plaka dosyası bulundu, OCR test ediliyor...\n")

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    debug_dir = Path("results/plates_debug")
    debug_dir.mkdir(exist_ok=True)

    for f in plate_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        print(f"\n=== {f.name} ({w}x{h}) ===")

        variants = [
            ("orijinal",       img),
            ("2x lanczos",     upscale_lanczos(img, 2)),
            ("3x lanczos",     upscale_lanczos(img, 3)),
            ("4x lanczos",     upscale_lanczos(img, 4)),
            ("4x + sharpen",   upscale_with_sharpen(img, 4)),
            ("6x + sharpen",   upscale_with_sharpen(img, 6)),
        ]

        for name, variant in variants:
            try:
                results = reader.readtext(variant, detail=1)
            except Exception as exc:
                print(f"  {name:20s}: ERROR {exc}")
                continue
            if not results:
                print(f"  {name:20s}: (boş)")
                continue
            txt = " ".join(r[1] for r in results)
            conf = float(np.mean([r[2] for r in results]))
            vh, vw = variant.shape[:2]
            print(f"  {name:20s} ({vw:>4}x{vh:>3}px) → conf={conf:.2f}  text={txt!r}")

        # Debug: en büyük varyantı kaydet
        cv2.imwrite(str(debug_dir / f"{f.stem}_6x.jpg"), upscale_with_sharpen(img, 6))


if __name__ == "__main__":
    main()
