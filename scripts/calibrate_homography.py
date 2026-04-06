"""Homografi kalibrasyon aracı — interaktif referans noktası seçimi.

Kullanım:
    python scripts/calibrate_homography.py --video VIDEO.mp4 --output configs/homography/cam1.json

Adımlar:
    1. Video'nun ilk karesi açılır
    2. Yol üzerinde 4+ referans noktası seçersin (sol tık)
    3. Her nokta için gerçek dünya koordinatını girersin (metre)
    4. Homografi matrisi hesaplanır ve kaydedilir

İpucu:
    - Şerit çizgileri, kaldırım kenarları, trafik adası köşeleri referans olarak uygundur
    - Şerit genişliği ≈ 3.5 metre (Türkiye standart)
    - Google Maps'te mesafe ölçme aracı kullanılabilir
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.homography import HomographyTransformer


def main():
    parser = argparse.ArgumentParser(description="Homografi kalibrasyon aracı")
    parser.add_argument("--video", required=True, help="Video dosyası")
    parser.add_argument("--output", default="configs/homography/calibration.json",
                        help="Çıktı JSON dosyası")
    parser.add_argument("--frame", type=int, default=0,
                        help="Kalibrasyon için kullanılacak kare numarası")
    args = parser.parse_args()

    # Video'dan kare al
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Video açılamadı: {args.video}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Kare okunamadı")
        return

    # Referans noktaları toplama
    pixel_points = []
    display = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel_points.append([x, y])
            display = frame.copy()
            for i, pt in enumerate(pixel_points):
                cv2.circle(display, (pt[0], pt[1]), 5, (0, 0, 255), -1)
                cv2.putText(display, f"P{i+1}", (pt[0]+8, pt[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if len(pixel_points) >= 2:
                for i in range(len(pixel_points) - 1):
                    cv2.line(display, tuple(pixel_points[i]),
                             tuple(pixel_points[i+1]), (0, 255, 0), 1)

    cv2.namedWindow("Kalibrasyon - Sol tik: nokta ekle, Q: bitir", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Kalibrasyon - Sol tik: nokta ekle, Q: bitir", mouse_callback)

    print("\n=== HOMOGRAFI KALIBRASYONU ===")
    print("Sol tıkla: Referans noktası ekle (en az 4)")
    print("Q: Noktaları bitir ve koordinat girişine geç\n")

    while True:
        cv2.imshow("Kalibrasyon - Sol tik: nokta ekle, Q: bitir", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") and len(pixel_points) >= 4:
            break
        if key == ord("q") and len(pixel_points) < 4:
            print(f"En az 4 nokta gerekli (şu an {len(pixel_points)})")

    cv2.destroyAllWindows()

    # Gerçek dünya koordinatlarını al
    print(f"\n{len(pixel_points)} referans noktası seçildi.")
    print("Her nokta için gerçek dünya koordinatlarını girin (metre cinsinden).")
    print("İpucu: Şerit genişliği ≈ 3.5m, araç uzunluğu ≈ 4.5m\n")

    world_points = []
    for i, px in enumerate(pixel_points):
        print(f"P{i+1} (piksel: {px[0]}, {px[1]})")
        x = float(input(f"  X (metre, yatay): "))
        y = float(input(f"  Y (metre, dikey): "))
        world_points.append([x, y])

    # Homografi hesapla
    transformer = HomographyTransformer()
    error = transformer.calibrate(pixel_points, world_points)

    print(f"\nReprojection error: {error:.3f} piksel")
    if error < 5.0:
        print("Kalite: İYİ")
    elif error < 15.0:
        print("Kalite: KABUL EDİLEBİLİR")
    else:
        print("Kalite: KÖTÜ — noktaları tekrar seçmeyi dene")

    # Test: noktalar arası mesafe
    print("\n--- Mesafe Testi ---")
    for i in range(len(pixel_points) - 1):
        dist = transformer.pixel_distance_to_meters(
            tuple(pixel_points[i]), tuple(pixel_points[i+1])
        )
        print(f"P{i+1} → P{i+2}: {dist:.2f} metre")

    # Bird's eye view göster
    bev = transformer.get_birds_eye_view(frame)
    cv2.imshow("Bird's Eye View", bev)
    print("\nBird's eye view gösteriliyor. Herhangi bir tuşa bas...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Kaydet
    transformer.save(args.output)
    print(f"\nKalibrasyon kaydedildi: {args.output}")

    # BEV görüntüsünü de kaydet
    bev_path = str(Path(args.output).parent / "birds_eye_view.png")
    cv2.imwrite(bev_path, bev)
    print(f"Bird's eye view kaydedildi: {bev_path}")


if __name__ == "__main__":
    main()
