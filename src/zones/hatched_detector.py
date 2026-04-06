"""Otomatik Taralı Alan Tespiti — Klasik Görüntü İşleme.

Algoritma:
    Video → Frame Stabilizasyon → Background Subtraction (medyan)
    → Araçsız temiz yol → Ön İşleme (Gray + CLAHE + Adaptive Threshold)
    → Canny Edge → Hough Lines → Açı Filtreleme (25°-65° çapraz)
    → DBSCAN Kümeleme → Convex Hull → Morfolojik Düzeltme → Polygon

Özgün Katkılar:
    1. ORB feature matching ile frame stabilizasyon (el kamerası desteği)
    2. Çok aşamalı çizgi tespiti (Canny + Adaptive Threshold birleşimi)
    3. Açı + uzunluk + yoğunluk bazlı çapraz çizgi filtreleme
    4. DBSCAN kümeleme ile çoklu taralı alan tespiti
    5. Morfolojik polygon düzeltme (dilate + erode + convex hull)

Yazar: Sertaç Akalın — İstanbul Arel Üniversitesi (220303053)
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


class HatchedAreaDetector:
    """Videodan otomatik taralı alan tespit eden sınıf.

    İki modda çalışır:
    1. Sabit kamera: Doğrudan medyan background
    2. El kamerası: Stabilizasyon + medyan background
    """

    def __init__(
        self,
        # Background extraction
        sample_count: int = 200,
        sample_interval: int = 30,
        stabilize: bool = True,
        # Line detection
        canny_low: int = 40,
        canny_high: int = 120,
        hough_threshold: int = 50,
        min_line_length: int = 20,
        max_line_gap: int = 15,
        # Angle filtering
        min_angle: float = 20.0,
        max_angle: float = 70.0,
        # Clustering
        dbscan_eps: float = 60.0,
        dbscan_min_samples: int = 3,
        # Preprocessing
        clahe_clip: float = 3.0,
        clahe_grid: int = 8,
        # Polygon
        min_polygon_area: float = 1500.0,
        polygon_buffer: float = 15.0,
    ):
        self.sample_count = sample_count
        self.sample_interval = sample_interval
        self.stabilize = stabilize
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.min_polygon_area = min_polygon_area
        self.polygon_buffer = polygon_buffer
        self._valid_mask = None

    # ════════════════════════════════════════════════════════════
    # Adım 0: Frame Stabilizasyon (EL KAMERASI İÇİN)
    # ════════════════════════════════════════════════════════════

    def stabilize_frames(
        self, frames: list[np.ndarray]
    ) -> list[np.ndarray]:
        """ORB feature matching ile frame'leri ilk frame'e hizala.

        El kamerasında titreme olduğunda medyan bulanıklaşır.
        Bu fonksiyon tüm frame'leri referans frame'e göre
        affine transform ile hizalar.

        Yöntem:
            1. Referans frame (ilk frame) seç
            2. Her frame'de ORB keypoint'leri bul
            3. BFMatcher ile eşleştir
            4. Affine transform hesapla (en az 3 nokta)
            5. warpAffine ile hizala
        """
        if len(frames) < 2:
            return frames

        logger.info(f"Frame stabilizasyon başlatılıyor ({len(frames)} frame)...")

        ref_frame = frames[0]
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=500)
        ref_kp, ref_desc = orb.detectAndCompute(ref_gray, None)

        if ref_desc is None:
            logger.warning("Referans frame'de feature bulunamadı, stabilizasyon atlanıyor")
            return frames

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        stabilized = [ref_frame]
        success_count = 0

        for i, frame in enumerate(frames[1:], 1):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, desc = orb.detectAndCompute(gray, None)

            if desc is None or len(kp) < 3:
                stabilized.append(frame)
                continue

            matches = bf.match(ref_desc, desc)
            matches = sorted(matches, key=lambda m: m.distance)

            # En iyi eşleşmeleri al
            good_matches = matches[:min(50, len(matches))]

            if len(good_matches) < 3:
                stabilized.append(frame)
                continue

            # Kaynak ve hedef noktaları
            src_pts = np.float32(
                [ref_kp[m.queryIdx].pt for m in good_matches]
            )
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in good_matches]
            )

            # Affine transform (RANSAC ile outlier'ları ele)
            M, inliers = cv2.estimateAffinePartial2D(
                dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
            )

            if M is not None:
                # Büyük dönüşleri reddet (sadece küçük titreme düzelt)
                # M = [[cos*s, -sin*s, tx], [sin*s, cos*s, ty]]
                tx, ty = M[0, 2], M[1, 2]
                scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
                angle_rad = np.arctan2(M[1, 0], M[0, 0])
                angle_deg = abs(np.degrees(angle_rad))

                # Maksimum izin verilen hareket
                if abs(tx) > 50 or abs(ty) > 50 or angle_deg > 3.0 or abs(scale - 1.0) > 0.1:
                    stabilized.append(frame)  # Çok büyük hareket, orijinali kullan
                    continue

                h, w = ref_frame.shape[:2]
                aligned = cv2.warpAffine(frame, M, (w, h))
                stabilized.append(aligned)
                success_count += 1
            else:
                stabilized.append(frame)

        logger.info(
            f"Stabilizasyon tamamlandı: {success_count}/{len(frames)-1} "
            f"frame hizalandı"
        )
        return stabilized

    # ════════════════════════════════════════════════════════════
    # Adım 1: Background Subtraction
    # ════════════════════════════════════════════════════════════

    def extract_background(self, video_path: str) -> np.ndarray:
        """Videodan araçsız temiz arka plan görüntüsü çıkarır.

        Yöntem: N kare örnekle → (opsiyonel stabilizasyon) → medyan.
        Hareket eden nesneler (araçlar) kaybolur, sabit şeyler kalır.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Video açılamadı: {video_path}")

        frames = []
        frame_idx = 0

        logger.info(
            f"Arka plan çıkarma: {self.sample_count} kare, "
            f"her {self.sample_interval} karede bir"
        )

        while len(frames) < self.sample_count:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.sample_interval == 0:
                frames.append(frame)
            frame_idx += 1

        cap.release()

        if len(frames) < 10:
            raise ValueError(f"Yeterli kare yok: {len(frames)} (min 10)")

        logger.info(f"{len(frames)} kare örneklendi")

        # Stabilizasyon (el kamerası için)
        if self.stabilize:
            frames = self.stabilize_frames(frames)

        # Piksel bazında medyan
        logger.info("Medyan hesaplanıyor...")
        background = np.median(np.array(frames), axis=0).astype(np.uint8)

        # Siyah kenarları tespit et ve maskele (stabilizasyon artifact'ı)
        gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_bg, 10, 255, cv2.THRESH_BINARY)
        # Maskeyi küçült (kenar artifact'larını temizle)
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        self._valid_mask = mask

        logger.info("Arka plan çıkarma tamamlandı")
        return background

    # ════════════════════════════════════════════════════════════
    # Adım 2: Çok Aşamalı Ön İşleme
    # ════════════════════════════════════════════════════════════

    def preprocess(self, background: np.ndarray) -> dict:
        """Çok aşamalı ön işleme — farklı yöntemlerle çizgi çıkarma.

        Returns:
            dict with 'enhanced', 'adaptive', 'edges_canny',
            'edges_adaptive', 'edges_combined'
        """
        gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # CLAHE kontrast artırma
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_grid, self.clahe_grid),
        )
        enhanced = clahe.apply(blurred)

        # Adaptive threshold — yerel parlaklık farklarını yakalar
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=3,
        )

        # Canny edge detection
        edges_canny = cv2.Canny(enhanced, self.canny_low, self.canny_high)

        # Adaptive threshold'dan edge
        kernel = np.ones((3, 3), np.uint8)
        edges_adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_GRADIENT, kernel)

        # Birleşik edge map (OR) — daha fazla çizgi yakalar
        edges_combined = cv2.bitwise_or(edges_canny, edges_adaptive)

        # Geçerli bölge maskesi uygula (siyah kenarları kaldır)
        if hasattr(self, "_valid_mask") and self._valid_mask is not None:
            edges_canny = cv2.bitwise_and(edges_canny, self._valid_mask)
            edges_adaptive = cv2.bitwise_and(edges_adaptive, self._valid_mask)
            edges_combined = cv2.bitwise_and(edges_combined, self._valid_mask)

        return {
            "gray": gray,
            "enhanced": enhanced,
            "adaptive": adaptive,
            "edges_canny": edges_canny,
            "edges_adaptive": edges_adaptive,
            "edges_combined": edges_combined,
        }

    # ════════════════════════════════════════════════════════════
    # Adım 3: Çok Kaynaklı Çizgi Tespiti
    # ════════════════════════════════════════════════════════════

    def detect_lines(self, preprocess_result: dict) -> np.ndarray | None:
        """Birden fazla edge map'ten çizgi tespit et ve birleştir."""

        all_lines = []

        for edge_name in ["edges_canny", "edges_adaptive", "edges_combined"]:
            edges = preprocess_result[edge_name]
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=self.hough_threshold,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap,
            )
            if lines is not None:
                all_lines.append(lines)
                logger.info(f"  {edge_name}: {len(lines)} çizgi")

        if not all_lines:
            logger.warning("Hiçbir edge map'ten çizgi bulunamadı!")
            return None

        # Tüm çizgileri birleştir
        combined = np.vstack(all_lines)

        # Tekrar eden çizgileri kaldır (yakın çizgileri filtrele)
        unique = self._remove_duplicate_lines(combined)

        logger.info(
            f"Toplam çizgi: {len(combined)} → benzersiz: {len(unique)}"
        )
        return unique

    def _remove_duplicate_lines(
        self, lines: np.ndarray, threshold: float = 10.0
    ) -> np.ndarray:
        """Birbirine çok yakın çizgileri kaldır."""
        if len(lines) <= 1:
            return lines

        keep = []
        used = set()

        for i, line_i in enumerate(lines):
            if i in used:
                continue
            x1i, y1i, x2i, y2i = line_i[0]
            mid_i = ((x1i + x2i) / 2, (y1i + y2i) / 2)

            for j in range(i + 1, len(lines)):
                if j in used:
                    continue
                x1j, y1j, x2j, y2j = lines[j][0]
                mid_j = ((x1j + x2j) / 2, (y1j + y2j) / 2)

                dist = np.sqrt(
                    (mid_i[0] - mid_j[0]) ** 2 + (mid_i[1] - mid_j[1]) ** 2
                )
                if dist < threshold:
                    used.add(j)

            keep.append(lines[i])

        return np.array(keep)

    # ════════════════════════════════════════════════════════════
    # Adım 4: Akıllı Açı Filtreleme
    # ════════════════════════════════════════════════════════════

    def filter_diagonal_lines(
        self, lines: np.ndarray | None
    ) -> list[dict]:
        """Çapraz çizgileri filtrele — açı + uzunluk + pozisyon.

        Taralı alan çizgileri:
            - 20°-70° arasında çapraz
            - Belirli bir minimum uzunluk
            - Benzer açıda paralel gruplar halinde
        """
        diagonal_lines = []

        if lines is None:
            return diagonal_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Açı filtresi
            if not (self.min_angle <= angle <= self.max_angle):
                continue

            # Çok kısa çizgileri ele (gürültü)
            if length < self.min_line_length * 0.8:
                continue

            # Çizginin yönünü hesapla (pozitif/negatif eğim)
            slope_sign = 1 if (y2 - y1) * (x2 - x1) > 0 else -1

            diagonal_lines.append({
                "line": (x1, y1, x2, y2),
                "angle": angle,
                "slope_sign": slope_sign,
                "midpoint": ((x1 + x2) / 2, (y1 + y2) / 2),
                "length": length,
            })

        logger.info(
            f"Açı filtreleme ({self.min_angle}°-{self.max_angle}°): "
            f"{len(diagonal_lines)} çapraz çizgi"
        )
        return diagonal_lines

    # ════════════════════════════════════════════════════════════
    # Adım 5: DBSCAN Kümeleme
    # ════════════════════════════════════════════════════════════

    def cluster_lines(
        self, diagonal_lines: list[dict]
    ) -> dict[int, list[dict]]:
        """Yakın çapraz çizgileri kümele.

        Özellikler: midpoint (x,y) + açı (normalize edilmiş)
        Bu sayede hem konumca hem açıca benzer çizgiler kümelenir.
        """
        if len(diagonal_lines) < self.dbscan_min_samples:
            logger.warning(
                f"Yetersiz çapraz çizgi: {len(diagonal_lines)} "
                f"(min {self.dbscan_min_samples})"
            )
            return {}

        # Feature vektörü: [x, y, açı*ağırlık]
        # Açı farkını da kümelemeye dahil et
        features = []
        for dl in diagonal_lines:
            mx, my = dl["midpoint"]
            angle_feature = dl["angle"] * 2.0  # Açı farkına duyarlılık
            features.append([mx, my, angle_feature])

        features = np.array(features)

        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
        ).fit(features)

        clusters: dict[int, list[dict]] = {}
        noise_count = 0
        for i, label in enumerate(clustering.labels_):
            if label == -1:
                noise_count += 1
                continue
            clusters.setdefault(label, []).append(diagonal_lines[i])

        logger.info(
            f"DBSCAN: {len(clusters)} küme (gürültü: {noise_count})"
        )
        return clusters

    # ════════════════════════════════════════════════════════════
    # Adım 6: Polygon Çıkarma + Morfolojik Düzeltme
    # ════════════════════════════════════════════════════════════

    def cluster_to_polygon(
        self, cluster_lines: list[dict]
    ) -> list[list[int]] | None:
        """Kümedeki çizgilerden polygon çıkar ve düzelt.

        1. Tüm uç noktaları topla
        2. Convex Hull
        3. Buffer ile genişlet (çizgi uçlarının dışını da kapsa)
        4. Simplify ile düzelt
        """
        points = []
        for line_info in cluster_lines:
            x1, y1, x2, y2 = line_info["line"]
            points.append([x1, y1])
            points.append([x2, y2])

        if len(points) < 3:
            return None

        points_arr = np.array(points, dtype=np.float32)

        # Convex Hull
        hull = cv2.convexHull(points_arr)
        hull_pts = hull.reshape(-1, 2).tolist()

        if len(hull_pts) < 3:
            return None

        # Shapely polygon
        poly = Polygon(hull_pts)

        if not poly.is_valid:
            poly = poly.buffer(0)  # Fix invalid geometry

        if poly.area < self.min_polygon_area:
            logger.debug(f"Polygon çok küçük: {poly.area:.0f}")
            return None

        # Buffer ile genişlet (çizgi uçlarının hemen dışını da kapsa)
        if self.polygon_buffer > 0:
            poly = poly.buffer(self.polygon_buffer, join_style=2)

        # Simplify (çok fazla nokta varsa azalt)
        poly = poly.simplify(5.0, preserve_topology=True)

        # Koordinatları çıkar
        if poly.is_empty:
            return None

        coords = list(poly.exterior.coords[:-1])  # Son nokta = ilk nokta
        return [[int(x), int(y)] for x, y in coords]

    # ════════════════════════════════════════════════════════════
    # Ana Fonksiyon: detect()
    # ════════════════════════════════════════════════════════════

    def detect(self, video_path: str) -> dict:
        """Videodan otomatik taralı alan tespiti — tam pipeline.

        Returns:
            {
                "polygon": [[x1,y1], ...] veya None,
                "confidence": float (0-1),
                "all_polygons": [...],
                "debug": { ... }
            }
        """
        logger.info(f"Otomatik taralı alan tespiti: {video_path}")

        # 1. Background
        background = self.extract_background(video_path)

        # 2. Çok aşamalı ön işleme
        preproc = self.preprocess(background)

        # 3. Çok kaynaklı çizgi tespiti
        lines = self.detect_lines(preproc)

        # 4. Çapraz çizgi filtreleme
        diagonal_lines = self.filter_diagonal_lines(lines)

        # 5. Kümeleme
        clusters = self.cluster_lines(diagonal_lines)

        # 6. Her küme için polygon
        all_polygons = []
        for cid in sorted(clusters.keys()):
            cluster = clusters[cid]
            polygon = self.cluster_to_polygon(cluster)
            if polygon is not None:
                area = Polygon(polygon).area
                avg_angle = np.mean([l["angle"] for l in cluster])
                all_polygons.append({
                    "cluster_id": cid,
                    "polygon": polygon,
                    "line_count": len(cluster),
                    "area": area,
                    "avg_angle": avg_angle,
                })

        # 7. En iyi polygon seç
        best_polygon = None
        confidence = 0.0

        if all_polygons:
            all_polygons.sort(key=lambda p: p["line_count"], reverse=True)
            best = all_polygons[0]
            best_polygon = best["polygon"]
            confidence = min(best["line_count"] / 15.0, 1.0)
            logger.info(
                f"En iyi polygon: {best['line_count']} çizgi, "
                f"alan={best['area']:.0f}, açı={best['avg_angle']:.1f}°, "
                f"güven={confidence:.2f}"
            )
        else:
            logger.warning("Taralı alan tespit edilemedi!")

        return {
            "polygon": best_polygon,
            "confidence": confidence,
            "all_polygons": all_polygons,
            "debug": {
                "background": background,
                "enhanced": preproc["enhanced"],
                "edges_canny": preproc["edges_canny"],
                "edges_adaptive": preproc["edges_adaptive"],
                "edges_combined": preproc["edges_combined"],
                "all_lines_count": len(lines) if lines is not None else 0,
                "diagonal_lines": diagonal_lines,
                "clusters": clusters,
            },
        }

    # ════════════════════════════════════════════════════════════
    # Yardımcı Fonksiyonlar
    # ════════════════════════════════════════════════════════════

    def save_zone_json(
        self,
        polygon: list[list[int]],
        output_path: str,
        camera_id: str = "auto_detected",
        frame_width: int = 1080,
        frame_height: int = 1920,
    ) -> None:
        """Tespit edilen polygon'u zone JSON formatına kaydet."""
        import json

        data = {
            "camera_id": camera_id,
            "description": "Otomatik tespit — Klasik CV (Hough+DBSCAN)",
            "detection_method": "hough_lines_dbscan_v2",
            "frame_width": frame_width,
            "frame_height": frame_height,
            "zones": [
                {
                    "zone_id": "zone_auto_01",
                    "name": "Taralı Alan (Otomatik)",
                    "polygon": polygon,
                    "type": "hatched_area",
                }
            ],
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Zone kaydedildi: {output_path}")

    def save_debug_images(
        self, result: dict, output_dir: str
    ) -> dict[str, str]:
        """Her adımın görsellerini kaydet (tez için)."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = {}
        debug = result["debug"]

        # 1. Arka plan (stabilize edilmiş)
        path = str(out / "01_background.jpg")
        cv2.imwrite(path, debug["background"])
        saved["background"] = path

        # 2. Enhanced (CLAHE)
        path = str(out / "02_enhanced.jpg")
        cv2.imwrite(path, debug["enhanced"])
        saved["enhanced"] = path

        # 3. Canny edges
        path = str(out / "03_edges_canny.jpg")
        cv2.imwrite(path, debug["edges_canny"])
        saved["edges_canny"] = path

        # 4. Adaptive edges
        path = str(out / "04_edges_adaptive.jpg")
        cv2.imwrite(path, debug["edges_adaptive"])
        saved["edges_adaptive"] = path

        # 5. Combined edges
        path = str(out / "05_edges_combined.jpg")
        cv2.imwrite(path, debug["edges_combined"])
        saved["edges_combined"] = path

        # 6. Tüm çapraz çizgiler (yeşil)
        bg_copy = debug["background"].copy()
        for dl in debug["diagonal_lines"]:
            x1, y1, x2, y2 = dl["line"]
            cv2.line(bg_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Açıyı yaz
            mx, my = int(dl["midpoint"][0]), int(dl["midpoint"][1])
            cv2.putText(
                bg_copy, f"{dl['angle']:.0f}",
                (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
            )
        path = str(out / "06_diagonal_lines.jpg")
        cv2.imwrite(path, bg_copy)
        saved["diagonal_lines"] = path

        # 7. Kümeler (farklı renkler)
        bg_copy = debug["background"].copy()
        colors = [
            (0, 0, 255), (255, 0, 0), (0, 255, 255),
            (255, 0, 255), (128, 255, 0), (0, 128, 255),
        ]
        for cid, cluster in debug["clusters"].items():
            color = colors[cid % len(colors)]
            for li in cluster:
                x1, y1, x2, y2 = li["line"]
                cv2.line(bg_copy, (x1, y1), (x2, y2), color, 2)
        path = str(out / "07_clusters.jpg")
        cv2.imwrite(path, bg_copy)
        saved["clusters"] = path

        # 8. Final polygon (kırmızı overlay)
        bg_copy = debug["background"].copy()
        if result["polygon"] is not None:
            pts = np.array(result["polygon"], dtype=np.int32)
            overlay = bg_copy.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            bg_copy = cv2.addWeighted(overlay, 0.4, bg_copy, 0.6, 0)
            cv2.polylines(bg_copy, [pts], True, (0, 0, 255), 3)
            cv2.putText(
                bg_copy,
                f"TARALI ALAN (guven: {result['confidence']:.2f})",
                (pts[0][0], max(pts[0][1] - 15, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
        # Diğer polygon'lar (turuncu)
        for poly_info in result["all_polygons"][1:]:
            pts = np.array(poly_info["polygon"], dtype=np.int32)
            cv2.polylines(bg_copy, [pts], True, (0, 165, 255), 2)

        path = str(out / "08_final_polygon.jpg")
        cv2.imwrite(path, bg_copy)
        saved["final_polygon"] = path

        logger.info(f"Debug görselleri kaydedildi: {out}")
        return saved


# ════════════════════════════════════════════════════════════
# IoU Hesaplama (Manuel ile Karşılaştırma)
# ════════════════════════════════════════════════════════════

def calculate_iou(polygon_auto: list, polygon_manual: list) -> float:
    """İki polygon arasındaki IoU değeri (0.0 - 1.0)."""
    poly_a = Polygon(polygon_auto)
    poly_m = Polygon(polygon_manual)

    if not poly_a.is_valid or not poly_m.is_valid:
        return 0.0

    intersection = poly_a.intersection(poly_m).area
    union = poly_a.union(poly_m).area

    return intersection / union if union > 0 else 0.0
