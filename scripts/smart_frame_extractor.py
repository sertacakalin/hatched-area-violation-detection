"""Akilli frame cikarma — etiketleme icin optimize edilmis.

Basit "her N karede bir" yerine:
  1. Sahne degisiklik tespiti (benzer frame'leri atla)
  2. Arac yogunluk filtresi (bos kareleri atla)
  3. Kalite filtresi (bulanik kareleri atla)
  4. Stratified sampling (videonun her bolumunden esit oranda)

Kullanim:
    # Tek video
    python scripts/smart_frame_extractor.py --video VIDEO.mp4 --output KLASOR

    # Toplu islem (Google Drive'dan)
    python scripts/smart_frame_extractor.py \
        --batch /path/to/drive/istanbul_trafik_kayit/ \
        --output data/frames_for_labeling/

    # Roboflow'a otomatik yukle
    python scripts/smart_frame_extractor.py --video VIDEO.mp4 \
        --upload-roboflow --rf-workspace WORKSPACE --rf-project PROJECT

Sertac Akalin — Bitirme Tezi 2026
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".MOV"}


@dataclass
class FrameMetadata:
    """Cikarilan frame hakkinda bilgi."""
    video: str
    frame_number: int
    timestamp_sec: float
    vehicle_count: int
    blur_score: float
    scene_change_score: float
    segment: int  # videonun hangi bolumu (stratified sampling)


class SmartFrameExtractor:
    """Akilli frame cikarma motoru."""

    def __init__(
        self,
        target_frames: int = 60,
        min_scene_change: float = 25.0,
        min_blur_score: float = 50.0,
        use_vehicle_filter: bool = True,
        vehicle_model_path: str = "yolov8s.pt",
    ):
        self.target_frames = target_frames
        self.min_scene_change = min_scene_change
        self.min_blur_score = min_blur_score
        self.use_vehicle_filter = use_vehicle_filter
        self.detector = None

        if use_vehicle_filter:
            try:
                from src.detection.vehicle_detector import VehicleDetector
                self.detector = VehicleDetector(
                    model_path=vehicle_model_path,
                    confidence_threshold=0.3,
                )
                logger.info("YOLOv8 modeli yuklendi (arac filtresi aktif)")
            except Exception as e:
                logger.warning(f"YOLOv8 yuklenemedi, arac filtresi devre disi: {e}")
                self.use_vehicle_filter = False

    def _blur_score(self, frame: np.ndarray) -> float:
        """Laplacian varyans ile netlik skoru. Dusuk = bulanik."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _scene_change_score(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Iki frame arasi fark skoru (histogram karsilastirma)."""
        h1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        h2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        # 1 - correlation = fark skoru (0=ayni, 2=tamamen farkli)
        score = 1.0 - cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        return score * 100  # 0-200 arasi, 25+ = anlamli degisiklik

    def _count_vehicles(self, frame: np.ndarray) -> int:
        """Frame'deki arac sayisini dondur."""
        if not self.detector:
            return -1  # filtre devre disi
        detections = self.detector.detect(frame)
        return len(detections)

    def _analyze_video(self, video_path: str) -> dict:
        """Video meta bilgilerini cikar."""
        cap = cv2.VideoCapture(video_path)
        info = {
            "path": video_path,
            "name": Path(video_path).stem,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        info["duration_sec"] = info["total_frames"] / max(info["fps"], 1)
        cap.release()
        return info

    def extract(self, video_path: str, output_dir: str) -> list[FrameMetadata]:
        """Tek bir videodan akilli frame cikar."""
        video_info = self._analyze_video(video_path)
        video_name = video_info["name"]
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"Video: {video_name}")
        logger.info(f"Sure: {video_info['duration_sec']:.0f}s | "
                     f"FPS: {video_info['fps']:.0f} | "
                     f"Cozunurluk: {video_info['width']}x{video_info['height']}")
        logger.info(f"Hedef: {self.target_frames} frame")
        logger.info(f"{'='*60}")

        total = video_info["total_frames"]
        fps = video_info["fps"]

        # Strateji: videoyu N segmente bol, her segmentten esit frame sec
        num_segments = min(self.target_frames, 10)
        segment_size = total // num_segments
        frames_per_segment = max(self.target_frames // num_segments, 1)

        # Aday frame'leri topla (her segmentten)
        # Her segment icinde daha sik ornekle, sonra en iyileri sec
        sample_interval = max(segment_size // (frames_per_segment * 5), 1)

        cap = cv2.VideoCapture(video_path)
        prev_frame = None
        candidates: list[tuple[int, np.ndarray, float, float, int, int]] = []
        # (frame_no, frame, blur, scene_change, vehicle_count, segment)

        logger.info("Adim 1/3: Aday frame'leri taraniyor...")
        frame_num = 0

        for seg in range(num_segments):
            seg_start = seg * segment_size
            seg_end = min((seg + 1) * segment_size, total)
            cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
            frame_num = seg_start

            seg_candidates = []
            while frame_num < seg_end:
                ret, frame = cap.read()
                if not ret:
                    break

                if (frame_num - seg_start) % sample_interval == 0:
                    blur = self._blur_score(frame)
                    scene = self._scene_change_score(prev_frame, frame) if prev_frame is not None else 100.0
                    vehicles = self._count_vehicles(frame) if self.use_vehicle_filter else -1

                    seg_candidates.append((frame_num, frame.copy(), blur, scene, vehicles, seg))

                prev_frame = frame.copy()
                frame_num += 1

            candidates.extend(seg_candidates)

        cap.release()
        logger.info(f"  {len(candidates)} aday frame bulundu")

        # Filtreleme
        logger.info("Adim 2/3: Filtreleme ve siralama...")
        filtered = []
        for c in candidates:
            frame_no, frame, blur, scene, vehicles, seg = c
            # Bulanik frame'leri at
            if blur < self.min_blur_score:
                continue
            # Cok benzer frame'leri at (ilk frame haric)
            if scene < self.min_scene_change and len(filtered) > 0:
                # Ama her segmentten en az 1 tane al
                seg_count = sum(1 for f in filtered if f[5] == seg)
                if seg_count > 0:
                    continue
            # Arac filtresi: bos karelere dusuk oncelik (atma, geride birak)
            filtered.append(c)

        # Siralama: arac sayisi (cok > az) → sahne degisikligi (yuksek > dusuk)
        if self.use_vehicle_filter:
            filtered.sort(key=lambda x: (x[4], x[3]), reverse=True)
        else:
            filtered.sort(key=lambda x: x[3], reverse=True)

        # Her segmentten esit almaya calis
        selected = []
        segment_counts = {s: 0 for s in range(num_segments)}

        for c in filtered:
            seg = c[5]
            if segment_counts[seg] < frames_per_segment:
                selected.append(c)
                segment_counts[seg] += 1
            if len(selected) >= self.target_frames:
                break

        # Hedef sayiya ulasilamazsa kalanlardan da ekle
        if len(selected) < self.target_frames:
            remaining = [c for c in filtered if c not in selected]
            for c in remaining:
                selected.append(c)
                if len(selected) >= self.target_frames:
                    break

        # Frame numarasina gore sirala (kronolojik)
        selected.sort(key=lambda x: x[0])

        # Kaydet
        logger.info(f"Adim 3/3: {len(selected)} frame kaydediliyor...")
        metadata_list = []

        for c in tqdm(selected, desc="Kaydediliyor"):
            frame_no, frame, blur, scene, vehicles, seg = c
            timestamp = frame_no / max(fps, 1)

            # Dosya adi: video_frame_NNNNNN_tSSS.jpg
            filename = f"{video_name}_frame_{frame_no:06d}_t{timestamp:.0f}s.jpg"
            cv2.imwrite(str(out / filename), frame)

            meta = FrameMetadata(
                video=video_name,
                frame_number=frame_no,
                timestamp_sec=round(timestamp, 1),
                vehicle_count=vehicles,
                blur_score=round(blur, 1),
                scene_change_score=round(scene, 1),
                segment=seg,
            )
            metadata_list.append(meta)

        # Metadata JSON kaydet
        meta_file = out / f"{video_name}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "video": video_info,
                "extraction_config": {
                    "target_frames": self.target_frames,
                    "min_blur_score": self.min_blur_score,
                    "min_scene_change": self.min_scene_change,
                    "vehicle_filter": self.use_vehicle_filter,
                },
                "frames": [asdict(m) for m in metadata_list],
                "summary": {
                    "total_extracted": len(metadata_list),
                    "avg_vehicles": round(np.mean([m.vehicle_count for m in metadata_list if m.vehicle_count >= 0]), 1) if any(m.vehicle_count >= 0 for m in metadata_list) else -1,
                    "avg_blur_score": round(np.mean([m.blur_score for m in metadata_list]), 1),
                },
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"\n  {len(metadata_list)} frame kaydedildi: {out}")
        logger.info(f"  Metadata: {meta_file}")

        return metadata_list

    def extract_batch(self, video_dir: str, output_dir: str) -> dict[str, list[FrameMetadata]]:
        """Bir klasordeki tum videolardan frame cikar."""
        video_dir = Path(video_dir)
        results = {}

        videos = sorted([
            f for f in video_dir.iterdir()
            if f.suffix.lower() in {e.lower() for e in VIDEO_EXTENSIONS} and f.is_file()
        ])

        if not videos:
            logger.error(f"Video bulunamadi: {video_dir}")
            return results

        logger.info(f"\n{len(videos)} video bulundu:")
        for v in videos:
            info = self._analyze_video(str(v))
            logger.info(f"  {v.name} — {info['duration_sec']:.0f}s, "
                         f"{info['width']}x{info['height']}")

        for video in videos:
            name = video.stem
            out = Path(output_dir) / name
            metadata = self.extract(str(video), str(out))
            results[name] = metadata

        # Toplam ozet
        total_frames = sum(len(v) for v in results.values())
        logger.info(f"\n{'='*60}")
        logger.info(f"TOPLAM: {len(videos)} video, {total_frames} frame cikarildi")
        logger.info(f"Cikti: {output_dir}")
        logger.info(f"{'='*60}")

        return results

    def upload_to_roboflow(self, image_dir: str, workspace: str, project: str,
                           api_key: str | None = None) -> None:
        """Cikarilan frame'leri Roboflow'a yukle."""
        try:
            from roboflow import Roboflow
        except ImportError:
            logger.error("roboflow paketi yuklu degil: pip install roboflow")
            return

        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)

        images = sorted(Path(image_dir).glob("*.jpg"))
        logger.info(f"\n{len(images)} frame Roboflow'a yukleniyor...")

        for img in tqdm(images, desc="Roboflow upload"):
            try:
                proj.upload(str(img))
            except Exception as e:
                logger.warning(f"  Yuklenemedi: {img.name} — {e}")

        logger.info(f"Yukleme tamamlandi: {project}")


def main():
    parser = argparse.ArgumentParser(
        description="Akilli frame cikarma — etiketleme icin optimize edilmis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  # Tek video, 60 frame
  python scripts/smart_frame_extractor.py --video cam1.mov --output frames/cam1

  # 100 frame, arac filtresi ile
  python scripts/smart_frame_extractor.py --video cam1.mov -n 100 --output frames/cam1

  # Toplu islem (klasordeki tum videolar)
  python scripts/smart_frame_extractor.py --batch /path/to/videos/ --output frames/

  # Roboflow'a yukle
  python scripts/smart_frame_extractor.py --video cam1.mov --output frames/cam1 \\
      --upload-roboflow --rf-workspace WORKSPACE --rf-project PROJECT
        """,
    )

    # Girdi
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Tek video dosyasi")
    group.add_argument("--batch", help="Video klasoru (toplu islem)")

    # Cikti
    parser.add_argument("--output", "-o", required=True, help="Cikti klasoru")

    # Parametreler
    parser.add_argument("-n", "--target-frames", type=int, default=60,
                        help="Hedef frame sayisi (varsayilan: 60)")
    parser.add_argument("--min-blur", type=float, default=50.0,
                        help="Minimum netlik skoru (varsayilan: 50)")
    parser.add_argument("--min-change", type=float, default=25.0,
                        help="Minimum sahne degisiklik skoru (varsayilan: 25)")
    parser.add_argument("--no-vehicle-filter", action="store_true",
                        help="Arac filtresi olmadan cikar (hizli)")

    # Roboflow
    parser.add_argument("--upload-roboflow", action="store_true",
                        help="Roboflow'a otomatik yukle")
    parser.add_argument("--rf-workspace", help="Roboflow workspace")
    parser.add_argument("--rf-project", help="Roboflow project")
    parser.add_argument("--rf-api-key", help="Roboflow API key (veya ROBOFLOW_API_KEY env)")

    args = parser.parse_args()

    extractor = SmartFrameExtractor(
        target_frames=args.target_frames,
        min_blur_score=args.min_blur,
        min_scene_change=args.min_change,
        use_vehicle_filter=not args.no_vehicle_filter,
    )

    if args.video:
        extractor.extract(args.video, args.output)
        upload_dir = args.output
    else:
        extractor.extract_batch(args.batch, args.output)
        upload_dir = args.output

    if args.upload_roboflow:
        if not args.rf_workspace or not args.rf_project:
            logger.error("--rf-workspace ve --rf-project gerekli")
            return
        import os
        api_key = args.rf_api_key or os.environ.get("ROBOFLOW_API_KEY")
        extractor.upload_to_roboflow(upload_dir, args.rf_workspace,
                                      args.rf_project, api_key)


if __name__ == "__main__":
    main()
