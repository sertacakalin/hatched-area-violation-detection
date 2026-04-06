"""İhlal anından kısa video klip çıkarma.

İhlal tespit edildiğinde, ihlalden 2 saniye önce ve 2 saniye sonrasını
kapsayan ~4 saniyelik bir video klip oluşturur. Klip üzerinde:
- Araç bbox'ı kırmızı çizilir
- Şiddet skoru ve ihlal tipi yazılır
- Taralı alan overlay gösterilir

Bu klipler jüri sunumunda ve dashboard'da kullanılır.
"""

import logging
from pathlib import Path
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ViolationClipExtractor:
    """İhlal anı video klipleri oluşturur.

    Pipeline çalışırken son N kareyi buffer'da tutar.
    İhlal tetiklendiğinde buffer'daki + sonraki kareleri
    birleştirip kısa bir video klip oluşturur.
    """

    def __init__(self, buffer_seconds: float = 2.0,
                 after_seconds: float = 2.0,
                 output_dir: str = "results/clips",
                 fps: float = 30.0):
        self.buffer_size = int(buffer_seconds * fps)
        self.after_frames = int(after_seconds * fps)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps

        # Son N kareyi tutan ring buffer
        self._buffer: deque[np.ndarray] = deque(maxlen=self.buffer_size)

        # Aktif klip kayıtları: event_id → {"frames": [...], "remaining": int, "event": ...}
        self._active_clips: dict[str, dict] = {}

    def feed_frame(self, frame: np.ndarray) -> None:
        """Her karede çağrılır — frame'i buffer'a ekler."""
        self._buffer.append(frame.copy())

        # Aktif kliplere frame ekle
        completed = []
        for eid, clip in self._active_clips.items():
            clip["frames"].append(frame.copy())
            clip["remaining"] -= 1
            if clip["remaining"] <= 0:
                completed.append(eid)

        # Tamamlanan klipleri kaydet
        for eid in completed:
            self._save_clip(eid)

    def on_violation(self, event_id: str, track_id: int,
                     severity_score: float, violation_type: str,
                     vehicle_bbox: np.ndarray,
                     zone_polygons: list[tuple[str, np.ndarray]]) -> None:
        """İhlal tetiklendiğinde çağrılır — klip kaydını başlatır."""
        # Buffer'daki önceki kareleri al
        before_frames = list(self._buffer)

        self._active_clips[event_id] = {
            "frames": before_frames,
            "remaining": self.after_frames,
            "track_id": track_id,
            "severity_score": severity_score,
            "violation_type": violation_type,
            "vehicle_bbox": vehicle_bbox.copy(),
            "zone_polygons": zone_polygons,
        }

        logger.info(f"Klip kaydı başladı: {event_id} ({len(before_frames)} önceki kare)")

    def _save_clip(self, event_id: str) -> str | None:
        """Klip kaydet ve aktif listeden çıkar."""
        clip = self._active_clips.pop(event_id, None)
        if clip is None or not clip["frames"]:
            return None

        frames = clip["frames"]
        h, w = frames[0].shape[:2]
        output_path = str(self.output_dir / f"violation_{event_id}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        # İhlal anı = buffer_size'ıncı kare
        violation_frame_idx = min(self.buffer_size, len(frames) - 1)

        for i, frame in enumerate(frames):
            display = frame.copy()

            # Taralı alan overlay
            for name, polygon in clip["zone_polygons"]:
                overlay = display.copy()
                pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(overlay, [pts], (255, 165, 0))
                display = cv2.addWeighted(overlay, 0.2, display, 0.8, 0)
                cv2.polylines(display, [pts], True, (255, 165, 0), 2)

            # İhlal anı civarında bbox çiz
            dist = abs(i - violation_frame_idx)
            if dist < 30:
                x1, y1, x2, y2 = clip["vehicle_bbox"].astype(int)
                color = (0, 0, 255) if dist < 5 else (0, 165, 255)
                thickness = 3 if dist < 5 else 2
                cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

            # Bilgi yazısı
            info = f"IHLAL: {clip['violation_type']} | Skor: {clip['severity_score']:.0f}"
            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Zaman göstergesi
            sec = i / self.fps
            time_label = f"{sec:.1f}s"
            if i == violation_frame_idx:
                time_label += " << IHLAL ANI"
            cv2.putText(display, time_label, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            writer.write(display)

        writer.release()
        logger.info(f"Klip kaydedildi: {output_path} ({len(frames)} kare)")
        return output_path

    def flush(self) -> None:
        """Kalan aktif klipleri zorla kaydet."""
        for eid in list(self._active_clips.keys()):
            self._save_clip(eid)
