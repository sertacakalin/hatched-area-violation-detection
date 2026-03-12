"""Video okuma ve kare iterasyonu."""

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class FrameProvider:
    """Video dosyasından veya kameradan kare okuyucu."""

    def __init__(self, source: str | int | Path):
        """
        Args:
            source: Video dosya yolu, URL veya kamera indeksi (0, 1, ...).
        """
        self._source = str(source)
        self._cap: cv2.VideoCapture | None = None
        self._frame_count = 0

    def open(self) -> "FrameProvider":
        src = int(self._source) if self._source.isdigit() else self._source
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise IOError(f"Video kaynağı açılamadı: {self._source}")
        self._frame_count = 0
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameProvider":
        return self.open()

    def __exit__(self, *args) -> None:
        self.close()

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Kare numarası ve kare döndüren iterator."""
        if self._cap is None:
            self.open()
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            self._frame_count += 1
            yield self._frame_count, frame

    @property
    def fps(self) -> float:
        if self._cap is None:
            self.open()
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self) -> int:
        if self._cap is None:
            self.open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        if self._cap is None:
            self.open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        if self._cap is None:
            self.open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    def get_frame_at(self, frame_number: int) -> np.ndarray | None:
        """Belirli bir kareyi oku."""
        if self._cap is None:
            self.open()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._cap.read()
        return frame if ret else None

    def get_timestamp(self, frame_number: int) -> float:
        """Kare numarasından saniye cinsinden zaman damgası."""
        fps = self.fps
        return frame_number / fps if fps > 0 else 0.0
