"""YAML konfigürasyon yükleyici."""

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Hiyerarşik YAML konfigürasyon yöneticisi."""

    def __init__(self, config_path: str | Path):
        self._path = Path(config_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Config dosyası bulunamadı: {self._path}")
        with open(self._path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Nokta notasyonu ile değer al: 'vehicle_detection.confidence_threshold'."""
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        result = self.get(key)
        if result is None:
            raise KeyError(f"Config anahtarı bulunamadı: {key}")
        return result

    @property
    def data(self) -> dict:
        return self._data

    @staticmethod
    def load_yaml(path: str | Path) -> dict:
        """Yardımcı: herhangi bir YAML dosyasını dict olarak yükle."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
