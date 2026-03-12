"""Config'den pipeline oluşturan factory."""

import logging
from pathlib import Path

from src.core.config import Config
from src.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


def create_pipeline(config_path: str | Path = "configs/config.yaml",
                    overrides: dict | None = None) -> Pipeline:
    """Config dosyasından pipeline oluştur.

    Args:
        config_path: YAML config dosya yolu.
        overrides: Config üzerinde geçersiz kılınacak değerler.
                   Örn: {"general.video_source": "test.mp4"}
    """
    config = Config(config_path)

    # Override'ları uygula
    if overrides:
        for key, value in overrides.items():
            keys = key.split(".")
            d = config._data
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
            logger.info(f"Config override: {key} = {value}")

    # Logging ayarla
    log_level = config.get("general.log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    return Pipeline(config)
