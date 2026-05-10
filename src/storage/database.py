"""SQLite veritabanı — ihlal kayıtları için şema ve CRUD işlemleri."""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    track_id INTEGER NOT NULL,
    frame_number INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    vehicle_class TEXT NOT NULL,
    vehicle_confidence REAL NOT NULL,
    vehicle_bbox TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    frames_in_zone INTEGER NOT NULL,
    plate_text TEXT,
    plate_raw TEXT,
    plate_confidence REAL,
    plate_valid INTEGER DEFAULT 0,
    city_code TEXT,
    city_name TEXT,
    severity_score REAL DEFAULT 0,
    severity_level TEXT,
    violation_type TEXT,
    trajectory_metrics TEXT,
    vehicle_crop_path TEXT,
    plate_crop_path TEXT,
    frame_image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    video_source TEXT
);
"""

INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_violations_plate ON violations(plate_text);
CREATE INDEX IF NOT EXISTS idx_violations_zone ON violations(zone_id);
CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp_sec);
CREATE INDEX IF NOT EXISTS idx_violations_created ON violations(created_at);
"""

# Eski DB'lere eklenmesi gerekebilen kolonlar. (kolon_adı, ALTER ifadesi)
# Sıra önemli — bir kolonu referans eden index'ten önce kolonu eklemeliyiz.
_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("plate_text", "ALTER TABLE violations ADD COLUMN plate_text TEXT"),
    ("plate_raw", "ALTER TABLE violations ADD COLUMN plate_raw TEXT"),
    ("plate_confidence", "ALTER TABLE violations ADD COLUMN plate_confidence REAL"),
    ("plate_valid", "ALTER TABLE violations ADD COLUMN plate_valid INTEGER DEFAULT 0"),
    ("city_code", "ALTER TABLE violations ADD COLUMN city_code TEXT"),
    ("city_name", "ALTER TABLE violations ADD COLUMN city_name TEXT"),
    ("severity_score", "ALTER TABLE violations ADD COLUMN severity_score REAL DEFAULT 0"),
    ("severity_level", "ALTER TABLE violations ADD COLUMN severity_level TEXT"),
    ("violation_type", "ALTER TABLE violations ADD COLUMN violation_type TEXT"),
    ("trajectory_metrics", "ALTER TABLE violations ADD COLUMN trajectory_metrics TEXT"),
    ("plate_crop_path", "ALTER TABLE violations ADD COLUMN plate_crop_path TEXT"),
    ("created_at", "ALTER TABLE violations ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
    ("video_source", "ALTER TABLE violations ADD COLUMN video_source TEXT"),
)


class ViolationDatabase:
    """SQLite veritabanı yöneticisi."""

    def __init__(self, db_path: str | Path = "results/violations.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        conn = self._get_connection()
        # Sıra: önce tablo (yoksa oluştur), sonra eksik kolonları ekle,
        # en son index'leri oluştur — index'ler kolonlara bağlı, kolon
        # olmadan index açmak hata verir.
        conn.executescript(TABLE_SCHEMA)
        self._migrate(conn)
        conn.executescript(INDEX_SCHEMA)
        conn.commit()
        logger.info(f"Veritabanı hazır: {self.db_path}")

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Eski DB'lere eksik kolonları ekle (idempotent)."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(violations)")}
        for col, alter_sql in _MIGRATION_COLUMNS:
            if col not in existing:
                conn.execute(alter_sql)
                logger.info(f"DB migration: '{col}' kolonu eklendi")

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False: Gradio handler thread-pool'unda
            # çalıştığında Pipeline'ın DB'ye yazması için gerekli. Tek
            # connection olduğu için yine de paralel write yapılmamalı;
            # _write_lock buna karşı koruma.
            self._conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False
            )
            self._conn.row_factory = sqlite3.Row
            # WAL mode: okuyucular yazıcıyı bloklamasın (DB browser, dashboard)
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.DatabaseError as exc:
                logger.warning("PRAGMA WAL ayarlanamadı: %s", exc)
        return self._conn

    def insert_violation(self, data: dict) -> int:
        """Yeni ihlal kaydı ekle."""
        conn = self._get_connection()
        cursor = conn.execute(
            """INSERT OR IGNORE INTO violations
            (event_id, track_id, frame_number, timestamp_sec,
             vehicle_class, vehicle_confidence, vehicle_bbox,
             zone_id, frames_in_zone,
             plate_text, plate_raw, plate_confidence, plate_valid,
             city_code, city_name,
             severity_score, severity_level, violation_type, trajectory_metrics,
             vehicle_crop_path, plate_crop_path, frame_image_path, video_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data.get("event_id"),
                data.get("track_id"),
                data.get("frame_number"),
                data.get("timestamp_sec"),
                data.get("vehicle_class"),
                data.get("vehicle_confidence"),
                data.get("vehicle_bbox"),
                data.get("zone_id"),
                data.get("frames_in_zone"),
                data.get("plate_text"),
                data.get("plate_raw"),
                data.get("plate_confidence"),
                data.get("plate_valid", 0),
                data.get("city_code"),
                data.get("city_name"),
                data.get("severity_score", 0),
                data.get("severity_level"),
                data.get("violation_type"),
                data.get("trajectory_metrics"),
                data.get("vehicle_crop_path"),
                data.get("plate_crop_path"),
                data.get("frame_image_path"),
                data.get("video_source"),
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def get_all_violations(self, limit: int = 100,
                           offset: int = 0) -> list[dict]:
        """Tüm ihlalleri getir."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM violations ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_violation_count(self) -> int:
        """Toplam ihlal sayısı."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM violations")
        return cursor.fetchone()["cnt"]

    def get_violations_by_plate(self, plate_text: str) -> list[dict]:
        """Plakaya göre ihlal ara."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM violations WHERE plate_text LIKE ? ORDER BY created_at DESC",
            (f"%{plate_text}%",),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_violations_by_zone(self, zone_id: str) -> list[dict]:
        """Bölgeye göre ihlalleri getir."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM violations WHERE zone_id = ? ORDER BY created_at DESC",
            (zone_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self) -> dict:
        """Genel istatistikler."""
        conn = self._get_connection()

        total = conn.execute("SELECT COUNT(*) as cnt FROM violations").fetchone()["cnt"]
        with_plate = conn.execute(
            "SELECT COUNT(*) as cnt FROM violations WHERE plate_text IS NOT NULL AND plate_text != ''"
        ).fetchone()["cnt"]
        valid_plate = conn.execute(
            "SELECT COUNT(*) as cnt FROM violations WHERE plate_valid = 1"
        ).fetchone()["cnt"]

        # Araç sınıfı dağılımı
        class_dist = conn.execute(
            "SELECT vehicle_class, COUNT(*) as cnt FROM violations GROUP BY vehicle_class ORDER BY cnt DESC"
        ).fetchall()

        # Bölge dağılımı
        zone_dist = conn.execute(
            "SELECT zone_id, COUNT(*) as cnt FROM violations GROUP BY zone_id ORDER BY cnt DESC"
        ).fetchall()

        return {
            "total_violations": total,
            "with_plate": with_plate,
            "valid_plate": valid_plate,
            "plate_detection_rate": with_plate / total if total > 0 else 0,
            "plate_validation_rate": valid_plate / with_plate if with_plate > 0 else 0,
            "class_distribution": {row["vehicle_class"]: row["cnt"] for row in class_dist},
            "zone_distribution": {row["zone_id"]: row["cnt"] for row in zone_dist},
        }

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
