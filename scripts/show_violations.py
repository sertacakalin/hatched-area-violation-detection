"""İhlal özet raporu — tek komutla tüm bilgi.

Kullanım:
    python scripts/show_violations.py results/cam4_30s_v3
    python scripts/show_violations.py results/cam4_30s_finetuned   # eski model
"""
import sqlite3
import sys
from pathlib import Path


def show_violations(result_dir: Path):
    db_path = result_dir / "violations.db"
    if not db_path.exists():
        print(f"❌ DB bulunamadı: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Total counts
    total = conn.execute("SELECT COUNT(*) as n FROM violations").fetchone()["n"]
    print(f"\n{'=' * 70}")
    print(f"  📊 İHLAL ÖZETİ — {result_dir.name}")
    print(f"{'=' * 70}")
    print(f"\n🔴 Toplam ihlal: {total}")

    if total == 0:
        return

    # By class
    print(f"\n📋 SINIF DAĞILIMI:")
    by_class = conn.execute(
        "SELECT vehicle_class, COUNT(*) as n FROM violations GROUP BY vehicle_class ORDER BY n DESC"
    ).fetchall()
    for row in by_class:
        print(f"  {row['vehicle_class']:>12s}: {row['n']:3d} ihlal")

    # By type
    print(f"\n🚦 İHLAL TİPİ:")
    by_type = conn.execute(
        "SELECT violation_type, COUNT(*) as n FROM violations GROUP BY violation_type ORDER BY n DESC"
    ).fetchall()
    for row in by_type:
        if row['violation_type']:
            print(f"  {row['violation_type']:>15s}: {row['n']:3d}")

    # By severity
    print(f"\n⚠️  ŞIDDET:")
    by_sev = conn.execute(
        "SELECT severity_level, COUNT(*) as n FROM violations GROUP BY severity_level ORDER BY n DESC"
    ).fetchall()
    for row in by_sev:
        if row['severity_level']:
            print(f"  {row['severity_level']:>10s}: {row['n']:3d}")

    # Detailed list
    print(f"\n📝 TÜM İHLALLER:")
    print(f"{'-' * 70}")
    print(f"{'#':>3s}  {'Track':>6s}  {'Sınıf':>8s}  {'Frame':>6s}  {'Saniye':>7s}  "
          f"{'Tip':>14s}  {'Skor':>5s}  {'Seviye':>8s}")
    print(f"{'-' * 70}")
    rows = conn.execute(
        """SELECT track_id, vehicle_class, frame_number, timestamp_sec,
                  violation_type, severity_score, severity_level
           FROM violations ORDER BY frame_number"""
    ).fetchall()
    for i, r in enumerate(rows, 1):
        score = r['severity_score'] or 0
        print(f"{i:>3d}  #{r['track_id']:>5d}  {r['vehicle_class']:>8s}  "
              f"{r['frame_number']:>6d}  {r['timestamp_sec']:>6.2f}s  "
              f"{r['violation_type'] or '?':>14s}  {score:>5.1f}  "
              f"{r['severity_level'] or '?':>8s}")

    print(f"\n{'=' * 70}")
    print(f"  📁 Video:    {result_dir / 'output.mp4'}")
    print(f"  📁 Frames:   {result_dir / 'frames'} ({len(list((result_dir / 'frames').glob('*.jpg'))) if (result_dir / 'frames').exists() else 0} dosya)")
    print(f"  📁 Crops:    {result_dir / 'crops'}")
    print(f"  📁 DB:       {db_path}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: en son v3 sonuç
        result = Path("results/cam4_30s_v3")
    else:
        result = Path(sys.argv[1])
    show_violations(result)
