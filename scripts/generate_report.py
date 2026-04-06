"""HTML ihlal raporu oluştur — tarayıcıda aç veya PDF'e çevir.

Kullanım:
    python scripts/generate_report.py --db results/violations.db

Çıktı: results/report.html (tarayıcıda aç, Ctrl+P ile PDF yap)
"""

import argparse
import base64
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import ViolationDatabase


def image_to_base64(path: str) -> str:
    """Görüntüyü base64'e çevir (HTML'e gömmek için)."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def generate_html(db_path: str, output_path: str) -> None:
    db = ViolationDatabase(db_path)
    stats = db.get_statistics()
    violations = db.get_all_violations(limit=100)
    db.close()

    # İhlal satırları
    rows_html = ""
    for i, v in enumerate(violations, 1):
        crop_b64 = ""
        crop_path = v.get("vehicle_crop_path", "")
        if crop_path and Path(crop_path).exists():
            crop_b64 = image_to_base64(crop_path)

        img_tag = f'<img src="data:image/jpeg;base64,{crop_b64}" style="max-width:120px;max-height:80px;">' if crop_b64 else "—"

        severity = v.get("severity_score", 0) or 0
        sev_level = v.get("severity_level", "") or ""
        viol_type = v.get("violation_type", "") or ""
        plate = v.get("plate_text", "") or "Okunamadı"

        # Renk
        if severity >= 75: color = "#9C27B0"
        elif severity >= 50: color = "#F44336"
        elif severity >= 25: color = "#FF9800"
        else: color = "#4CAF50"

        rows_html += f"""
        <tr>
            <td>{i}</td>
            <td>{img_tag}</td>
            <td>{v.get('vehicle_class','—')}</td>
            <td>{v.get('timestamp_sec',0):.1f}s</td>
            <td style="color:{color};font-weight:bold;">{severity:.0f}</td>
            <td>{sev_level}</td>
            <td>{viol_type}</td>
            <td>{plate}</td>
        </tr>"""

    # Dağılım
    class_dist = stats.get("class_distribution", {})
    class_html = ", ".join(f"{k}: {v}" for k, v in class_dist.items()) or "—"

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>İhlal Tespit Raporu</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
    h2 {{ color: #283593; margin-top: 30px; }}
    .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
    .stat-card {{
        background: #f5f5f5; border-radius: 10px; padding: 20px;
        text-align: center; flex: 1; border-left: 4px solid #1a237e;
    }}
    .stat-card .number {{ font-size: 2em; font-weight: bold; color: #1a237e; }}
    .stat-card .label {{ font-size: 0.9em; color: #666; }}
    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
    th {{ background: #1a237e; color: white; padding: 10px; text-align: left; }}
    td {{ padding: 8px; border-bottom: 1px solid #ddd; vertical-align: middle; }}
    tr:hover {{ background: #f5f5f5; }}
    .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd;
               color: #999; font-size: 0.8em; }}
    @media print {{
        body {{ margin: 20px; }}
        .stat-card {{ border: 1px solid #ddd; }}
    }}
</style>
</head>
<body>

<h1>Taralı Alan İhlal Tespit Raporu</h1>
<p><strong>Tarih:</strong> {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
<p><strong>Sistem:</strong> YOLOv8 + ByteTrack + Yörünge Analizi + Şiddet Skorlama</p>

<div class="stats">
    <div class="stat-card">
        <div class="number">{stats.get('total_violations', 0)}</div>
        <div class="label">Toplam İhlal</div>
    </div>
    <div class="stat-card">
        <div class="number">{stats.get('with_plate', 0)}</div>
        <div class="label">Plaka Okunan</div>
    </div>
    <div class="stat-card">
        <div class="number">{stats.get('plate_detection_rate', 0):.0%}</div>
        <div class="label">Plaka Oranı</div>
    </div>
    <div class="stat-card">
        <div class="number">{class_html}</div>
        <div class="label">Araç Dağılımı</div>
    </div>
</div>

<h2>İhlal Detayları</h2>
<table>
    <thead>
        <tr>
            <th>#</th>
            <th>Görüntü</th>
            <th>Araç</th>
            <th>Zaman</th>
            <th>Skor</th>
            <th>Seviye</th>
            <th>Tip</th>
            <th>Plaka</th>
        </tr>
    </thead>
    <tbody>
        {rows_html}
    </tbody>
</table>

<div class="footer">
    <p>Bu rapor otomatik olarak oluşturulmuştur.</p>
    <p>Sertaç Akalın — İstanbul Arel Üniversitesi — Bitirme Projesi 2026</p>
</div>

</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Rapor oluşturuldu: {output_path}")
    print("Tarayıcıda aç, Ctrl+P ile PDF olarak kaydet.")


def main():
    parser = argparse.ArgumentParser(description="İhlal raporu oluştur")
    parser.add_argument("--db", default="results/violations.db")
    parser.add_argument("--output", default="results/report.html")
    args = parser.parse_args()
    generate_html(args.db, args.output)


if __name__ == "__main__":
    main()
