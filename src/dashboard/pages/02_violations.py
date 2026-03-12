"""Streamlit sayfası — İhlal tablosu + galeri görünümü."""

import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.database import ViolationDatabase

st.set_page_config(page_title="İhlaller", layout="wide")
st.title("İhlal Kayıtları")


def main():
    db_path = PROJECT_ROOT / "results" / "violations.db"
    if not db_path.exists():
        st.warning("Veritabanı bulunamadı.")
        return

    db = ViolationDatabase(str(db_path))

    # Filtreler
    st.sidebar.header("Filtreler")
    search_plate = st.sidebar.text_input("Plaka Ara")
    vehicle_class = st.sidebar.selectbox(
        "Araç Sınıfı", ["Tümü", "car", "bus", "truck", "motorcycle", "minibus"]
    )
    view_mode = st.sidebar.radio("Görünüm", ["Tablo", "Galeri"])

    # Veri çek
    if search_plate:
        violations = db.get_violations_by_plate(search_plate)
    else:
        violations = db.get_all_violations(limit=50)

    # Araç sınıfı filtresi
    if vehicle_class != "Tümü":
        violations = [v for v in violations if v.get("vehicle_class") == vehicle_class]

    st.markdown(f"**{len(violations)} ihlal bulundu**")

    if not violations:
        st.info("Filtrelere uygun ihlal bulunamadı.")
        db.close()
        return

    if view_mode == "Tablo":
        import pandas as pd
        df = pd.DataFrame(violations)
        display_cols = [
            "event_id", "track_id", "vehicle_class", "vehicle_confidence",
            "zone_id", "plate_text", "plate_confidence", "plate_valid",
            "timestamp_sec", "created_at",
        ]
        existing_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[existing_cols], use_container_width=True)

    else:  # Galeri
        cols = st.columns(3)
        for i, v in enumerate(violations):
            col = cols[i % 3]
            with col:
                st.markdown(f"**İhlal: {v.get('event_id', 'N/A')}**")

                # Araç kırpması
                crop_path = v.get("vehicle_crop_path")
                if crop_path and Path(crop_path).exists():
                    st.image(crop_path, use_container_width=True)

                # Detaylar
                st.markdown(f"""
                - **Araç:** {v.get('vehicle_class', 'N/A')}
                - **Plaka:** {v.get('plate_text', 'Okunamadı')}
                - **Bölge:** {v.get('zone_id', 'N/A')}
                - **Zaman:** {v.get('timestamp_sec', 0):.1f}s
                - **Geçerli:** {'Evet' if v.get('plate_valid') else 'Hayır'}
                """)
                st.divider()

    db.close()


if __name__ == "__main__":
    main()
