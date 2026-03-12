"""Streamlit Dashboard — Ana sayfa."""

import streamlit as st
from pathlib import Path
import sys

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.database import ViolationDatabase

# Sayfa ayarları
st.set_page_config(
    page_title="Taralı Alan İhlal Tespit Sistemi",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("Taralı Alan İhlal Tespit Sistemi")
    st.markdown("Istanbul trafiğinde taralı alanlara giren araçların otomatik tespiti")

    # Sidebar
    st.sidebar.title("Navigasyon")
    st.sidebar.info(
        "Bu sistem, taralı alanlara giren araçları tespit eder, "
        "plakalarını okur ve kayıt altına alır."
    )

    # Veritabanı bağlantısı
    db_path = PROJECT_ROOT / "results" / "violations.db"
    if not db_path.exists():
        st.warning("Henüz ihlal verisi yok. Pipeline'ı çalıştırarak veri oluşturun.")
        st.code("python scripts/run_pipeline.py --config configs/config.yaml", language="bash")
        return

    db = ViolationDatabase(str(db_path))
    stats = db.get_statistics()

    # Özet metrikler
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Toplam İhlal", stats["total_violations"])
    with col2:
        st.metric("Plaka Okunan", stats["with_plate"])
    with col3:
        rate = f"{stats['plate_detection_rate']:.1%}"
        st.metric("Plaka Okuma Oranı", rate)
    with col4:
        valid_rate = f"{stats['plate_validation_rate']:.1%}"
        st.metric("Geçerli Plaka Oranı", valid_rate)

    st.divider()

    # Araç sınıfı dağılımı
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Araç Sınıfı Dağılımı")
        if stats["class_distribution"]:
            import plotly.express as px
            import pandas as pd
            df = pd.DataFrame(
                list(stats["class_distribution"].items()),
                columns=["Sınıf", "Sayı"],
            )
            fig = px.pie(df, values="Sayı", names="Sınıf", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Bölge Dağılımı")
        if stats["zone_distribution"]:
            import plotly.express as px
            import pandas as pd
            df = pd.DataFrame(
                list(stats["zone_distribution"].items()),
                columns=["Bölge", "Sayı"],
            )
            fig = px.bar(df, x="Bölge", y="Sayı")
            st.plotly_chart(fig, use_container_width=True)

    # Son ihlaller
    st.subheader("Son İhlaller")
    violations = db.get_all_violations(limit=10)
    if violations:
        import pandas as pd
        df = pd.DataFrame(violations)
        display_cols = [
            "event_id", "vehicle_class", "zone_id",
            "plate_text", "plate_valid", "timestamp_sec", "created_at",
        ]
        existing_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[existing_cols], use_container_width=True)
    else:
        st.info("Henüz ihlal kaydı yok.")

    db.close()


if __name__ == "__main__":
    main()
