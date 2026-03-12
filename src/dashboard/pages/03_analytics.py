"""Streamlit sayfası — İstatistikler ve grafikler."""

import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.database import ViolationDatabase

st.set_page_config(page_title="Analitik", layout="wide")
st.title("Analitik ve İstatistikler")


def main():
    db_path = PROJECT_ROOT / "results" / "violations.db"
    if not db_path.exists():
        st.warning("Veritabanı bulunamadı.")
        return

    db = ViolationDatabase(str(db_path))
    stats = db.get_statistics()
    violations = db.get_all_violations(limit=1000)

    if not violations:
        st.info("Analiz için yeterli veri yok.")
        db.close()
        return

    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    df = pd.DataFrame(violations)

    # --- Özet Metrikler ---
    st.subheader("Genel Bakış")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam İhlal", stats["total_violations"])
    c2.metric("Plaka Okunan", stats["with_plate"])
    c3.metric("Plaka Okuma Oranı", f"{stats['plate_detection_rate']:.1%}")
    c4.metric("Geçerli Plaka Oranı", f"{stats['plate_validation_rate']:.1%}")

    st.divider()

    # --- Araç Sınıfı Dağılımı ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Araç Sınıfı Dağılımı")
        if "vehicle_class" in df.columns:
            class_counts = df["vehicle_class"].value_counts()
            fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                hole=0.4,
                title="Araç Türlerine Göre İhlal Dağılımı",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Güvenilirlik Dağılımı")
        if "vehicle_confidence" in df.columns:
            fig = px.histogram(
                df, x="vehicle_confidence",
                nbins=20,
                title="Araç Tespit Güvenilirlik Dağılımı",
                labels={"vehicle_confidence": "Güvenilirlik Skoru"},
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Zaman Serisi ---
    st.subheader("Zamana Göre İhlal Dağılımı")
    if "timestamp_sec" in df.columns:
        df["timestamp_min"] = df["timestamp_sec"] / 60
        fig = px.histogram(
            df, x="timestamp_min",
            nbins=30,
            title="Videonun Dakikalarına Göre İhlal Sayısı",
            labels={"timestamp_min": "Dakika"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Plaka İstatistikleri ---
    st.subheader("Plaka OCR Performansı")
    col3, col4 = st.columns(2)

    with col3:
        if "plate_confidence" in df.columns:
            plate_df = df[df["plate_text"].notna() & (df["plate_text"] != "")]
            if not plate_df.empty:
                fig = px.histogram(
                    plate_df, x="plate_confidence",
                    nbins=20,
                    title="Plaka OCR Güvenilirlik Dağılımı",
                    labels={"plate_confidence": "OCR Güvenilirlik"},
                )
                st.plotly_chart(fig, use_container_width=True)

    with col4:
        if "city_name" in df.columns:
            city_df = df[df["city_name"].notna()]
            if not city_df.empty:
                city_counts = city_df["city_name"].value_counts().head(10)
                fig = px.bar(
                    x=city_counts.index,
                    y=city_counts.values,
                    title="İl Bazlı İhlal Dağılımı (Top 10)",
                    labels={"x": "İl", "y": "İhlal Sayısı"},
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- Ham Veri ---
    st.subheader("Ham Veri")
    with st.expander("Tüm verileri göster"):
        st.dataframe(df, use_container_width=True)

    db.close()


if __name__ == "__main__":
    main()
