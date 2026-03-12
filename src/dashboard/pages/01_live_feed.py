"""Streamlit sayfası — Canlı video akışı + overlay."""

import streamlit as st
from pathlib import Path
import sys
import cv2
import tempfile

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Canlı Görüntü", layout="wide")
st.title("Canlı Video Akışı")


def main():
    st.sidebar.header("Video Ayarları")

    # Video kaynağı
    video_source = st.sidebar.text_input(
        "Video Yolu",
        value="data/videos/test/sample.mp4",
    )
    video_path = PROJECT_ROOT / video_source

    if not video_path.exists():
        st.warning(f"Video dosyası bulunamadı: {video_path}")
        uploaded = st.file_uploader("Video yükleyin", type=["mp4", "avi", "mov"])
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                f.write(uploaded.read())
                video_path = Path(f.name)
        else:
            return

    # Video oynatma
    st.video(str(video_path))

    # İşlenmiş video
    output_path = PROJECT_ROOT / "results" / "output.mp4"
    if output_path.exists():
        st.subheader("İşlenmiş Video (Overlay)")
        st.video(str(output_path))
    else:
        st.info(
            "İşlenmiş video henüz oluşturulmamış. "
            "Pipeline'ı çalıştırarak oluşturabilirsiniz."
        )

    # Kare analizi
    st.subheader("Kare Analizi")
    frame_num = st.slider("Kare Numarası", 0, 1000, 0)

    if st.button("Kareyi Analiz Et"):
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=f"Kare #{frame_num}", use_container_width=True)
        else:
            st.error("Kare okunamadı.")


if __name__ == "__main__":
    main()
