#!/bin/bash
# Video'dan kesit al + v3 modeliyle test et.
#
# Kullanim:
#   ./test_clip.sh <video> [start] [duration] [mode]
#
# Ornek:
#   ./test_clip.sh ~/Downloads/cam6.MOV 00:02:00 10 detect
#   ./test_clip.sh ~/Downloads/cam6.MOV 00:02:00 10 pipeline
#
# Parametreler:
#   video     : Kaynak video yolu (zorunlu)
#   start     : Baslangic zamani (HH:MM:SS, varsayilan 00:00:00)
#   duration  : Sure (saniye, varsayilan 10)
#   mode      : detect (sadece YOLO) | pipeline (zone + ihlal tespiti)
#               varsayilan: detect
set -e

VIDEO="${1:?Video yolu gerekli. Ornek: ./test_clip.sh ~/Downloads/cam6.MOV 00:02:00 10}"
START="${2:-00:00:00}"
DURATION="${3:-10}"
MODE="${4:-detect}"

if [ ! -f "$VIDEO" ]; then
  echo "HATA: Video bulunamadi: $VIDEO"
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Python yorumlayicisi: venv bozuksa conda base'e dus
if [ -x "venv/bin/python" ] && venv/bin/python -c "import sys" 2>/dev/null; then
  PY="venv/bin/python"
else
  PY="$(which python)"
  echo "(venv bozuk veya yok, conda base kullaniliyor: $PY)"
fi

# Kesit ismi: <video-stem>_<start-tag>_<duration>s.mp4
STEM=$(basename "$VIDEO" | sed 's/\.[^.]*$//' | tr '[:upper:]' '[:lower:]')
START_TAG=$(echo "$START" | tr ':' '_')
CLIP_NAME="${STEM}_${START_TAG}_${DURATION}s.mp4"
CLIP_PATH="data/videos/test/${CLIP_NAME}"
RESULT_NAME="${STEM}_${START_TAG}_${DURATION}s_v3"
ZONE_PATH="configs/zones/${STEM}_${START_TAG}_${DURATION}s.json"

mkdir -p data/videos/test results configs/zones

# 1) Kesit al (varsa atla)
if [ ! -f "$CLIP_PATH" ]; then
  echo "=== Kesit aliniyor: $START ($DURATION sn) ==="
  ffmpeg -ss "$START" -i "$VIDEO" -t "$DURATION" -c copy -y "$CLIP_PATH" 2>&1 | tail -3
  echo "Kesit: $CLIP_PATH"
else
  echo "Kesit zaten var: $CLIP_PATH"
fi

# 2) Calistir
case "$MODE" in
  detect)
    echo ""
    echo "=== YOLO Detection (v3) ==="
    "$PY" -c "
from ultralytics import YOLO
import collections
m = YOLO('weights/best_v4.pt')
print('Classes:', m.names)
results = m.predict(
    source='$CLIP_PATH', conf=0.35, save=True,
    project='results', name='$RESULT_NAME', exist_ok=True, verbose=False,
)
counts = collections.Counter()
for r in results:
    if r.boxes is not None:
        for c in r.boxes.cls.tolist():
            counts[m.names[int(c)]] += 1
print(f'Frames: {len(results)}')
print(f'Detections: {dict(counts)}')
"
    OUT_VIDEO=$(find "results/$RESULT_NAME" -name "*.avi" -o -name "*.mp4" | head -1)
    [ -f "$OUT_VIDEO" ] && open "$OUT_VIDEO" || echo "Cikti video bulunamadi"
    ;;

  pipeline)
    if [ ! -f "$ZONE_PATH" ]; then
      echo ""
      echo "=== Zone cizimi gerekli ==="
      echo "OpenCV penceresi acilacak. Sol tik=nokta, Sag tik=geri al, r=sifirla, q=bitir"
      "$PY" scripts/select_roi.py \
        --video "$CLIP_PATH" --output "$ZONE_PATH" \
        --zone-id zone_01 --name "Tarali Alan"
    else
      echo "Zone zaten var: $ZONE_PATH"
    fi
    echo ""
    echo "=== Pipeline (v3 + ihlal tespiti) ==="
    "$PY" scripts/run_pipeline.py \
      --video "$CLIP_PATH" \
      --zone "$ZONE_PATH" \
      --weights weights/best_v4.pt \
      --classes 0,1,2,3 \
      --output "results/${RESULT_NAME}_pipeline"
    open "results/${RESULT_NAME}_pipeline/output.mp4" 2>/dev/null || true
    ;;

  *)
    echo "Bilinmeyen mod: $MODE (kullan: detect | pipeline)"
    exit 1
    ;;
esac

echo ""
echo "=== Bitti ==="
echo "Kesit:  $CLIP_PATH"
echo "Sonuc:  results/$RESULT_NAME${MODE:+_$MODE}/"
