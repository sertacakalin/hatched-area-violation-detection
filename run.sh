#!/bin/bash
# Taralı alan ihlal tespit — tam pipeline çalıştırıcı
#
# Kullanım:
#   ./run.sh              → zone çiz + pretrained ile koş
#   ./run.sh finetuned    → pretrained karşılaştırması (zone yeniden çizilmez)
#   ./run.sh both         → önce pretrained, sonra fine-tuned, sonuçları karşılaştır
set -e

VIDEO="data/videos/test/cam4_30s.mp4"
ZONE="configs/zones/cam4_30s.json"

MODE="${1:-pretrained}"

draw_zone_if_missing() {
  if [ ! -f "$ZONE" ]; then
    echo "=== Zone polygon çizimi ==="
    echo "OpenCV penceresi açılacak. Sol tık=nokta, Sağ tık=geri al, r=sıfırla, q=bitir"
    echo ""
    venv/bin/python scripts/select_roi.py --video "$VIDEO" --output "$ZONE" --zone-id zone_01 --name "Tarali Alan"
  else
    echo "Zone zaten var: $ZONE (yeniden çizmek için önce sil)"
  fi
}

run_pretrained() {
  echo ""
  echo "=== Pretrained YOLOv8s (COCO) koşumu ==="
  venv/bin/python scripts/run_pipeline.py \
    --video "$VIDEO" \
    --zone "$ZONE" \
    --weights weights/yolov8s.pt \
    --classes 2,3,5,7 \
    --output results/cam4_30s_pretrained
}

run_finetuned() {
  echo ""
  echo "=== Fine-tuned best.pt (Istanbul) koşumu ==="
  venv/bin/python scripts/run_pipeline.py \
    --video "$VIDEO" \
    --zone "$ZONE" \
    --weights weights/best.pt \
    --classes 0,1,2,3 \
    --output results/cam4_30s_finetuned
}

draw_zone_if_missing

case "$MODE" in
  pretrained)
    run_pretrained
    open results/cam4_30s_pretrained/output.mp4
    ;;
  finetuned)
    run_finetuned
    open results/cam4_30s_finetuned/output.mp4
    ;;
  both)
    run_pretrained
    run_finetuned
    echo ""
    echo "=== KARŞILAŞTIRMA ==="
    echo "Pretrained:  results/cam4_30s_pretrained/output.mp4"
    echo "Fine-tuned:  results/cam4_30s_finetuned/output.mp4"
    ;;
  *)
    echo "Bilinmeyen mod: $MODE (kullan: pretrained | finetuned | both)"
    exit 1
    ;;
esac
