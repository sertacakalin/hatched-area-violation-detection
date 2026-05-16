#!/bin/bash
# Eski model (best.pt) ile pipeline test
set -e
cd /Users/sertacakalin/Desktop/Projects/bitirme/hatched-area-violation-detection
source venv/bin/activate

# Her test için temiz DB (eski stats karışmasın)
rm -f results/violations.db

python scripts/run_pipeline.py \
  --video data/videos/test/cam4_30s.mp4 \
  --zone configs/zones/cam4_30s.json \
  --weights weights/best.pt \
  --output results/cam4_30s_best
echo ""
echo "=== TAMAMLANDI ==="
echo "Output video: results/cam4_30s_best/output.mp4"
echo "Otomatik açılıyor..."
open results/cam4_30s_best/output.mp4
