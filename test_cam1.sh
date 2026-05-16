#!/bin/bash
# Cam1 — best_v3.pt ile pipeline test (tek satır kullanım için)
set -e
cd /Users/sertacakalin/Desktop/Projects/bitirme/hatched-area-violation-detection
source venv/bin/activate

rm -rf results/cam1_test
rm -f results/violations.db

python scripts/run_pipeline.py \
  --video data/videos/test/cam1_test.mp4 \
  --zone configs/zones/cam1_test.json \
  --weights weights/best_v3.pt \
  --output results/cam1_test

echo ""
echo "=== TAMAMLANDI ==="
echo "Output: results/cam1_test/output.mp4"
echo "İhlal tablosu: python scripts/show_violations.py results/cam1_test"
echo ""
open results/cam1_test/output.mp4
