#!/bin/bash
# v4 dataset'i Colab'a yuklemek icin tek zip dosyasi olusturur.
# Image'lar ve label'lar Colab disk path'lerinde olacak sekilde duzenler.
#
# Cikti: ~/Desktop/havd_v4_dataset.zip (~5 GB)
# Sonra: Google Drive > MyDrive > havd > havd_v4_dataset.zip
#         Google Drive > MyDrive > havd > best_v3.pt
#
# Kullanim: ./scripts/prepare_colab_zip.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

STAGE="/tmp/havd_v4_colab"
ZIP_OUT="$HOME/Desktop/havd_v4_dataset.zip"

echo "=== Stage dizini temizle ==="
rm -rf "$STAGE"
mkdir -p "$STAGE"

echo "=== Roboflow kopyala ==="
mkdir -p "$STAGE/roboflow/train"
cp -r /Users/sertacakalin/Desktop/istanbul-traffic-vehicles/train/images "$STAGE/roboflow/train/"
cp -r /Users/sertacakalin/Desktop/istanbul-traffic-vehicles/train/labels "$STAGE/roboflow/train/"

echo "=== cleaned/cam10 kopyala (symlink resolve) ==="
mkdir -p "$STAGE/cleaned/cam10/images" "$STAGE/cleaned/cam10/labels"
cp -L data/datasets/cleaned/cam10/images/*.jpg "$STAGE/cleaned/cam10/images/" 2>/dev/null || true
cp data/datasets/cleaned/cam10/labels/*.txt "$STAGE/cleaned/cam10/labels/"

echo "=== cleaned/cam11 kopyala ==="
mkdir -p "$STAGE/cleaned/cam11/images" "$STAGE/cleaned/cam11/labels"
cp -L data/datasets/cleaned/cam11/images/*.jpg "$STAGE/cleaned/cam11/images/" 2>/dev/null || true
cp data/datasets/cleaned/cam11/labels/*.txt "$STAGE/cleaned/cam11/labels/"

echo "=== auto_labeled pseudo klasorler kopyala ==="
for f in file1_night file3_day file4_evening; do
  mkdir -p "$STAGE/auto_labeled/$f/images" "$STAGE/auto_labeled/$f/labels"
  cp data/datasets/auto_labeled/$f/images/*.jpg "$STAGE/auto_labeled/$f/images/" 2>/dev/null
  cp data/datasets/auto_labeled/$f/labels/*.txt "$STAGE/auto_labeled/$f/labels/"
done

echo "=== Boyut ==="
du -sh "$STAGE"

echo "=== Sikistir ==="
cd /tmp
rm -f "$ZIP_OUT"
zip -r -q "$ZIP_OUT" havd_v4_colab/
ls -lh "$ZIP_OUT"

echo ""
echo "BITTI."
echo "Yapacaklarin:"
echo "  1. $ZIP_OUT dosyasini Google Drive'a yukle (MyDrive/havd/havd_v4_dataset.zip)"
echo "  2. weights/best_v3.pt'yi de Drive'a yukle (MyDrive/havd/best_v3.pt)"
echo "  3. Colab'da scripts/notebooks/08_train_mobese_v4.ipynb'i ac, sirayla calistir"
