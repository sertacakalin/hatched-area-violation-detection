#!/bin/bash
# Label Studio'yu local file serving aktif sekilde baslatir.
# Auto-labeled klasoru dogrudan disk uzerinden serve eder.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${PROJECT_DIR}/data/datasets/auto_labeled"

if [ ! -d "$DATA_ROOT" ]; then
  echo "HATA: $DATA_ROOT yok"
  exit 1
fi

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$DATA_ROOT"

echo "=== Label Studio Baslatiliyor ==="
echo "Local files root: $DATA_ROOT"
echo "URL: http://localhost:8080"
echo "Durdurmak icin: Ctrl+C"
echo ""

label-studio start --port 8080
