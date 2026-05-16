#!/bin/bash
# Fast merge using bash + cp -l (hardlink mode, instant)

set -e

PROJECT_ROOT="/Users/sertacakalin/Desktop/Proje/bitirme/hatched-area-violation-detection"
ROBOFLOW_DIR="/Users/sertacakalin/Desktop/istanbul-traffic-vehicles"
AUTO_DIR="$PROJECT_ROOT/data/datasets/auto_labeled"
FINAL="$PROJECT_ROOT/data/datasets/final_v3"

# Temizle
rm -rf "$FINAL"
mkdir -p "$FINAL"/all/images "$FINAL"/all/labels

echo "1/4: Roboflow frame'leri prefix'le hardlink..."
for img in "$ROBOFLOW_DIR"/train/images/*.jpg; do
  name=$(basename "$img")
  cp -l "$img" "$FINAL/all/images/roboflow_$name"
done
for lbl in "$ROBOFLOW_DIR"/train/labels/*.txt; do
  name=$(basename "$lbl")
  cp -l "$lbl" "$FINAL/all/labels/roboflow_$name"
done
echo "  Roboflow done: $(ls $FINAL/all/images/ | grep -c roboflow_) image"

echo "2/4: Auto-labeled hardlink..."
for src in file1_night file2_dense file3_day file4_evening; do
  count=0
  for img in "$AUTO_DIR/$src/images"/*.jpg; do
    name=$(basename "$img")
    cp -l "$img" "$FINAL/all/images/${src}_${name}"
    count=$((count+1))
  done
  for lbl in "$AUTO_DIR/$src/labels"/*.txt; do
    name=$(basename "$lbl")
    cp -l "$lbl" "$FINAL/all/labels/${src}_${name}"
  done
  echo "  $src: $count image"
done

echo "3/4: Train/Val/Test split (70/20/10)..."
mkdir -p "$FINAL"/train/images "$FINAL"/train/labels
mkdir -p "$FINAL"/valid/images "$FINAL"/valid/labels
mkdir -p "$FINAL"/test/images "$FINAL"/test/labels

# Listele ve seed'li shuffle
ALL_IMAGES=$(mktemp)
ls "$FINAL/all/images/" > "$ALL_IMAGES"
TOTAL=$(wc -l < "$ALL_IMAGES")

# Reproducible shuffle (Python ile, seed=42)
SHUFFLED=$(mktemp)
python3 -c "
import random
with open('$ALL_IMAGES') as f:
    lines = f.read().splitlines()
random.seed(42)
random.shuffle(lines)
with open('$SHUFFLED', 'w') as f:
    f.write('\n'.join(lines))
"

TRAIN_END=$(echo "$TOTAL * 70 / 100" | bc)
VAL_END=$(echo "$TOTAL * 90 / 100" | bc)

echo "  Total: $TOTAL"
echo "  Train: 0..$TRAIN_END"
echo "  Valid: $TRAIN_END..$VAL_END"
echo "  Test:  $VAL_END..$TOTAL"

# Train
sed -n "1,${TRAIN_END}p" "$SHUFFLED" | while read img; do
  stem="${img%.jpg}"
  mv "$FINAL/all/images/$img" "$FINAL/train/images/$img"
  mv "$FINAL/all/labels/${stem}.txt" "$FINAL/train/labels/${stem}.txt"
done

# Valid
sed -n "$((TRAIN_END+1)),${VAL_END}p" "$SHUFFLED" | while read img; do
  stem="${img%.jpg}"
  mv "$FINAL/all/images/$img" "$FINAL/valid/images/$img"
  mv "$FINAL/all/labels/${stem}.txt" "$FINAL/valid/labels/${stem}.txt"
done

# Test
sed -n "$((VAL_END+1)),${TOTAL}p" "$SHUFFLED" | while read img; do
  stem="${img%.jpg}"
  mv "$FINAL/all/images/$img" "$FINAL/test/images/$img"
  mv "$FINAL/all/labels/${stem}.txt" "$FINAL/test/labels/${stem}.txt"
done

# all/ kalanı temizle
rm -rf "$FINAL/all"
rm "$ALL_IMAGES" "$SHUFFLED"

echo "4/4: data.yaml oluştur..."
cat > "$FINAL/data.yaml" <<EOF
train: $FINAL/train/images
val:   $FINAL/valid/images
test:  $FINAL/test/images

nc: 4
names: ['bus', 'car', 'motorcycle', 'truck']
EOF

echo ""
echo "=== BİTTİ ==="
echo "Train: $(ls $FINAL/train/images/ | wc -l) image"
echo "Valid: $(ls $FINAL/valid/images/ | wc -l) image"
echo "Test:  $(ls $FINAL/test/images/ | wc -l) image"
echo "Disk:  $(du -sh $FINAL | cut -f1)"
