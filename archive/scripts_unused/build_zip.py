"""
Build final_v3.zip ready for Colab upload.
Streams files directly into zip — no intermediate copy.
"""
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "data/datasets/final_v3"
ZIP_PATH = PROJECT_ROOT / "data/datasets/final_v3.zip"

# Read path lists
splits = {}
for split in ["train", "valid", "test"]:
    txt = OUT / f"{split}.txt"
    splits[split] = txt.read_text().strip().split("\n")

total = sum(len(v) for v in splits.values())
print(f"Total: {total} pairs")

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_STORED) as z:
    # data.yaml — Colab path version
    yaml_colab = """train: /content/final_v3/train/images
val:   /content/final_v3/valid/images
test:  /content/final_v3/test/images

nc: 4
names: ['bus', 'car', 'motorcycle', 'truck']
"""
    z.writestr("final_v3/data.yaml", yaml_colab)

    written = 0
    for split, paths in splits.items():
        print(f"\n{split}: {len(paths)} pairs")
        for img_path in paths:
            img = Path(img_path)
            if not img.exists():
                continue
            # Find label
            lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
            if not lbl.exists():
                continue

            # Source-prefix to avoid name collisions
            src_tag = img.parent.parent.name  # "train" for roboflow, "file1_night" etc
            new_name = f"{src_tag}_{img.name}"

            z.write(img, arcname=f"final_v3/{split}/images/{new_name}")
            z.write(lbl, arcname=f"final_v3/{split}/labels/{Path(new_name).stem}.txt")

            written += 1
            if written % 1000 == 0:
                print(f"  written: {written}/{total}")

print(f"\nWritten: {written}")
print(f"Zip: {ZIP_PATH}")
print(f"Size: {ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MB")
