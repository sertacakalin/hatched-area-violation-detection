"""
Fast merge using hardlinks (instant, no actual copy).
Both source and target on same disk → hardlinks work.
"""
import os
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROBOFLOW_EXPORT = Path("/Users/sertacakalin/Desktop/istanbul-traffic-vehicles")
AUTO_LABELED_ROOT = PROJECT_ROOT / "data/datasets/auto_labeled"
FINAL_DATASET = PROJECT_ROOT / "data/datasets/final_v3"
CLASSES = ["bus", "car", "motorcycle", "truck"]
TRAIN_RATIO, VALID_RATIO = 0.70, 0.20
SEED = 42


def hardlink(src, dst):
    """Hardlink instead of copy — instant, no disk usage."""
    try:
        os.link(src, dst)
    except FileExistsError:
        pass
    except OSError:
        # Cross-filesystem fallback to copy
        shutil.copyfile(src, dst)


def main():
    pairs = []

    # Roboflow source
    rf_imgs = ROBOFLOW_EXPORT / "train/images"
    rf_lbls = ROBOFLOW_EXPORT / "train/labels"
    for img in rf_imgs.glob("*.jpg"):
        lbl = rf_lbls / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl, "roboflow"))
    print(f"Roboflow: {len([p for p in pairs if p[2]=='roboflow'])}")

    # Auto-labeled
    for f in ["file1_night", "file2_dense", "file3_day", "file4_evening"]:
        d = AUTO_LABELED_ROOT / f
        for img in (d / "images").glob("*.jpg"):
            lbl = d / "labels" / f"{img.stem}.txt"
            if lbl.exists():
                pairs.append((img, lbl, f))
    print(f"Total: {len(pairs)} pairs")

    random.seed(SEED)
    random.shuffle(pairs)
    n = len(pairs)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VALID_RATIO))
    splits = {
        "train": pairs[:train_end],
        "valid": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    if FINAL_DATASET.exists():
        shutil.rmtree(FINAL_DATASET)

    for split, items in splits.items():
        img_d = FINAL_DATASET / split / "images"
        lbl_d = FINAL_DATASET / split / "labels"
        img_d.mkdir(parents=True, exist_ok=True)
        lbl_d.mkdir(parents=True, exist_ok=True)
        print(f"{split}: hardlinking {len(items)} pairs...")
        for img, lbl, source in items:
            new_name = f"{source}_{img.name}"
            hardlink(img, img_d / new_name)
            hardlink(lbl, lbl_d / f"{Path(new_name).stem}.txt")

    yaml = f"""train: {FINAL_DATASET}/train/images
val:   {FINAL_DATASET}/valid/images
test:  {FINAL_DATASET}/test/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    (FINAL_DATASET / "data.yaml").write_text(yaml)

    print()
    print("DONE")
    for split, items in splits.items():
        print(f"  {split}: {len(items)}")


if __name__ == "__main__":
    main()
