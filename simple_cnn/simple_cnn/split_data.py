import os
import shutil
import random
from pathlib import Path

RAW_DIR       = Path(__file__).parent.parent / 'data' / 'raw'
PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

SEED = 42


def split_data():
    random.seed(SEED)

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {RAW_DIR}")

    classes = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    if not classes:
        raise ValueError(f"No class subfolders found in: {RAW_DIR}")

    print(f"Found classes ({len(classes)}): {[c.name for c in classes]}")
    print(f"Split: train={TRAIN_RATIO:.0%} | val={VAL_RATIO:.0%} | test={1-TRAIN_RATIO-VAL_RATIO:.0%}")
    print(f"Destination: {PROCESSED_DIR}\n")

    for split in ['train', 'val', 'test']:
        (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        images = [
            f for f in cls.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            'train': images[:n_train],
            'val':   images[n_train:n_train + n_val],
            'test':  images[n_train + n_val:],
        }

        print(f"  {cls.name:20s} — total: {n:4d} | "
              f"train: {len(splits['train']):4d} | "
              f"val: {len(splits['val']):4d} | "
              f"test: {len(splits['test']):4d}")

        for split, files in splits.items():
            dest_dir = PROCESSED_DIR / split / cls.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest_dir / f.name)

    print("Split completed successfully.")


if __name__ == '__main__':
    split_data()