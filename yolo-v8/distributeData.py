import os
import shutil
import random


DATA_DIR = "data"
TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"

TRAIN_RATIO = 0.7   # 70% train
VAL_RATIO = 0.15    # 15% val
TEST_RATIO = 0.15   # 15% test

SEED = 42

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def clear_directory(path):
    if not os.path.exists(path):
        return
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

def get_images(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]


def split_dataset():
    random.seed(SEED)

    # Wyczyść katalogi
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        clear_directory(d)
        os.makedirs(d, exist_ok=True)

    classes = [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ]

    if not classes:
        raise ValueError("Brak folderów klas w katalogu 'data'.")

    for cls in classes:
        class_path = os.path.join(DATA_DIR, cls)
        images = get_images(class_path)

        if len(images) < 3:
            print(f"[WARN] Za mało obrazów w klasie: {cls}")
            continue

        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for split_name, split_imgs in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs)
        ]:
            split_dir = os.path.join(split_name, cls)
            os.makedirs(split_dir, exist_ok=True)

            for img in split_imgs:
                shutil.copy2(img, split_dir)

        print(
            f"[OK] {cls}: "
            f"{len(train_imgs)} train / "
            f"{len(val_imgs)} val / "
            f"{len(test_imgs)} test"
        )

    print("\n✅ Podział datasetu (train/val/test) zakończony.")


if __name__ == "__main__":
    split_dataset()
