"""
data_prep.py
------------
Prepares the image dataset for training.

WHAT IT DOES
  Takes a "raw" dataset that is organized as one folder per class:

      raw_dataset/
        Tomato___Early_blight/   img1.jpg, img2.jpg, ...
        Tomato___Late_blight/    ...
        Tomato___Leaf_Mold/      ...
        Tomato___healthy/        ...
        Potato___Early_blight/   ...
        Potato___healthy/        ...

  ...and splits each class into train / val / test (default 80/10/10),
  copying the images into:

      data/
        train/<class>/...
        val/<class>/...
        test/<class>/...

WHERE TO GET THE RAW DATA
  This project uses the "PlantVillage" dataset. Because Kaggle requires
  authentication (and to keep this laptop-friendly), the recommended flow is:

    1. Download PlantVillage from Kaggle, e.g.:
       https://www.kaggle.com/datasets/emmarex/plantdisease
       (or any PlantVillage mirror).
    2. Keep ONLY the class folders listed in KEEP_CLASSES below
       (delete the rest to save space and training time).
    3. Point --raw_dir at that folder and run this script.

  You can override which classes to keep with the KEEP_CLASSES list.

USAGE
    python data_prep.py --raw_dir path/to/raw_dataset --out_dir data
    # optional: --max_per_class 500  to cap images per class for speed
"""

import argparse
import os
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# The 6 classes we keep (matches diagnosis_map.py). These are exactly the
# PlantVillage folder names. Edit this list if you want different classes.
# ---------------------------------------------------------------------------
KEEP_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
]

# Image file extensions we treat as valid images.
VALID_EXTS = {".jpg", ".jpeg", ".png"}


def gather_images(class_dir: Path):
    """Return a list of image file paths inside a class directory."""
    return [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]


def split_indices(n, train_frac, val_frac):
    """Compute how many items go to train/val given total count n.

    The remainder goes to test, so fractions always sum to 1.
    """
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    # test gets whatever is left (avoids rounding errors dropping images)
    return n_train, n_val


def prepare(raw_dir, out_dir, train_frac, val_frac, max_per_class, seed):
    """Main routine: split each class and copy files into train/val/test."""
    random.seed(seed)  # reproducible splits

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset folder not found: {raw_dir}\n"
            "Download PlantVillage first (see the docstring at the top)."
        )

    # Create the three top-level split folders.
    for split in ("train", "val", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    summary = []  # collect counts for a final printed report

    for class_name in KEEP_CLASSES:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            print(f"[WARN] Class folder missing, skipping: {class_dir}")
            continue

        images = gather_images(class_dir)
        random.shuffle(images)  # shuffle before splitting

        # Optionally cap the number of images per class to keep training fast.
        if max_per_class is not None:
            images = images[:max_per_class]

        n = len(images)
        if n == 0:
            print(f"[WARN] No images found in {class_dir}, skipping.")
            continue

        n_train, n_val = split_indices(n, train_frac, val_frac)
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        # Copy each subset into data/<split>/<class>/
        for split, subset in (("train", train_imgs),
                              ("val", val_imgs),
                              ("test", test_imgs)):
            dest_dir = out_dir / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in subset:
                shutil.copy2(src, dest_dir / src.name)

        summary.append((class_name, n, len(train_imgs),
                        len(val_imgs), len(test_imgs)))
        print(f"[OK] {class_name:28s} total={n:4d} "
              f"train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")

    # ------------------------------------------------------------------
    # Print a tidy summary table so you can confirm the split looks right.
    # ------------------------------------------------------------------
    print("\n================ SPLIT SUMMARY ================")
    print(f"{'Class':28s} {'Total':>6s} {'Train':>6s} {'Val':>5s} {'Test':>5s}")
    tot = [0, 0, 0, 0]
    for name, total, tr, va, te in summary:
        print(f"{name:28s} {total:6d} {tr:6d} {va:5d} {te:5d}")
        tot[0] += total; tot[1] += tr; tot[2] += va; tot[3] += te
    print("-" * 47)
    print(f"{'TOTAL':28s} {tot[0]:6d} {tot[1]:6d} {tot[2]:5d} {tot[3]:5d}")
    print(f"\nDone. Prepared data written to: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a class-folder image dataset into train/val/test.")
    parser.add_argument("--raw_dir", required=True,
                        help="Path to the raw dataset (one folder per class).")
    parser.add_argument("--out_dir", default="data",
                        help="Output folder for the split dataset (default: data).")
    parser.add_argument("--train_frac", type=float, default=0.8,
                        help="Fraction of images for training (default: 0.8).")
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="Fraction of images for validation (default: 0.1). "
                            "Test gets the remainder.")
    parser.add_argument("--max_per_class", type=int, default=500,
                        help="Cap images per class to keep training fast "
                            "(default: 500). Use 0 for no cap.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits (default: 42).")
    args = parser.parse_args()

    max_per_class = None if args.max_per_class in (0, None) else args.max_per_class

    prepare(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        max_per_class=max_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
