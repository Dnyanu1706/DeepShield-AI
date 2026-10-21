
from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError
import hashlib

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"

for split in ["Train", "Test"]:
    print(f"\n===== {split} Dataset =====")

    for label in ["fake", "real"]:
        folder = DATASET_DIR / split / label
        files = [
            p for p in folder.iterdir()
            if p.is_file()
        ] if folder.exists() else []

        sizes = Counter()
        valid_images = 0
        invalid_images = []
        hashes = {}

        for path in files:
            try:
                with Image.open(path) as img:
                    img.verify()

                with Image.open(path) as img:
                    sizes[img.size] += 1

                valid_images += 1

                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes.setdefault(digest, []).append(path.name)

            except (OSError, UnidentifiedImageError):
                invalid_images.append(path.name)

        duplicates = [
            names for names in hashes.values()
            if len(names) > 1
        ]

        print(f"\nClass: {label}")
        print(f"Total files: {len(files)}")
        print(f"Valid images: {valid_images}")
        print(f"Image dimensions: {dict(sizes)}")
        print(f"Invalid images: {invalid_images}")
        print(f"Exact duplicates in folder: {duplicates}")

print("\nDataset check completed.")
