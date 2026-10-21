
import random
import shutil
from pathlib import Path

# Original dataset location
SOURCE_DIR = Path(r"C:\Users\DELL\Desktop\Dataset")

# Balanced dataset will be created inside our project
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "dataset" / "balanced"

# Number of images per class
IMAGES_PER_CLASS = 2000

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

random.seed(42)


def get_images(folder):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    return sorted(
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    classes = {
        "real": SOURCE_DIR / "Train" / "Real",
        "fake": SOURCE_DIR / "Train" / "Fake",
    }

    selected_images = {}

    # Check both source folders before copying anything
    for label, folder in classes.items():
        images = get_images(folder)
        print(f"{label.upper()} images found: {len(images)}")

        if len(images) < IMAGES_PER_CLASS:
            raise ValueError(
                f"Not enough {label} images. "
                f"Need {IMAGES_PER_CLASS}, found {len(images)}."
            )

        selected_images[label] = random.sample(
            images, IMAGES_PER_CLASS
        )

    # Avoid accidentally mixing with an earlier generated dataset
    for label in classes:
        destination = OUTPUT_DIR / label
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(
                f"{destination} is not empty. "
                "Rename or move this folder before running again."
            )

    # Copy selected images; originals remain unchanged
    for label, images in selected_images.items():
        destination = OUTPUT_DIR / label
        destination.mkdir(parents=True, exist_ok=True)

        for index, source_file in enumerate(images, start=1):
            target_file = destination / (
                f"{label}_{index:05d}{source_file.suffix.lower()}"
            )
            shutil.copy2(source_file, target_file)

        print(f"Copied {len(images)} images to: {destination}")

    print("\nBalanced training dataset created successfully.")
    print(f"Location: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
