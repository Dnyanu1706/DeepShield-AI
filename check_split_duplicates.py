
from pathlib import Path
import hashlib
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"

hash_locations = defaultdict(list)

for split in ["Train", "Test"]:
    for label in ["fake", "real"]:
        folder = DATASET_DIR / split / label

        if not folder.exists():
            print("Folder missing:", folder)
            continue

        for path in folder.iterdir():
            if not path.is_file():
                continue

            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hash_locations[digest].append(
                    (split, label, path.name)
                )
            except OSError as error:
                print("Could not read:", path, error)

cross_split_duplicates = []
cross_label_duplicates = []

for locations in hash_locations.values():
    splits = {item[0] for item in locations}
    labels = {item[1] for item in locations}

    if len(splits) > 1:
        cross_split_duplicates.append(locations)

    if len(labels) > 1:
        cross_label_duplicates.append(locations)

print("\n===== Dataset Split Check =====")
print("Unique image hashes:", len(hash_locations))
print("Duplicates across Train and Test:",
      len(cross_split_duplicates))
print("Same exact image in Fake and Real:",
      len(cross_label_duplicates))

if cross_split_duplicates:
    print("\nTrain-Test duplicate details:")
    for group in cross_split_duplicates:
        print(group)

if cross_label_duplicates:
    print("\nFake-Real duplicate details:")
    for group in cross_label_duplicates:
        print(group)

print("\nCheck completed.")
