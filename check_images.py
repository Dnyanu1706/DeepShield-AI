
from pathlib import Path
from collections import defaultdict
import hashlib

root = Path("Dataset")
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

hash_locations = defaultdict(list)

for split in ["Train", "Test"]:
    for label in ["fake", "real"]:
        folder = root / split / label

        for path in folder.iterdir():
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_locations[digest].append(
                f"{split}/{label}/{path.name}"
            )

duplicates = [
    locations
    for locations in hash_locations.values()
    if len(locations) > 1
]

print("Unique image files:", len(hash_locations))
print("Exact duplicate groups:", len(duplicates))

for locations in duplicates:
    print("DUPLICATE:", " <--> ".join(locations))
