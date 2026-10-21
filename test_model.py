
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "image_model.keras"
TEST_DIR = BASE_DIR / "Dataset" / "Test"
IMAGE_SIZE = (128, 128)

model = tf.keras.models.load_model(MODEL_PATH)

# Training class mapping: fake = 0, real = 1
for label in ["fake", "real"]:
    folder = TEST_DIR / label
    images = []

    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        images.extend(folder.glob(pattern))

    print(f"\nActual class: {label} | Images: {len(images)}")

    for image_path in images[:5]:
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize(IMAGE_SIZE)
            array = np.asarray(img, dtype=np.float32)

        array = np.expand_dims(array, axis=0)
        score = float(model.predict(array, verbose=0)[0][0])

        predicted = "real" if score >= 0.5 else "fake"
        confidence = score * 100 if score >= 0.5 else (1 - score) * 100

        print(
            f"{image_path.name}: predicted={predicted}, "
            f"score={score:.4f}, confidence={confidence:.2f}%"
        )
