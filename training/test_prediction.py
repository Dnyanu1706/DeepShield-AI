
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "image_model.keras"
IMAGE_SIZE = (128, 128)


def predict_image(image_path):
    image_path = Path(image_path)

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize(IMAGE_SIZE)
            image_array = np.asarray(img, dtype=np.float32)

    except (OSError, UnidentifiedImageError) as error:
        print(f"Could not open image: {error}")
        return

    # Model includes Rescaling(1/255), so pass pixel values 0-255.
    image_array = np.expand_dims(image_array, axis=0)

    model = tf.keras.models.load_model(MODEL_PATH)
    score = float(model.predict(image_array, verbose=0)[0][0])

    # Dataset folders are alphabetically mapped:
    # fake = 0, real = 1
    # Therefore, higher score indicates the REAL class.
    predicted_class = "Real" if score >= 0.5 else "Fake"
    confidence = score if predicted_class == "Real" else 1.0 - score

    print(f"Prediction: {predicted_class}")
    print(f"Model score for Real: {score:.4f}")
    print(f"Model confidence for predicted class: {confidence * 100:.2f}%")
    print("\nNote: This is an experimental model output, not proof of authenticity.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            'Usage: python training\\test_prediction.py '
            '"C:\\path\\to\\image.jpg"'
        )
    else:
        predict_image(sys.argv[1])
