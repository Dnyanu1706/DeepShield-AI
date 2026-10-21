
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, f1_score

BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR / "Dataset" / "Test"
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8

MODELS = {
    "Original CNN": (
        BASE_DIR / "models" / "image_model.keras",
        (128, 128),
    ),
    "MobileNetV2": (
        BASE_DIR / "models" / "image_model_mobilenet.keras",
        (160, 160),
    ),
    "MobileNet Augmented": (
        BASE_DIR / "models" / "image_model_mobilenet_aug.keras",
        (160, 160),
    ),
    "MobileNet Fine-tuned": (
        BASE_DIR / "models" / "image_model_mobilenet_finetune.keras",
        (160, 160),
    ),
}


def main():
    print("Class mapping: 0 = Fake, 1 = Real")

    for model_name, (model_path, image_size) in MODELS.items():
        if not model_path.exists():
            print(f"\nSkipping {model_name}: model file not found")
            print(f"Expected: {model_path}")
            continue

        test_ds = tf.keras.utils.image_dataset_from_directory(
            TEST_DIR,
            image_size=image_size,
            batch_size=BATCH_SIZE,
            class_names=["fake", "real"],
            label_mode="binary",
            shuffle=False,
        )

        model = tf.keras.models.load_model(model_path)

        y_true = np.concatenate([
            labels.numpy().reshape(-1)
            for _, labels in test_ds
        ]).astype(int)

        scores = model.predict(test_ds, verbose=0).reshape(-1)
        y_pred = (scores >= 0.5).astype(int)

        print(f"\n{'=' * 50}")
        print(model_name)
        print(f"{'=' * 50}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

        print("\nClassification Report:")
        print(classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Fake", "Real"],
            zero_division=0,
        ))

        print(
            "Macro F1:",
            round(f1_score(y_true, y_pred, average="macro"), 4),
        )


if __name__ == "__main__":
    main()
