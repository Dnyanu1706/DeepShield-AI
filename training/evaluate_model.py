
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(r"C:\Users\DELL\Desktop\Dataset\Test")
MODEL_PATH = PROJECT_DIR / "models" / "image_model.keras"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Folder names are sorted alphabetically:
    # Fake = 0, Real = 1
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="binary",
        seed=SEED,
    )

    class_names = test_ds.class_names
    print("Class mapping:", dict(enumerate(class_names)))

    model = tf.keras.models.load_model(MODEL_PATH)

    true_labels = np.concatenate([
        labels.numpy().reshape(-1)
        for _, labels in test_ds
    ]).astype(int)

    scores = model.predict(test_ds, verbose=1).reshape(-1)

    # With alphabetical class order, score >= 0.5 means Real.
    predicted_labels = (scores >= 0.5).astype(int)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(
        true_labels, predicted_labels, zero_division=0
    )
    recall = recall_score(
        true_labels, predicted_labels, zero_division=0
    )
    f1 = f1_score(
        true_labels, predicted_labels, zero_division=0
    )

    # Rows = actual class; columns = predicted class
    matrix = confusion_matrix(true_labels, predicted_labels)

    print("\n--- Test Results ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(matrix)

    print("\nDetailed classification report:")
    print(classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0,
    ))

    results = {
        "class_mapping": {str(i): name for i, name in enumerate(class_names)},
        "test_images": int(len(true_labels)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist(),
    }

    output_path = PROJECT_DIR / "models" / "evaluation_metrics.json"
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path}")
    print(
        "These results evaluate this dataset only; "
        "they do not guarantee performance on new images."
    )


if __name__ == "__main__":
    main()
