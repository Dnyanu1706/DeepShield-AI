
from pathlib import Path
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "Dataset" / "Test"
MODEL_PATH = BASE_DIR / "models" / "image_model_mobilenet_finetune.keras"

IMAGE_SIZE = (160, 160)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=1,
    class_names=["fake", "real"],
    shuffle=False,
)

model = tf.keras.models.load_model(MODEL_PATH)

file_paths = test_ds.file_paths
correct = 0

print("\n===== Prediction Inspection =====")
print("0 = Fake, 1 = Real")

for index, (images, labels) in enumerate(test_ds):
    score = float(model.predict(images, verbose=0)[0][0])
    actual = int(labels.numpy().reshape(-1)[0])
    predicted = 1 if score >= 0.5 else 0

    actual_name = "Real" if actual == 1 else "Fake"
    predicted_name = "Real" if predicted == 1 else "Fake"
    confidence = score if predicted == 1 else 1 - score

    is_correct = actual == predicted
    correct += int(is_correct)

    status = "CORRECT" if is_correct else "WRONG"

    print(
        f"{Path(file_paths[index]).name} | "
        f"Actual: {actual_name} | "
        f"Predicted: {predicted_name} | "
        f"Real score: {score:.3f} | "
        f"Confidence: {confidence:.3f} | {status}"
    )

print(f"\nCorrect: {correct}/50")
print(f"Accuracy: {correct / len(file_paths) * 100:.2f}%")

