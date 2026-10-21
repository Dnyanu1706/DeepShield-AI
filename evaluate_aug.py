
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, f1_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "image_model_mobilenet_aug.keras"
TEST_DIR = BASE_DIR / "Dataset" / "Test"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=["fake", "real"],
    label_mode="binary",
    shuffle=False,
)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in test_ds
]).astype(int)

scores = model.predict(test_ds, verbose=0).reshape(-1)
y_pred = (scores >= 0.5).astype(int)

print("\nClass mapping: 0 = Fake, 1 = Real")
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

print("Macro F1-score:", round(
    f1_score(y_true, y_pred, average="macro"), 4
))
