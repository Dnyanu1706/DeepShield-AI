
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_DIR = Path(__file__).resolve().parent
TEST_DIR = PROJECT_DIR / "Dataset" / "Test"
MODEL_PATH = PROJECT_DIR / "models" / "image_model_mobilenet_finetune_v2.keras"

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

y_true = []
y_scores = []

for images, labels in test_ds:
    scores = model.predict(images, verbose=0).ravel()
    y_true.extend(labels.numpy().astype(int))
    y_scores.extend(scores)

y_true = tf.constant(y_true).numpy()
y_scores = tf.constant(y_scores).numpy()
y_pred = (y_scores >= 0.5).astype(int)

print("\n===== Confusion Matrix =====")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\n===== Classification Report =====")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        digits=4,
        zero_division=0,
    )
)
