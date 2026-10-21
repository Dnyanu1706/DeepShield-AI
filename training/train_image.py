
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Dataset"
TRAIN_DIR = DATA_DIR / "Train"
TEST_DIR = DATA_DIR / "Test"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "image_model.keras"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

# Training settings
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 15
SEED = 42


def main():
    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Check dataset folders
    for folder in [
        TRAIN_DIR / "fake",
        TRAIN_DIR / "real",
        TEST_DIR / "fake",
        TEST_DIR / "real",
    ]:
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

    # Load training images and reserve 20% for validation
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        class_names=["fake", "real"],
    )

    val_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        class_names=["fake", "real"],
    )

    # Keep the test set separate until final evaluation
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="binary",
        class_names=["fake", "real"],
    )

    print("Class order:", train_ds.class_names)
    print("Training, validation and test datasets loaded.")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    # CNN model
    model = keras.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Restore and save the best validation model
    best_model = keras.models.load_model(MODEL_PATH)

    # Evaluate only after training has finished
    test_results = best_model.evaluate(
        test_ds,
        return_dict=True,
        verbose=1,
    )

    metrics = {
        "training_history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "test_results": {
            key: float(value)
            for key, value in test_results.items()
        },
        "class_names": ["Fake", "Real"],
        "test_image_count": 50,
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nFinal test results:")
    for name, value in test_results.items():
        print(f"{name}: {value:.4f}")

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
