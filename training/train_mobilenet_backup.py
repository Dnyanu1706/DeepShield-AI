
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Dataset"
TRAIN_DIR = DATA_DIR / "Train"
TEST_DIR = DATA_DIR / "Test"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "image_model_mobilenet.keras"
METRICS_PATH = MODEL_DIR / "mobilenet_metrics.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 12
SEED = 42


def main():
    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for folder in [
        TRAIN_DIR / "fake",
        TRAIN_DIR / "real",
        TEST_DIR / "fake",
        TEST_DIR / "real",
    ]:
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=["fake", "real"],
        label_mode="binary",
    )

    val_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=["fake", "real"],
        label_mode="binary",
        shuffle=False,
    )

    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=["fake", "real"],
        label_mode="binary",
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    # Load pretrained MobileNetV2 feature extractor
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    # MobileNetV2 expects input pixels scaled to [-1, 1]
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

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

    best_model = keras.models.load_model(MODEL_PATH)
    test_results = best_model.evaluate(test_ds, return_dict=True, verbose=1)

    metrics = {
        "training_history": {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        },
        "test_results": {
            key: float(value)
            for key, value in test_results.items()
        },
        "class_names": ["fake", "real"],
        "test_image_count": 50,
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nMobileNetV2 test results:")
    for name, value in test_results.items():
        print(f"{name}: {value:.4f}")

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
