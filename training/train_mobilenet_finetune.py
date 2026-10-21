
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

MODEL_PATH = MODEL_DIR / "image_model_mobilenet_finetune_v2.keras"
METRICS_PATH = MODEL_DIR / "mobilenet_finetune_metrics_v2.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
HEAD_EPOCHS = 12
FINE_TUNE_EPOCHS = 8
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
    class_weights = {
        0: 570 / (2 * 350),  # Fake
        1: 570 / (2 * 220),  # Real
    }

    print("Class weights:", class_weights)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

    metrics = [
        "accuracy",
        keras.metrics.AUC(name="auc"),
    ]

    # Stage 1: train the classification head
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=metrics,
    )

    head_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    print("\n===== Stage 1: Frozen MobileNetV2 =====")
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        class_weight=class_weights,
        callbacks=head_callbacks,
    )

    # Stage 2: unfreeze only the last 20 layers.
    # Keep BatchNormalization layers frozen for stability.
    base_model.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )

    fine_callbacks = [
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

    print("\n===== Stage 2: Fine-Tuning =====")
    fine_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=fine_callbacks,
    )

    # Save the final fine-tuning model if no checkpoint was created.
    if not MODEL_PATH.exists():
        model.save(MODEL_PATH)

    best_model = keras.models.load_model(MODEL_PATH)

    print("\n===== Test Evaluation =====")
    test_results = best_model.evaluate(
        test_ds,
        return_dict=True,
        verbose=1,
    )

    output = {
        "class_names": ["fake", "real"],
        "image_size": list(IMAGE_SIZE),
        "head_history": {
            key: [float(v) for v in values]
            for key, values in head_history.history.items()
        },
        "fine_tuning_history": {
            key: [float(v) for v in values]
            for key, values in fine_history.history.items()
        },
        "test_results": {
            key: float(value)
            for key, value in test_results.items()
        },
    }

    METRICS_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    for name, value in test_results.items():
        print(f"{name}: {value:.4f}")

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
