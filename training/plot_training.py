
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = PROJECT_DIR / "models" / "training_metrics.json"
REPORTS_DIR = PROJECT_DIR / "reports"


def main():
    if not METRICS_PATH.exists():
        print(f"Training metrics file not found: {METRICS_PATH}")
        return

    # Load saved training history
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        history = json.load(file)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Training and validation accuracy graph
    plt.figure(figsize=(8, 5))
    plt.plot(history["accuracy"], label="Training Accuracy")

    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="Validation Accuracy")

    plt.title("Model Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    accuracy_path = REPORTS_DIR / "training_accuracy.png"
    plt.savefig(accuracy_path, dpi=300)
    plt.close()

    # 2. Training and validation loss graph
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="Training Loss")

    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation Loss")

    plt.title("Model Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_path = REPORTS_DIR / "training_loss.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()

    print("Training graphs created successfully!")
    print(f"Accuracy graph: {accuracy_path}")
    print(f"Loss graph: {loss_path}")


if __name__ == "__main__":
    main()
