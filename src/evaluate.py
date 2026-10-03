from pathlib import Path
import tensorflow as tf
from load_data import create_test_dataset
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "improved"
MODEL_PATH = PROJECT_ROOT / "models" / f"{EXPERIMENT_NAME}_best.keras"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
RESULTS_DIRECTORY.mkdir(exist_ok=True)

def main():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find model at: {MODEL_PATH}")

    test_dataset = create_test_dataset()
    class_names = test_dataset.class_names

    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    print(f"\nLoading the {EXPERIMENT_NAME} model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Class names:", class_names)
    print("evaluating...")

    test_loss, test_accuracy = model.evaluate(test_dataset)

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test accuracy percentage: {test_accuracy * 100:.2f}%")

    print("\nGenerating predictions...")

    prediction_probabilities = model.predict(test_dataset)
    predicted_labels = np.argmax(prediction_probabilities, axis=1)
    true_labels = np.concatenate([labels.numpy() for images, labels in test_dataset])
    report = classification_report(true_labels, predicted_labels, target_names=class_names, digits=4, zero_division=0)
    print("\nClassification report:")
    print(report)

    report_data = classification_report(true_labels, predicted_labels, target_names=class_names, digits=4, zero_division=0, output_dict=True)

    raw_confusion_matrix = confusion_matrix(true_labels, predicted_labels)
    normalised_confusion_matrix = confusion_matrix(true_labels, predicted_labels, normalize="true")

    metrics = {
        "model": EXPERIMENT_NAME,
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "parameter_count": model.count_params(),
        "model_size_bytes": MODEL_PATH.stat().st_size,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_f1": float(report_data["macro avg"]["f1-score"]),
        "weighted_f1": float(report_data["weighted avg"]["f1-score"]),
        "class_names": class_names,
        "classification_report": report_data,
        "confusion_matrix_counts": raw_confusion_matrix.tolist(),
        "confusion_matrix_normalised": normalised_confusion_matrix.tolist(),
    }

    metrics_path = RESULTS_DIRECTORY / f"{EXPERIMENT_NAME}_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
    print(f"{EXPERIMENT_NAME.capitalize()} metrics saved to: {metrics_path}")

    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    raw_display = ConfusionMatrixDisplay(confusion_matrix=raw_confusion_matrix, display_labels=class_names)
    raw_display.plot(ax=axes[0], cmap="Blues", values_format="d", colorbar=False)
    axes[0].set_title(f"{EXPERIMENT_NAME.capitalize()} confusion matrix - counts")
    normalised_display = ConfusionMatrixDisplay(confusion_matrix=normalised_confusion_matrix, display_labels=class_names)
    normalised_display.plot(ax=axes[1], cmap="Blues", values_format=".2f", colorbar=False)
    axes[1].set_title(f"{EXPERIMENT_NAME.capitalize()} confusion matrix - row-normalised")
    for axis in axes:
        plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    figure.tight_layout()
    confusion_matrix_path = RESULTS_DIRECTORY / f"{EXPERIMENT_NAME}_confusion_matrices.png"
    figure.savefig(confusion_matrix_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"\nConfusion matrices saved to: "
          f"{confusion_matrix_path}")

if __name__ == "__main__":
    main()

