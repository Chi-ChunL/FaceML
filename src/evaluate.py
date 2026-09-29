from pathlib import Path
import tensorflow as tf
from load_data import create_test_dataset
import numpy as np
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_best.keras"

def main():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not found model at: {MODEL_PATH}")

    test_dataset = create_test_dataset()
    class_names = test_dataset.class_names

    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    print("\nLoading the baseline model...")
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

if __name__ == "__main__":
    main()

