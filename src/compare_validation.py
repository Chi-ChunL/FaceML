from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from load_data import create_training_datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

_, validation_dataset = create_training_datasets()
class_names = validation_dataset.class_names

validation_dataset = validation_dataset.cache()
true_labels = np.concatenate([labels.numpy() for images, labels in validation_dataset ])

model_files = ["baseline_best.keras", "baseline_augmented_best.keras", "improved_best.keras"]
for filename in model_files:
    model_path = PROJECT_ROOT / "models" / filename
    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(validation_dataset, verbose=0)
    probabilities = model.predict(validation_dataset, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    print(f"\nModel: {filename}")
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    print(classification_report(true_labels, predicted_labels, labels=list(range(len(class_names))), target_names=class_names, digits=4, zero_division=0))
del model
tf.keras.backend.clear_session()