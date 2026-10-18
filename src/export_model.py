from pathlib import Path
import tensorflow as tf 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "improved_best.keras"
OUTPUT_PATH = PROJECT_ROOT / "models" / "improved_export.h5"

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.save(OUTPUT_PATH, include_optimizer=False)

print("Tensorflow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print(f"Exported model to: {OUTPUT_PATH}")