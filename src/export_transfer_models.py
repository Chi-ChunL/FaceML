import tensorflow as tf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILES = {
    "b0": "transfer_full_expanded_best.keras",
    "b2": "transfer_b2_best.keras",
}

for name, filename in MODEL_FILES.items():
    model_path = PROJECT_ROOT / "models" / filename

    model = tf.keras.models.load_model(model_path, compile=False)
    export_path = PROJECT_ROOT / "models" / "checkpoints" / f"{name}_saved_model"
    model.export(
        export_path,
        format="tf_saved_model",
        input_signature=[tf.TensorSpec(shape=(None, 48, 48, 1), dtype=tf.float32, name="face_image")],
        verbose=False,
    )

    print(f"Exported {name}: {export_path}")
    del model
    tf.keras.backend.clear_session()