import tensorflow as tf
from pathlib import Path

tf.keras.utils.set_random_seed(42)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

backbone = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

backbone.trainable = False

print("Input shape:", backbone.input_shape)
print("Feature map shape:", backbone.output_shape)
print("Parameters:", backbone.count_params())

inputs = tf.keras.Input(shape=(48, 48, 1), name="face_image")

x = tf.keras.layers.Resizing(224, 224)(inputs)
x = tf.keras.layers.Concatenate(axis=-1)([x,x,x])

x = backbone(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    units=7,
    activation="softmax",
    name="expression_probabilities",
)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="transfer_cnn")

model.summary()

sample = tf.fill((1, 48, 48, 1), 127.0)
predictions = model(sample, training=False)

print("Predicition shape:", predictions.shape)
print("Probability sum:", tf.reduce_sum(predictions).numpy())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "transfer_probe_saved_model"
)

EXPORT_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)

model.export(
    EXPORT_DIRECTORY,
    input_signature=[
        tf.TensorSpec(
            shape=(None, 48, 48, 1),
            dtype=tf.float32,
            name="face_image",
        )
    ],
    verbose=False,
)

print("Prototype exported to:", EXPORT_DIRECTORY)