import tensorflow as tf
from pathlib import Path

IMAGE_SHAPE = (48, 48, 1)
NUMBER_OF_CLASSES = 7
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve.parents[1]
TRAIN_DIRECTORY = PROJECT_ROOT / "data" / "fer2013" / "train"

#allow tensorflow to acqurire gpu gradually in here as well

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#sets up the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=IMAGE_SHAPE, name="face_image"),
        tf.keras.layers.Rescaling(scale=1.0 / 255, name="normalise_pixels"),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", name="convolution_1"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pooling_1"),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", name="convolution_2"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pooling_2"),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", name="convolution_3"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pooling_3"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(units=128, activation="relu", name="dense"),
        tf.keras.layers.Dropout(rate=0.5, name="dropout"),
        tf.keras.layers.Dense(units=NUMBER_OF_CLASSES, activation="softmax", name="expression_probabilities")
    ],
    name="baseline_cnn",
)

#compiles the actual model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.summary()