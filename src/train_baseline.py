import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from load_data import create_training_datasets

IMAGE_SHAPE = (48, 48, 1)
NUMBER_OF_CLASSES = 7
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 4
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIRECTORY = PROJECT_ROOT / "models"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"

MODELS_DIRECTORY.mkdir(exist_ok=True)
RESULTS_DIRECTORY.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(RANDOM_SEED)

#allow tensorflow to acqurire gpu gradually in here as well
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

train_dataset, validation_dataset = create_training_datasets()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

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

best_model_path = MODELS_DIRECTORY / "baseline_best.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
]

print("\nTraining the baseline CNN")
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, callbacks=callbacks)

validation_loss, validation_accuracy = model.evaluate(validation_dataset, verbose=0)
print(f"\nbest validation loss: {validation_loss:.4f}")
print(f"best validation accuracy: {validation_accuracy:.4f}")
print(f"best model saved to: {best_model_path}")

#Start drawing the stuff using matplotlib to check for accuracy
completed_epochs = range(1, len(history.history["loss"]) + 1)

figure, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(completed_epochs, history.history["accuracy"], label="Training accuracy")
axes[0].plot(completed_epochs, history.history["val_accuracy"], label="Validation accuracy")
axes[0].set_title("Baseline CNN accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(completed_epochs, history.history["loss"], label="Training loss")
axes[1].plot(completed_epochs, history.history["val_loss"], label="Validation loss")
axes[1].set_title("Baseline CNN loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Cross-entropy loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

figure.tight_layout()

curves_path = RESULTS_DIRECTORY / "baseline_training_curves.png"
figure.savefig(curves_path, dpi=150, bbox_inches="tight")
plt.close(figure)
print(f"Training curves saved to: {curves_path}")

