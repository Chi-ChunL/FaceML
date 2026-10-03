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

#define training augmentation
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomRotation(factor=0.02, fill_mode="reflect"),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="reflect"),
    ],
    name="data_augmentation",
)

train_dataset, validation_dataset = create_training_datasets()

class_names = train_dataset.class_names
class_counts = {index: 0 for index in range(NUMBER_OF_CLASSES)}
for image_path in train_dataset.file_paths:
    class_name = Path(image_path).parent.name
    class_index = class_names.index(class_name)
    class_counts[class_index] += 1
total_images = sum(class_counts.values())
class_weights = {index: total_images / (NUMBER_OF_CLASSES * count) for index, count in class_counts.items()}
print("\nTraining class weights:")
for index, class_name in enumerate(class_names):
    print(
        f"{class_name:>8}: "
        f"count={class_counts[index]}, "
        f"weight={class_weights[index]:.3f}"
    )


def augment_batch(images, labels):
    augmented_images = data_augmentation(images, training=True)
    return augmented_images, labels
train_dataset = train_dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#sets up the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=IMAGE_SHAPE, name="face_image"),
        tf.keras.layers.Rescaling(scale=1.0 / 255, name="normalise_pixels"),
        #block 1 learn 32 feature maps
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(rate=0.2, name="dropout_block1"),
        #block 2 learn 64 feature maps
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(rate=0.3, name="dropout_block2"),
        #block 3 learn 128 feature maps
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", use_bias=False, name="last_convolution"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(rate=0.4, name="dropout_block3"),
        # combine all the features and classify the expression
        tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling"),
        tf.keras.layers.Dense(units=128, activation="relu", name="dense"),
        tf.keras.layers.Dropout(rate=0.5, name="dropout_dense"),
        tf.keras.layers.Dense(units=NUMBER_OF_CLASSES, activation="softmax", name="expression_probabilities")
    ],
    name="improved_cnn",
)

#compiles the actual model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.summary()

best_model_path = MODELS_DIRECTORY / "improved_weighted_best.keras"

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

print("\nTraining the improved CNN")
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, callbacks=callbacks, class_weight=class_weights)

validation_loss, validation_accuracy = model.evaluate(validation_dataset, verbose=0)
print(f"\nbest validation loss: {validation_loss:.4f}")
print(f"best validation accuracy: {validation_accuracy:.4f}")
print(f"best model saved to: {best_model_path}")

#Start drawing the stuff using matplotlib to check for accuracy
completed_epochs = range(1, len(history.history["loss"]) + 1)

figure, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(completed_epochs, history.history["accuracy"], label="Training accuracy")
axes[0].plot(completed_epochs, history.history["val_accuracy"], label="Validation accuracy")
axes[0].set_title("Improved CNN accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(completed_epochs, history.history["loss"], label="Training loss")
axes[1].plot(completed_epochs, history.history["val_loss"], label="Validation loss")
axes[1].set_title("Improved CNN loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Cross-entropy loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

figure.tight_layout()

curves_path = RESULTS_DIRECTORY / "improved_weighted_training_curves.png"
figure.savefig(curves_path, dpi=150, bbox_inches="tight")
plt.close(figure)
print(f"Training curves saved to: {curves_path}")

