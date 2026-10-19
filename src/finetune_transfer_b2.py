from pathlib import Path
import tensorflow as tf
from collections import Counter
from load_data import create_expanded_training_datasets

tf.keras.utils.set_random_seed(42)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WARMUP_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "transfer_b2_warmup_best.keras"
)

model = tf.keras.models.load_model(
    WARMUP_MODEL_PATH,
    compile=False,
)

backbone = model.get_layer("efficientnetb2")
backbone.trainable = True

for layer in backbone.layers:
    layer.trainable = not isinstance(
        layer,
        tf.keras.layers.BatchNormalization,
    )

model.summary()

train_dataset, validation_dataset = create_expanded_training_datasets()

class_names = train_dataset.class_names
class_counts = Counter(Path(path).parent.name for path in train_dataset.file_paths)
total_images = len(train_dataset.file_paths)
class_weights = {
    index: (
        total_images / (len(class_names) * class_counts[name])
    ) ** 0.5
    for index, name in enumerate(class_names)
}
for index, name in enumerate(class_names):
    print(f"{name:>8}: {class_counts[name]} images, "
          f"weight={class_weights[index]:.3f}")

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomRotation(factor=0.02, fill_mode="reflect"),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="reflect"),
])


def augment_batch(images, labels):
    images = data_augmentation(images, training=True)
    labels = tf.one_hot(labels, depth=len(class_names))
    return images, labels

def prepare_validation_batch(images, labels):
    labels = tf.one_hot(labels, depth=len(class_names))
    return images, labels


train_dataset = train_dataset.map(
    augment_batch,
    num_parallel_calls=tf.data.AUTOTUNE,
)

validation_dataset = validation_dataset.map(
    prepare_validation_batch,
    num_parallel_calls=tf.data.AUTOTUNE,
)

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

BEST_MODEL_PATH = PROJECT_ROOT / "models" / "transfer_b2_best.keras"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
RESULTS_DIRECTORY.mkdir(exist_ok=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
    metrics=["accuracy"],
)


callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.0000001,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(str(RESULTS_DIRECTORY / "transfer_b2_finetune_history.csv")),
]
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=40,
    callbacks=callbacks,
    class_weight=None,
)

print("Best model saved to:", BEST_MODEL_PATH)
