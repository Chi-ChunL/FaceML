import tensorflow as tf
from pathlib import Path
from load_data import create_training_datasets

tf.keras.utils.set_random_seed(42)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

backbone = tf.keras.applications.EfficientNetB2(
    include_top=False,
    weights="imagenet",
    input_shape=(260, 260, 3),
)

backbone.trainable = False

print("Input shape:", backbone.input_shape)
print("Feature map shape:", backbone.output_shape)
print("Parameters:", backbone.count_params())

inputs = tf.keras.Input(shape=(48, 48, 1), name="face_image")

x = tf.keras.layers.Resizing(260, 260)(inputs)
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

WARMUP_MODEL_PATH = (PROJECT_ROOT / "models" / "checkpoints" / "transfer_b2_warmup_best.keras")
RESULT_DIRECTORY = PROJECT_ROOT / "results"

WARMUP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULT_DIRECTORY.mkdir(exist_ok=True)

train_dataset, validation_dataset = create_training_datasets()

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(tf.data.AUTOTUNE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

callbacks = [tf.keras.callbacks.ModelCheckpoint(
    filepath=WARMUP_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(str(RESULT_DIRECTORY / "transfer_b2_warmup_history.csv")),
]


history = model.fit(train_dataset, validation_data=validation_dataset, epochs=5, callbacks=callbacks)

print("best warmup model saved to:", WARMUP_MODEL_PATH)