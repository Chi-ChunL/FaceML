from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

#defining my constant so its easier to link to it
IMAGE_SIZE = (48,48)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
RESULTS_DIRECTORY.mkdir(exist_ok=True)
DATA_DIRECTORY = PROJECT_ROOT / "data" / "fer2013"
TRAIN_DIRECTORY = DATA_DIRECTORY / "train"
TEST_DIRECTORY = DATA_DIRECTORY / "test"

EXPECTED_CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

train_dataset = tf.keras.utils.image_dataset_from_directory(TRAIN_DIRECTORY, validation_split=VALIDATION_SPLIT, subset="training", seed=RANDOM_SEED, image_size=IMAGE_SIZE, color_mode="grayscale", batch_size=BATCH_SIZE, label_mode="int")
validation_dataset = tf.keras.utils.image_dataset_from_directory(TRAIN_DIRECTORY, validation_split=VALIDATION_SPLIT, subset="validation", seed=RANDOM_SEED, image_size=IMAGE_SIZE, color_mode="grayscale", batch_size=BATCH_SIZE, label_mode="int")
test_dataset = tf.keras.utils.image_dataset_from_directory(TEST_DIRECTORY, image_size=IMAGE_SIZE, color_mode="grayscale", batch_size=BATCH_SIZE, label_mode="int", shuffle=False)

class_names = train_dataset.class_names

print("\nClass names:", class_names)

if class_names != EXPECTED_CLASS_NAMES:
    raise ValueError(f"this class is unexpected. Expected {EXPECTED_CLASS_NAMES}," 
                     f"but found {class_names}")

images, labels = next(iter(train_dataset))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Image data type:", images.dtype)
print("Minimum pixel value:", tf.reduce_min(images).numpy())
print("Maximum pixel value:", tf.reduce_max(images).numpy())
print("First batch of labels:", labels.numpy())

#create nine plotting areas
figure, axes = plt.subplots(3, 3, figsize=(8, 8))

for index, axis in enumerate(axes.flat):
    image = images[index].numpy().squeeze()
    label_index = int(labels[index].numpy())
    label_name = class_names[label_index]

    axis.imshow(image, cmap="gray", vmin=0, vmax=255)

    axis.set_title(label_name.capitalize())
    axis.axis("off")

figure.suptitle("FER-2013 training exmaples", fontsize=16)
plt.tight_layout()
output_path = RESULTS_DIRECTORY / "training_examples.png"

figure.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close(figure)
print(f"example grid is saved to: {output_path}")
