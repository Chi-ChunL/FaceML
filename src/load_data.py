from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

#acquire only allocated gpu memory and not blow up my computer
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#defining my constant so its easier to link to it
IMAGE_SIZE = (48, 48)
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

figure.suptitle("FER-2013 training examples", fontsize=16)
plt.tight_layout(rect=(0, 0, 1, 0.96))
output_path = RESULTS_DIRECTORY / "training_examples.png"

figure.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close(figure)
print(f"example grid is saved to: {output_path}")

train_class_counts = {}
test_class_counts = {}

for class_name in class_names:
    train_class_directory = TRAIN_DIRECTORY / class_name
    test_class_directory = TEST_DIRECTORY / class_name

    train_class_counts[class_name] = sum(1 for path in train_class_directory.iterdir() if path.is_file())
    test_class_counts[class_name] = sum(1 for path in test_class_directory.iterdir() if path.is_file())

print("\nImages per class:")

for class_name in class_names:
    print(
        f"{class_name:>8}: "
        f"train={train_class_counts[class_name]:>4}, "
        f"test={test_class_counts[class_name]:>4}"
    )
training_counts = [train_class_counts[class_name] for class_name in class_names]
distribution_figure, distribution_axis = plt.subplots(figsize=(9, 5))
bars = distribution_axis.bar(class_names, training_counts, color="steelblue")

distribution_axis.bar_label(bars, padding=3)
distribution_axis.set_title("FER-2013 training images per class")
distribution_axis.set_xlabel("Expression class")
distribution_axis.set_ylabel("Number of images")
distribution_axis.set_ylim(0, max(training_counts) *1.12)

distribution_figure.tight_layout()
distribution_output_path = (RESULTS_DIRECTORY / "class_distribution.png")
distribution_figure.savefig(distribution_output_path, dpi=150, bbox_inches="tight")

plt.close(distribution_figure)

print(
    f"Class-distribution graph saved to: "
    f"{distribution_output_path}"
)

