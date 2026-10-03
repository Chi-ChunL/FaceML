import argparse
from pathlib import Path
import tensorflow as tf
from load_data import EXPECTED_CLASS_NAMES
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "improved_best.keras"
#parser thro
parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=Path)
args = parser.parse_args()

if not args.image_path.is_file():
    parser.error(f"Image not found: {args.image_path}")

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
image = tf.keras.utils.load_img(args.image_path, color_mode="grayscale", target_size=(48, 48), interpolation="bilinear")
image_array = tf.keras.utils.img_to_array(image)
image_batch = tf.expand_dims(image_array, axis=0)

probabilities = model.predict(image_batch, verbose=0)[0]
predicted_index = int(probabilities.argmax())
print("Input shape:", image_batch.shape)
print("\nExpression probabilities")
for class_name, probability in zip(EXPECTED_CLASS_NAMES, probabilities):
    print(f"{class_name:>8}: {probability * 100:6.2f}%")

print(f"\nPrediction: {EXPECTED_CLASS_NAMES[predicted_index]} "
    f"({probabilities[predicted_index] * 100:.2f}%)")
print(f"Probability sum: {probabilities.sum():.4f}")

figure, axis = plt.subplots(figsize=(5, 5))
axis.imshow(image, cmap="gray", vmin=0, vmax=255)
axis.set_title(f"Predicition: {EXPECTED_CLASS_NAMES[predicted_index]}\n"
               f"Model confidence: {probabilities[predicted_index] * 100:.2f}%")
axis.axis("off")

output_path = PROJECT_ROOT / "results"/ f"prediction_{args.image_path.stem}.png"

figure.tight_layout()
figure.savefig(output_path, dpi=150)
plt.close(figure)

print(f"Predicition image saved to: {output_path}")