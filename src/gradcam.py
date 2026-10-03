"""Save a Grad-CAM explanation for one cropped face image."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from load_data import EXPECTED_CLASS_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "improved_best.keras"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "gradcam"


def make_gradcam(image_batch, model, target_index=None):
    """Return a normalised heatmap and the seven expression probabilities."""
    output_layer = model.layers[-1]
    last_convolution = model.get_layer("last_convolution")

    with tf.GradientTape() as tape:
        # Trace a single forward pass through this Sequential model. This avoids
        # disconnected symbolic outputs in some saved Keras Sequential models.
        dense_inputs = image_batch
        for layer in model.layers[:-1]:
            dense_inputs = layer(dense_inputs, training=False)
            if layer is last_convolution:
                feature_maps = dense_inputs
                tape.watch(feature_maps)

        # Use scores before softmax to avoid its saturation at high confidence.
        # This reuses the output layer's weights without changing the model.
        logits = tf.linalg.matmul(dense_inputs, output_layer.kernel)
        if output_layer.use_bias:
            logits = logits + output_layer.bias

        if target_index is None:
            target_index = int(tf.argmax(logits[0]).numpy())
        target_score = logits[:, target_index]

    gradients = tape.gradient(target_score, feature_maps)
    if gradients is None:
        raise ValueError("The class score is not connected to the feature maps.")

    # Average spatial gradients to obtain one importance weight per feature map.
    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * channel_weights, axis=-1)

    # Keep positive contributions and scale to 0-1, including an all-zero case.
    heatmap = tf.nn.relu(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    probabilities = tf.nn.softmax(logits, axis=-1)[0]
    return heatmap.numpy(), probabilities.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path, help="Path to a cropped face")
    parser.add_argument(
        "--class-name",
        choices=EXPECTED_CLASS_NAMES,
        help="Explain this class instead of the predicted class",
    )
    parser.add_argument("--output", type=Path, help="Optional output PNG path")
    args = parser.parse_args()

    if not args.image_path.is_file():
        parser.error(f"Image not found: {args.image_path}")
    if not MODEL_PATH.is_file():
        parser.error(f"Model not found: {MODEL_PATH}")

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    image = tf.keras.utils.load_img(
        args.image_path,
        color_mode="grayscale",
        target_size=(48, 48),
        interpolation="bilinear",
    )
    image_array = tf.keras.utils.img_to_array(image)
    image_batch = tf.expand_dims(image_array, axis=0)
    # Keep pixels in 0-255: the saved model already contains rescaling.

    target_index = None
    if args.class_name is not None:
        target_index = EXPECTED_CLASS_NAMES.index(args.class_name)
    heatmap, probabilities = make_gradcam(image_batch, model, target_index)
    predicted_index = int(probabilities.argmax())
    if target_index is None:
        target_index = predicted_index

    predicted_name = EXPECTED_CLASS_NAMES[predicted_index]
    target_name = EXPECTED_CLASS_NAMES[target_index]
    print("Input shape:", image_batch.shape)
    print("Grad-CAM heatmap shape:", heatmap.shape)
    print("\nExpression probabilities:")
    for class_name, probability in zip(EXPECTED_CLASS_NAMES, probabilities):
        print(f"{class_name:>8}: {probability * 100:6.2f}%")
    print(f"\nPrediction: {predicted_name} ({probabilities[predicted_index]:.2%})")
    print(f"Explaining class: {target_name}")
    print(f"Probability sum: {probabilities.sum():.4f}")
    if heatmap.max() == 0:
        print("No positive Grad-CAM contributions; the heatmap is all zero.")

    resized_heatmap = tf.image.resize(
        heatmap[..., None], (48, 48), method="bilinear"
    ).numpy()[..., 0]

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    axes[0].imshow(image_array[..., 0], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Model input: 48 x 48")
    heatmap_plot = axes[1].imshow(heatmap, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM: 12 x 12")
    axes[2].imshow(image_array[..., 0], cmap="gray", vmin=0, vmax=255)
    axes[2].imshow(resized_heatmap, cmap="magma", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Heatmap overlay")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(heatmap_plot, ax=axes[1:], label="Relative positive contribution")
    figure.suptitle(
        f"Prediction: {predicted_name} ({probabilities[predicted_index]:.2%})"
        f" | Grad-CAM target: {target_name}",
        fontsize=12,
    )

    output_path = args.output or RESULTS_DIRECTORY / f"{args.image_path.stem}_{target_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Grad-CAM image saved to: {output_path}")


if __name__ == "__main__":
    main()
