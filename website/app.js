import * as tf from "@tensorflow/tfjs";
import { stopCamera, renderProbabilities, isCurrentMediaRequest } from "./camera.js";
import { classNames, loadExpressionModel, predictExpression } from "./model.js";
import "./cursor.js";
import "./style.css";
import "./flight.js";

const status = document.querySelector("#status");
const imageUpload = document.querySelector("#image-upload");
const uploadButton = document.querySelector("#upload-button");
const preview = document.querySelector("#preview");

let model;

async function main() {
    try {
        await tf.ready();
        model = await loadExpressionModel();
        model.summary();
        const probabilities = tf.tidy(() => {
            const input = tf.zeros([1, 48, 48, 1]);
            const output = model.predict(input);

            return Array.from(output.dataSync());
        });
        const probabilitySum = probabilities.reduce((total, probability) => total + probability, 0);
        console.info("Model ready.", {
            inputShape: model.inputs[0].shape,
            outputShape: model.outputs[0].shape,
            probabilitySum,
        });
        status.textContent = "";
        imageUpload.disabled = false;
        uploadButton.disabled = false;
        document.querySelector("#start-camera").disabled = false;
        } catch (error) {
        console.error(error);
        status.textContent = `Model test failed:\n${error.message}`;
    }
}
imageUpload.addEventListener("change", async () => {
    const file = imageUpload.files[0];
    if (!file || !model) return;

    const imageRequest = stopCamera();
    imageUpload.disabled = true;
    uploadButton.disabled = true;
    status.textContent = "Predicting..";

    const imageUrl = URL.createObjectURL(file);

    try {
        preview.src = imageUrl;
        await preview.decode();
        if (!isCurrentMediaRequest(imageRequest)) {
            status.textContent = "";
            return;
        }
        document.querySelector("#webcam").hidden = true;
        document.querySelector("#face-overlay").hidden = true;
        preview.hidden = false;
        document.querySelector("#camera-status").textContent = "Showing uploaded image.";

        const canvas = document.createElement("canvas");
        canvas.width = 48;
        canvas.height = 48;

        const context = canvas.getContext("2d");
        context.drawImage(preview, 0, 0, 48, 48);

        const probabilities = predictExpression(canvas);

        renderProbabilities(probabilities);
        status.textContent = "";
    } catch (error) {
        if (!isCurrentMediaRequest(imageRequest)) {
            status.textContent = "";
            return;
        }
        console.error(error);
        status.textContent = `Prediction failed:\n${error.message}`;
    } finally {
        URL.revokeObjectURL(imageUrl);
        imageUpload.disabled = false;
        uploadButton.disabled = false;
        imageUpload.value = "";
    }
});

main();

window.addEventListener("load", () => {
    if (!window.location.hash) {
        document.querySelector("#city").scrollIntoView({
            behavior: "instant",
            block: "start",
        });
    }
});

const faceMLDemo = document.querySelector("#faceml-demo");
document.querySelector("#leave-planet").addEventListener("click", () => {
    faceMLDemo.hidden = true;
    window.dispatchEvent(new Event("spaceml-exited"));
})

window.addEventListener("spaceml-landed", () => {
    faceMLDemo.hidden = false;
    faceMLDemo.tabIndex = -1;
    faceMLDemo.focus({ preventScroll: true });

    faceMLDemo.scrollIntoView({
        behavior: "smooth",
        block: "start",
    });
});

window.addEventListener("keydown", event => {
    if(
        event.code !== "Escape" ||
        event.repeat ||
        event.defaultPrevented ||
        faceMLDemo.hidden
    ) return;
    event.preventDefault()
    faceMLDemo.hidden = true;

    window.dispatchEvent(new Event("spaceml-exited"));
});

uploadButton.addEventListener("click", () => {
    imageUpload.click();
});
