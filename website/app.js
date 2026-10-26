import * as tf from "@tensorflow/tfjs";
import "./camera.js";
import { classNames, loadExpressionModel, predictExpression } from "./model.js";
import "./cursor.js";
import "./style.css";
import "./flight.js";

const status = document.querySelector("#status");
const imageUpload = document.querySelector("#image-upload");
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
        status.textContent = [
        "Model loaded and prediction completed!",
        `Input shape: ${JSON.stringify(model.inputs[0].shape)}`,
        `Output shape: ${JSON.stringify(model.outputs[0].shape)}`,
        `Number of probabilities: ${probabilities.length}`,
        `Probability sum: ${probabilitySum.toFixed(4)}`,
        ].join("\n");
        imageUpload.disabled = false;
        document.querySelector("#start-camera").disabled = false;
        } catch (error) {
        console.error(error);
        status.textContent = `Model test failed:\n${error.message}`;
    }
}
imageUpload.addEventListener("change", async () => {
    const file = imageUpload.files[0];
    if (!file || !model) return;

    imageUpload.disabled = true;
    status.textContent = "Predicting..";

    const imageUrl = URL.createObjectURL(file);

    try {
        preview.src = imageUrl;
        await preview.decode();
        preview.hidden = false;

        const canvas = document.createElement("canvas");
        canvas.width = 48;
        canvas.height = 48;

        const context = canvas.getContext("2d");
        context.drawImage(preview, 0, 0, 48, 48);

        const probabilities = predictExpression(canvas);

        const bestIndex = probabilities.indexOf(Math.max(...probabilities));
        const probabilitySum = probabilities.reduce((total, probability) => total + probability, 0);
        status.textContent = [`Prediction: ${(classNames[bestIndex])} ` + `${(probabilities[bestIndex] * 100).toFixed(2)}%)`,
        "",
        ...classNames.map((name, index) => 
            `${name}: ${(probabilities[index] * 100).toFixed(2)}%`,
        ),
        "",
        `Probability sum: ${probabilitySum.toFixed(4)}`,
    ].join("\n");
    } catch (error) {
        console.error(error);
        status.textContent = `Prediction failed:\n${error.message}`;
    } finally {
        URL.revokeObjectURL(imageUrl);
        imageUpload.disabled = false;
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

const faceMLDemo = document.querySelector("#faceml-demo")

window.addEventListener("spaceml-landed", () => {
    faceMLDemo.hidden = false;
    faceMLDemo.tabIndex = -1;
    faceMLDemo.focus({ preventScroll: true });

    faceMLDemo.scrollIntoView({
        behavior: "smooth",
        block: "start",
    });
});