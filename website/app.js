import * as tf from "@tensorflow/tfjs";
import { stopCamera, renderProbabilities, isCurrentMediaRequest } from "./camera.js";
import { classNames, loadExpressionModel, predictExpression } from "./model.js";
import "./cursor.js";
import "./flight.js";

const status = document.querySelector("#status");
const imageUpload = document.querySelector("#image-upload");
const uploadButton = document.querySelector("#upload-button");
const preview = document.querySelector("#preview");
const certificateButton = document.querySelector("#claim-certificate");
const certificateCanvas = document.querySelector("#certificate-preview");
const certificateContext = certificateCanvas.getContext("2d");
const certificateStatus = document.querySelector("#certificate-status");
const capturedPortrait = document.querySelector("#astronaut-portrait");
const certificateBackground = new Image();
const certificateName = document.querySelector("#certificate-name");
const downloadCertificateButton = document.querySelector("#download-certificate");
certificateBackground.src = new URL("./assets/mini-deco/certificate.png", import.meta.url).href;

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

function wait(milliseconds) {
    return new Promise(resolve => {
        window.setTimeout(resolve, milliseconds);
    });
}

async function playIntro() {
    const loadingScreen = document.querySelector("#loading-screen");
    const loadingContent = document.querySelector("#loading-content");
    const nameSplash = document.querySelector("#name-splash");
    loadingScreen.classList.add("is-playing");
    await wait(2100);

    loadingContent.hidden = true;
    nameSplash.hidden = false;
    await wait(1500);

    document.body.classList.remove("is-loading");

    if (!window.location.hash) {
        document.querySelector("#city").scrollIntoView({
            behavior: "instant",
            block: "start",
        });
    }
    const fade = loadingScreen.animate([{ opacity: 1 }, { opacity: 0 }], {
        duration: 700,
        easing: "ease-in-out",
        fill: "forwards",
    });
    
    await fade.finished;
    loadingScreen.remove();
}

playIntro();

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

certificateButton.addEventListener("click", async () => {
    if (capturedPortrait.hidden) {
        certificateStatus.textContent = "Take your portrait in SpaceML first.";
        return;
    }

    certificateButton.disabled = true;
    downloadCertificateButton.disabled = true;

    try {
        await certificateBackground.decode();
        await document.fonts.ready;

        certificateContext.clearRect(0, 0, 768, 512);
        certificateContext.imageSmoothingEnabled = false;
        certificateContext.drawImage(certificateBackground, 0, 0, 768, 512);
        certificateContext.fillStyle = "#FCF7EF";
        certificateContext.font = "32px Pixelify Sans";
        certificateContext.textAlign = "center";
        certificateContext.textBaseline = "middle";
        const explorerName = certificateName.value.trim() || "Space Explorer";

        certificateContext.fillText(explorerName, 384, 292, 600);
        certificateContext.drawImage(capturedPortrait, 32, 352, 128, 128);
        certificateCanvas.hidden = false;
        downloadCertificateButton.disabled = false;
        certificateStatus.textContent = "Certified by StarWeb.";
    } catch (error) {
        console.error(error);
        certificateStatus.textContent = "The certificate couldnt be created.";
    } finally {
        certificateButton.disabled = false;
    }
});

certificateName.addEventListener("input", () => {
    downloadCertificateButton.disabled = true;
});

downloadCertificateButton.addEventListener("click", () => {
    if (certificateCanvas.hidden || downloadCertificateButton.disabled) return;
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = certificateCanvas.width * 2;
    exportCanvas.height = certificateCanvas.height * 2;

    const exportContext = exportCanvas.getContext("2d");
    exportContext.imageSmoothingEnabled = false;
    exportContext.drawImage(certificateCanvas, 0, 0, exportCanvas.width, exportCanvas.height);
    exportCanvas.toBlob(blob => {
        if(!blob) {
            certificateStatus.textContent = "PNG  couldnt be created"; 
            return;
        }

        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "StarWeb-certificate.png";
        link.hidden = true;

        document.body.append(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, "image/png");
});
