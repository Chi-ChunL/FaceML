import * as tf from "@tensorflow/tfjs";
import {load as loadFaceDetector} from "@tensorflow-models/face-detection/dist/tfjs/detector.js";
import { classNames, predictExpression } from "./model";

const video = document.querySelector("#webcam");
const startButton = document.querySelector("#start-camera");
const stopButton = document.querySelector("#stop-camera");
const cameraStatus = document.querySelector("#camera-status");
const overlay = document.querySelector("#face-overlay");
const overlayContext = overlay.getContext("2d");
const faceCanvas = document.querySelector("#face-crop");
const faceContext = faceCanvas.getContext("2d");
const livePrediction = document.querySelector("#live-prediction");
const preview = document.querySelector("#preview");
const captureButton = document.querySelector("#capture-portrait");
const portrait = document.querySelector("#astronaut-portrait")
const portraitContext = portrait.getContext("2d");

const spacesuit = new Image();
spacesuit.src = new URL("./assets/mini-deco/spacesuit.png", import.meta.url).href;


const probabilityRows = classNames.map(name => {
    const row = document.createElement("div");
    row.className = "probability-row";

    row.innerHTML = 
        `<div class="probability-label">
            <span>${name}</span>
            <span class="probability-value">0%</span>
        </div>
        <div class="probability-track" aria-hidden="true">
            <div class="probability-fill"></div>
        </div>`;
        document.querySelector("#probability-bars").append(row);
        return {
            value: row.querySelector(".probability-value"),
            fill: row.querySelector(".probability-fill"),
        };
});

let stream = null;
let detector = null;
let detectionBusy = false;
let detectionTimer = null;
let cameraRequest = 0;
let cameraStarting = false;
let latestFaceCrop = null;

export function stopCamera() {
    latestFaceCrop = null;
    captureButton.disabled = true;
    const wasActive = Boolean(stream) || cameraStarting;
    cameraRequest += 1;
    faceContext.clearRect(0,0,48,48);
    renderProbabilities(null, "Camera is off.");
    window.clearInterval(detectionTimer);
    detectionTimer = null;
    overlayContext.clearRect(0, 0, overlay.width, overlay.height);
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
    }

    video.srcObject = null;
    preview.hidden = true;
    video.hidden = false;
    overlay.hidden = false;
    if (wasActive) {
        startButton.disabled = cameraStarting;
    }
    stopButton.disabled = true;
    cameraStatus.textContent = "Camera is off.";
    return cameraRequest;
}

export function isCurrentMediaRequest(request) {
    return request === cameraRequest;
}

async function startCamera() {
    if (cameraStarting || stream) return;
    
    preview.hidden = true;
    video.hidden = false;
    overlay.hidden = false;
    renderProbabilities(null, "Waiting for a face.");
    cameraStarting = true;
    const request = ++cameraRequest;
    startButton.disabled = true;

    try {
        if (!detector) {
            cameraStatus.textContent = "Loading the face detector...";
            await tf.ready();

            if (request !== cameraRequest) return;

            detector = await loadFaceDetector(
                {
                    runtime: "tfjs",
                    modelType: "short",
                    maxFaces: 1,
                }
            );
        }
        if (request !== cameraRequest) return;
        cameraStatus.textContent = "Waiting for camera access...";
        const nextStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640},
                height: { ideal:480 },
                facingMode: "user",
            },
            audio: false,
        });

        if (request !== cameraRequest) {
            nextStream.getTracks().forEach(track => track.stop());
            return;
        }
        stream = nextStream
        video.srcObject = stream;
        await video.play();

        if (request !== cameraRequest) return;
        
        video.width = video.videoWidth;
        video.height = video.videoHeight;

        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        detectionTimer = window.setInterval(detectFace, 100);

        stopButton.disabled = false;
        cameraStatus.textContent = "Camera is on.";
    } catch (error)  {
        if (request !== cameraRequest) return;
        stopCamera();
        console.error(error);

        if (error.name === "NotAllowedError") {
            cameraStatus.textContent = "Camera permission was denied you twit. Allow the camera and try again.";
        } else if (error.name === "NotFoundError") {
            cameraStatus.textContent = "No camera was found. Bro go get a cam.";
        } else {
            cameraStatus.textContent = `Camera failed: ${error.message}`;
        }
    } finally {
        cameraStarting = false;

        if (!stream) {
            startButton.disabled = false;
        }
    }
}

async function detectFace() {
    if (!stream || detectionBusy || video.readyState < 2) return;
    detectionBusy = true;
    const activeStream = stream;

    try {
        const faces = await detector.estimateFaces(video, {flipHorizontal: false});
        if (stream !== activeStream) return;

        overlayContext.clearRect(0, 0, overlay.width, overlay.height);
        if (faces.length === 0) {
            latestFaceCrop = null;
            captureButton.disabled = true;
            cameraStatus.textContent = "No face detected.";
            faceContext.clearRect(0,0,48,48);
            renderProbabilities(null, "No face detected.");
            return;
        }

        const box = faces[0].box;
        overlayContext.strokeStyle = "#00ff88";
        overlayContext.lineWidth = 3;
        overlayContext.strokeRect(box.xMin, box.yMin, box.width, box.height);
        cameraStatus.textContent = "Face detected.";
        predictDetectedFace(box);
    } catch (error) {
        if (stream === activeStream) {
            stopCamera();
            console.error(error);
            cameraStatus.textContent = `Face detection failed: ${error.message}`;
        }
    } finally {
        detectionBusy = false;
    }
}

function predictDetectedFace(box) {
    const side = Math.min(
        Math.max(box.width, box.height) * 1.15,
        video.videoWidth,
        video.videoHeight
    );

    const centerX = box.xMin + box.width /2;
    const centerY = box.yMin + box.height / 2;

    const x = Math.max(0, Math.min(centerX - side / 2, video.videoWidth - side));
    const y = Math.max(0, Math.min(centerY - side / 2, video.videoHeight - side));
    latestFaceCrop = { x, y, side };
    captureButton.disabled = !spacesuit.complete || spacesuit.naturalWidth === 0;
    faceContext.drawImage(
        video,
        x, y, side, side,
        0, 0, 48, 48
    );

    const probabilities = predictExpression(faceCanvas);
    renderProbabilities(probabilities);
}

export function renderProbabilities(probabilities = null, message = "Waiting for a face.") {
    probabilityRows.forEach((row, index) => {
        const percentage = (probabilities?.[index] ?? 0) * 100;
        row.value.textContent = `${percentage.toFixed(1)}%`;
        row.fill.style.width = `${percentage}%`
    });

    if (!probabilities) {
        livePrediction.textContent = message;
        return;
    }

    const bestIndex = probabilities.indexOf(Math.max(...probabilities));

    livePrediction.textContent = `${classNames[bestIndex]} - ` + `${(probabilities[bestIndex] * 100).toFixed(1)}%`;
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("pagehide", stopCamera);
window.addEventListener("spaceml-exited", stopCamera);
captureButton.addEventListener("click", () => {
    if (
        !latestFaceCrop || 
        !stream ||
        video.readyState < 2 ||
        !spacesuit.naturalWidth
    ) return;
    const { x, y, side } = latestFaceCrop;
    const cropHeight = side * 59 / 70;

    portraitContext.clearRect(0, 0 , 128, 128);
    portraitContext.save();
    portraitContext.translate(98, 37);
    portraitContext.scale(-1, 1);

    portraitContext.drawImage(
        video, x, y + (side - cropHeight) /2,
        side, cropHeight,
        0, 0 , 70, 59
    );

    portraitContext.restore()
    portraitContext.drawImage(spacesuit, 0, 0, 128, 128);
    portrait.hidden = false;
});