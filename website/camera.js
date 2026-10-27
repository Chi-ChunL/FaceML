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

let stream = null;
let detector = null;
let detectionBusy = false;
let detectionTimer = null;
let cameraRequest = 0;
let cameraStarting = false;

function stopCamera() {
    const wasActive = Boolean(stream) || cameraStarting;
    cameraRequest += 1;
    faceContext.clearRect(0,0,48,48);
    livePrediction.textContent = "No live Predicition";
    window.clearInterval(detectionTimer);
    detectionTimer = null;
    overlayContext.clearRect(0, 0, overlay.width, overlay.height);
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
    }

    video.srcObject = null;
    if (wasActive) {
        startButton.disabled = cameraStarting;
    }
    stopButton.disabled = true;
    cameraStatus.textContent = "Camera is off.";
}

async function startCamera() {
    if (cameraStarting || stream) return;

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
            cameraStatus.textContent = "No face detected.";
            faceContext.clearRect(0,0,48,48);
            livePrediction.textContent = "No face detected.";
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
    faceContext.drawImage(
        video,
        x, y, side, side,
        0, 0, 48, 48
    );

    const probabilities = predictExpression(faceCanvas);
    const bestIndex = probabilities.indexOf(Math.max(...probabilities));

    livePrediction.textContent = [
    `Prediction: ${classNames[bestIndex]} ` +
        `(${(probabilities[bestIndex] * 100).toFixed(2)}%)`,
    "",
    ...classNames.map((name, index) =>
        `${name}: ${(probabilities[index] * 100).toFixed(2)}%`
    ),
    "",
    `Probability sum: ${
        probabilities.reduce((total, value) => total + value, 0)
        .toFixed(4)
    }`,
    ].join("\n");
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("pagehide", stopCamera);
window.addEventListener("spaceml-exited", stopCamera);