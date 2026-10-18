import * as tf from "@tensorflow/tfjs";
import {load as loadFaceDetector} from "@tensorflow-models/face-detection/dist/tfjs/detector.js";

const video = document.querySelector("#webcam");
const startButton = document.querySelector("#start-camera");
const stopButton = document.querySelector("#stop-camera");
const cameraStatus = document.querySelector("#camera-status");
const overlay = document.querySelector("#face-overlay");
const overlayContext = overlay.getContext("2d");

let stream = null;
let detector = null;
let detectionBusy = false;
let detectionTimer = null;

function stopCamera() {
    window.clearInterval(detectionTimer);
    detectionTimer = null;
    overlayContext.clearRect(0, 0, overlay.width, overlay.height);
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
    }

    video.srcObject = null;
    startButton.disabled = false;
    stopButton.disabled = true;
    cameraStatus.textContent = "Camera is off.";
}

async function startCamera() {
    startButton.disabled = true;
    cameraStatus.textContent = "Waiting for camera access...";

    try {
        if (!detector) {
            cameraStatus.textContent = "Loading the face detector...";
            await tf.ready();

            detector = await loadFaceDetector(
                {
                    runtime: "tfjs",
                    modelType: "short",
                    maxFaces: 1,
                }
            );
        }

        cameraStatus.textContent = "Waiting for camera access...";
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640},
                height: { ideal:480 },
                facingMode: "user",
            },
            audio: false,
        });
        video.srcObject = stream;
        await video.play();
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        detectionTimer = window.setInterval(detectFace, 100);

        stopButton.disabled = false;
        cameraStatus.textContent = "Camera is on.";
    } catch (error)  {
        stopCamera();
        console.error(error);

        if (error.name === "NotAllowedError") {
            cameraStatus.textContent = "Camera permission was denied you twit. Allow the camera and try again.";
        } else if (error.name === "NotFoundError") {
            cameraStatus.textContent = "No camera was found. Bro go get a cam.";
        } else {
            cameraStatus.textContent = `Camera failed: ${error.message}`;
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
            return;
        }

        const box = faces[0].box;
        overlayContext.strokeStyle = "#00ff88";
        overlayContext.lineWidth = 3;
        overlayContext.strokeRect(box.xMin, box.yMin, box.width, box.height);
        cameraStatus.textContent = "Face detected.";
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

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("pagehide", stopCamera);