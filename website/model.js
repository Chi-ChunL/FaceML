import * as tf from "@tensorflow/tfjs";

export const classNames = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"];

let expressionModel = null;

export async function loadExpressionModel() {
    await tf.ready()

    const modelURL = `${import.meta.env.BASE_URL}models/expression/model.json`;
    expressionModel = await tf.loadLayersModel(modelURL);
    return expressionModel;
}

export function predictExpression(canvas) {
    if (!expressionModel) {
        throw new Error("The expression model has not loaded.");
    }

    return tf.tidy(() => {
        const pixels = tf.browser.fromPixels(canvas).toFloat();
        const rgbWeights = tf.tensor1d([0.299, 0.587, 0.114]);

        const grayscale = pixels.mul(rgbWeights).sum(2);
        const input = grayscale.expandDims(-1).expandDims(0);
        
        return Array.from(expressionModel.predict(input).dataSync());
    });
}

