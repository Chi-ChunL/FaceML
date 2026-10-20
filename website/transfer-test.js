import * as tf from "@tensorflow/tfjs";

const status = document.querySelector("#status");

async function main() {
    await tf.ready();

    const baseURL = `${import.meta.env.BASE_URL}models/`;

    status.textContent = "Loading B0 and B2...";

    const b0Model = await tf.loadGraphModel(`${baseURL}b0/model.json`);
    const b2Model = await tf.loadGraphModel(`${baseURL}b2/model.json`);

    const input = tf.fill([1, 48, 48, 1], 127);

    try {
        status.textContent = "Warming up and measuring...";

        const warmupRuns = 5;
        const measuredRuns = 31;
        const times = [];
        let probabilities;
        let baselineTensors;

        for (let i = 0; i < warmupRuns + measuredRuns; i++) {
            const started = performance.now();
            const prediction = tf.tidy(() => {
                const b0probabilities = b0Model.predict(input);
                const b2probabilities = b2Model.predict(input);

                return b0probabilities.add(b2probabilities).mul(0.5);
            });

            try {
                probabilities = Array.from(await prediction.data());

                if (i >= warmupRuns) {
                    times.push(performance.now() - started);
                }
            } finally {
                prediction.dispose();
            }

            if (i === warmupRuns - 1) {
                baselineTensors = tf.memory().numTensors;
            }

            await tf.nextFrame();
        }

        times.sort((a, b) => a - b);

        status.textContent = JSON.stringify({
            backend: tf.getBackend(),
            probabilitySum: probabilities.reduce((a, b) => a + b, 0),
            finiteProbabilities: probabilities.every(Number.isFinite),
            medianMs: times[Math.floor(times.length / 2)].toFixed(1),
            p95Ms: times[Math.ceil(times.length * 0.95) - 1].toFixed(1),
            tensorsAfterWarmup: baselineTensors,
            tensorsAfterTesting: tf.memory().numTensors,
        }, null, 2);
    } finally {
        input.dispose();
        b0Model.dispose();
        b2Model.dispose();
    }
}

main().catch(error => {
    status.textContent = `Test failed: ${error.message}`;
    console.error(error);
});