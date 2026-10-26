const ship = document.querySelector("#spaceship");
const launchButton = document.querySelector("#launch-flight");
const city = document.querySelector("#city");

const keys = new Set();
const movementKeys = ["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowLeft", "ArrowDown", "ArrowRight"];
const speed = 240;

let x = (document.documentElement.clientWidth - ship.offsetWidth) / 2;
let y = city.offsetTop + city.offsetHeight - ship.offsetHeight - 24;
let previousTime = performance.now()

launchButton.addEventListener("click", () => {
    ship.style.visibility = "visible";
    ship.focus({ preventScroll: true });
});

ship.addEventListener("keydown", event => {
    if (event.code == "Escape") {
        keys.clear();
        launchButton.focus({preventScroll: true});
        return;
    }

    if (movementKeys.includes(event.code)) {
        event.preventDefault();
        keys.add(event.code);
    }
});

ship.addEventListener("keyup", event => {
    keys.delete(event.code);
});

ship.addEventListener("blur", () => keys.clear());
window.addEventListener("blur", () => keys.clear());

function updateFlight(time) {
    const deltaTime = Math.min((time - previousTime) / 1000, 0.05);
    previousTime = time;

    const right = keys.has("KeyD") || keys.has("ArrowRight");
    const left = keys.has("KeyA") || keys.has("ArrowLeft");
    const down = keys.has("KeyS") || keys.has("ArrowDown");
    const up = keys.has("KeyW") || keys.has("ArrowUp");

    const dx = Number(right) - Number(left);
    const dy = Number(down) - Number(up);
    const length = Math.hypot(dx, dy);

    if (length > 0) {
        x += (dx / length) * speed * deltaTime;
        y += (dy / length) * speed * deltaTime;
    }

    const maxX = document.documentElement.clientWidth - ship.offsetWidth;
    const worldHeight = document.body.clientHeight;
    const maxY = worldHeight - ship.offsetHeight;

    x = Math.max(0, Math.min(x, maxX));
    y = Math.max(0, Math.min(y, maxY));

    ship.style.left = `${x}px`;
    ship.style.top = `${y}px`;

    if (document.activeElement === ship) {
        const shipCenterY = y + ship.offsetHeight / 2;
        const upperLimit = window.innerHeight * 0.25;
        const lowerLimit = window.innerHeight * 0.75;

        const screenY = shipCenterY - window.scrollY;
        let targetScroll = window.scrollY;

        if (screenY < upperLimit) {
            targetScroll = shipCenterY - upperLimit;
        } else if (screenY > lowerLimit){
            targetScroll = shipCenterY - lowerLimit;
        }

        const maxScroll = Math.max(0, worldHeight - window.innerHeight);
        targetScroll = Math.max(0, Math.min(targetScroll, maxScroll));

        if (targetScroll !== window.scrollY) {
            window.scrollTo({
                top: targetScroll,
                behavior: "instant",
            });
        }
    }

    requestAnimationFrame(updateFlight);
}

requestAnimationFrame(updateFlight);