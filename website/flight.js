const ship = document.querySelector("#spaceship");
const launchButton = document.querySelector("#launch-flight");
const city = document.querySelector("#city");
const planet = document.querySelector("#spaceml-planet");
const demo = document.querySelector("#faceml-demo");

const keys = new Set();
const movementKeys = ["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowLeft", "ArrowDown", "ArrowRight"];
const speed = 240;
const turnSpeed = 180;
let angle = 0;

let x = (document.documentElement.clientWidth - ship.offsetWidth) / 2;
let y = city.offsetTop + city.offsetHeight - ship.offsetHeight - 24;
let previousTime = performance.now()

launchButton.addEventListener("click", () => {
    if (!demo.hidden) return;
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

function touchesPlanet(width, height, sin, cos) {
    const bounds = planet.getBoundingClientRect();
    const planetX = bounds.left + bounds.width / 2 + window.scrollX;
    const planetY = bounds.top + bounds.height / 2 + window.scrollY;
    const dx = planetX - (x + width / 2);
    const dy = planetY - (y + height / 2);
    const localX = dx * cos + dy * sin;
    const localY = -dx * sin + dy * cos;
    const hullTop = -height / 2;
    const hullBottom = hullTop + height * (44 / 64);
    const closestX = Math.max(-width / 2, Math.min(localX, width/2));
    const closestY = Math.max(hullTop, Math.min(localY, hullBottom));
    const distance = Math.hypot(localX - closestX, localY - closestY);
    return distance <= bounds.width / 2;

}

function updateFlight(time) {
    const deltaTime = Math.min((time - previousTime) / 1000, 0.05);
    previousTime = time;

    const right = keys.has("KeyD") || keys.has("ArrowRight");
    const left = keys.has("KeyA") || keys.has("ArrowLeft");
    const up = keys.has("KeyW") || keys.has("ArrowUp");

    const turn = Number(right) - Number(left)
    const throttle = Number(up);
    ship.classList.toggle("is-thrusting", up);

    angle += turn * turnSpeed * deltaTime;

    const radians = angle * Math.PI / 180;
    const sin = Math.sin(radians);
    const cos = Math.cos(radians);

    x += sin * throttle * speed * deltaTime;
    y -= cos * throttle * speed * deltaTime;

    const width = ship.offsetWidth;
    const height = ship.offsetHeight;
    const worldHeight = document.body.clientHeight;

    const rotatedWidth = Math.abs(cos) * width + Math.abs(sin) * height;
    const rotatedHeight = Math.abs(sin) * width + Math.abs(cos) * height;

    const paddingX = Math.max(0, (rotatedWidth - width) / 2);
    const paddingY = Math.max(0, (rotatedHeight - height) / 2);
    
    const maxX = document.documentElement.clientWidth - width - paddingX;
    const maxY = worldHeight - height - paddingY;

    x = Math.max(paddingX, Math.min(x, maxX));
    y = Math.max(paddingY, Math.min(y, maxY));

    ship.style.left = `${x}px`;
    ship.style.top = `${y}px`;
    ship.style.transform = `rotate(${angle}deg)`

    if (document.activeElement === ship && demo.hidden && touchesPlanet(width, height, sin, cos)) {
        keys.clear();
        ship.classList.remove("is-thrusting");
        ship.style.visibility = "hidden";
        ship.blur();

        window.dispatchEvent(new Event("spaceml-landed"));
    }

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