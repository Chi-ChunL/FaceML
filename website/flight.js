const ship = document.querySelector("#spaceship");
const launchButton = document.querySelector("#launch-flight");
const city = document.querySelector("#city");
const planet = document.querySelector("#spaceml-planet");
const demo = document.querySelector("#faceml-demo");
const planetPrompt = document.querySelector("#planet-prompt");

const keys = new Set();
const movementKeys = ["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowLeft", "ArrowDown", "ArrowRight"];
const speed = 400;
const thrust = 200;
const drag = 0.1;
const bounce = 0.6;
const turnSpeed = 150;
const hull = [
    [0, -64],          
    [19, -42], [19, -12],   
    [32, 8], [32, 22],     
    [0, 24],               
    [-32, 22], [-32, 8],    
    [-19, -12], [-19, -42], 
];

let canEnterPlanet = false;
let angle = 0;
let vx = 0;
let vy = 0;


let x = (document.documentElement.clientWidth - ship.offsetWidth) / 2;
let y = city.offsetTop + city.offsetHeight - ship.offsetHeight - 24;
let previousTime = performance.now()

launchButton.addEventListener("click", () => {
    if (!demo.hidden) return;
    ship.style.visibility = "visible";
    ship.focus({ preventScroll: true });
});

ship.addEventListener("keydown", event => {
    if (event.ctrlKey || event.repeat) return;
    
    if (event.code === "KeyF") {
        event.preventDefault();

        if (!canEnterPlanet || event.repeat) return;
        canEnterPlanet = false;
        planetPrompt.hidden = true;
        vx = 0;
        vy = 0;
        keys.clear()

        ship.classList.remove("is-thrusting");
        ship.style.visibility = "hidden";
        ship.blur();

        window.dispatchEvent(new Event("spaceml-landed"));
        return;
    }
    if (event.code == "Escape") {
        event.preventDefault();
        
        keys.clear();
        vx = 0;
        vy = 0;
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

    vx += sin * throttle * thrust * deltaTime;
    vy -= cos * throttle * thrust * deltaTime;

    const down = keys.has("KeyS") || keys.has("ArrowDown")
    if (down) {
        const brake = Math.exp(-3 * deltaTime);
        vx *= brake;
        vy *= brake;
    }
    const damping = Math.exp(-drag * deltaTime)
    vx *= damping;
    vy *= damping;

    const currentSpeed = Math.hypot(vx, vy);
    if (currentSpeed > speed) {
        vx *= speed / currentSpeed;
        vy *= speed / currentSpeed;
    }

    x += vx * deltaTime;
    y += vy * deltaTime;

    const width = ship.offsetWidth;
    const height = ship.offsetHeight;
    const worldHeight = document.body.clientHeight;

    const rotatedWidth = Math.abs(cos) * width + Math.abs(sin) * height;
    const rotatedHeight = Math.abs(sin) * width + Math.abs(cos) * height;

    const paddingX = Math.max(0, (rotatedWidth - width) / 2);
    const paddingY = Math.max(0, (rotatedHeight - height) / 2);
    
    const maxX = document.documentElement.clientWidth - width - paddingX;
    const maxY = worldHeight - height - paddingY;

    if (x < paddingX) {
        x= paddingX;
        vx = Math.abs(vx) * bounce;
    } else if (x > maxX) {
        x = maxX;
        vx = -Math.abs(vx) * bounce;
    }
    if (y < paddingY) {
        y = paddingY;
        vy = Math.abs(vy) * bounce;
    } else if (y > maxY) {
        y = maxY;
        vy = -Math.abs(vy) * bounce;
    }

    ship.style.left = `${x}px`;
    ship.style.top = `${y}px`;
    ship.style.transform = `rotate(${angle}deg)`

    canEnterPlanet = document.activeElement === ship && demo.hidden && touchesPlanet(width, height, sin, cos)
    planetPrompt.hidden = !canEnterPlanet;

    if (document.activeElement === ship) {
        const shipCenterY = y + ship.offsetHeight / 2;
        const upperLimit = window.innerHeight * 0.30;
        const lowerLimit = window.innerHeight * 0.65;

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

window.addEventListener("spaceml-exited", () =>{
    keys.clear();
    vx = 0;
    vy = 0; 
    ship.classList.remove("is-thrusting");
    ship.style.visibility = "visible";

    ship.scrollIntoView({
        behavior: "instant",
        block: "center",
    });

    ship.focus({ preventScroll: true })
});

