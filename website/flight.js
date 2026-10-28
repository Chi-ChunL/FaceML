const ship = document.querySelector("#spaceship");
const launchButton = document.querySelector("#launch-flight");
const city = document.querySelector("#city");
const planet = document.querySelector("#spaceml-planet");
const demo = document.querySelector("#faceml-demo");
const planetPrompt = document.querySelector("#planet-prompt");
const destinations = [
    {
        element: planet,
        name: "SpaceML",
        centerX: 0.5,
        centerY: 0.5,
        radius: 0.5,
    },
    {
        element: document.querySelector("#homeworld-planet"),
        name: "My homeworld",
        centerX: 45.5 / 96,
        centerY: 47 / 96,
        radius: 32.5 / 96,
    },
    {
        element: document.querySelector("#github-planet"),
        name: "GitHub",
        centerX: 46 / 96,
        centerY: 43 / 96,
        radius: 32 / 96,
    },
    {
        element: document.querySelector("#youtube-planet"),
        name: "YouTube",
        centerX: 48 / 96,
        centerY: 46 / 96,
        radius: 32 / 96,
    },
];

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

let activePlanet = null;
let angle = 0;
let vx = 0;
let vy = 0;
let launching = false;
let launchAnimation = null;
let pauseAfterLaunch = false;

let x = (document.documentElement.clientWidth - ship.offsetWidth) / 2;
let y = city.offsetTop + city.offsetHeight - ship.offsetHeight - 24;
let previousTime = performance.now()


launchButton.addEventListener("click", async () => {
    if (!demo.hidden || launching) return;

    launching = true;
    pauseAfterLaunch = false;
    launchButton.disabled = true;
    launchButton.textContent = "Relaunch rocket";

    keys.clear();
    vx = 0;
    vy = 0;
    angle = 0;
    activePlanet = null;
    planetPrompt.hidden = true;

    city.scrollIntoView({
        behavior: "instant",
        block: "start",
    });

    const screenLeft =
        (document.documentElement.clientWidth - ship.offsetWidth) / 2;

    const finishTop = Math.max(0, Math.min(
        window.innerHeight * 0.75 - ship.offsetHeight / 2,
        window.innerHeight - ship.offsetHeight - 24
    ));

    ship.style.position = "fixed";
    ship.style.left = `${screenLeft}px`;
    ship.style.top = `${finishTop}px`;
    ship.style.transform = "rotate(0deg)";
    ship.style.visibility = "visible";
    ship.classList.add("is-thrusting");
    ship.focus({ preventScroll: true });

    launchAnimation = ship.animate([
        {
            transform:
                `translateY(${window.innerHeight + 16 - finishTop}px)`,
        },
        { transform: "translateY(0)" },
    ], {
        duration: window.matchMedia(
            "(prefers-reduced-motion: reduce)"
        ).matches ? 1 : 1200,
        easing: "cubic-bezier(0.16, 1, 0.3, 1)",
    });

    await launchAnimation.finished;

    x = screenLeft + window.scrollX;
    y = finishTop + window.scrollY;

    ship.style.position = "absolute";
    ship.style.left = `${x}px`;
    ship.style.top = `${y}px`;
    ship.classList.remove("is-thrusting");

    launching = false;
    launchAnimation = null;
    launchButton.disabled = false;

    if (pauseAfterLaunch) {
        launchButton.focus({ preventScroll: true });
    } else {
        ship.focus({ preventScroll: true });
    }
});

ship.addEventListener("click", () => {
    if(!demo.hidden) return;
    ship.focus({ preventScroll: true });
});

ship.addEventListener("keydown", event => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    
    if (launching) {
        if (movementKeys.includes(event.code) || event.code == "Escape") {
            event.preventDefault();
        }

        if (event.code === "Escape") {
            pauseAfterLaunch = true;
            launchAnimation?.finish();
        }
        return;
    }

    if (event.code === "KeyF") {
        event.preventDefault();

        if (!activePlanet || event.repeat) return;
        const destinations = activePlanet.element;
        activePlanet = null;
        planetPrompt.hidden = true;
        destinations.focus({ preventScroll: true });
        destinations.click();
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

function enterPlanet() {
    if (!demo.hidden || launching) return;

    activePlanet = null;
    planetPrompt.hidden = true;
    keys.clear();
    vx = 0;
    vy = 0;

    ship.classList.remove("is-thrusting");
    ship.style.visibility = "hidden";
    ship.blur();

    window.dispatchEvent(new Event("spaceml-landed"));
}

planet.addEventListener("click", enterPlanet);
document.querySelectorAll(".planet-entry").forEach(entry => {
    entry.addEventListener("click", event => {
        if (launching) {
            event.preventDefault();
            return;
        }

        keys.clear();
        vx = 0;
        vy = 0;
        ship.classList.remove("is-thrusting");
    });
});
function touchesPlanet(destination, width, height, sin, cos) {
    const image = destination.element.querySelector("img");
    const bounds = (image || destination.element).getBoundingClientRect();
    const planetX = bounds.left + bounds.width * destination.centerX + window.scrollX;
    const planetY = bounds.top + bounds.height * destination.centerY + window.scrollY;
    const radius = bounds.width * destination.radius;

    const dx = planetX - (x + width / 2);
    const dy = planetY - (y + height / 2);
    const localX = dx * cos + dy * sin;
    const localY = -dx * sin + dy * cos;

    for (let i = 0; i < hull.length; i++) {
        const [ax, ay] = hull[i];
        const [bx, by] = hull[(i + 1) % hull.length];
        const ex = bx - ax;
        const ey = by - ay;
        const t = Math.max(0, Math.min(1, ((localX - ax) * ex + (localY - ay) * ey) / (ex * ex + ey * ey)));
        if (Math.hypot(localX - (ax + t * ex), localY - (ay + t * ey)) <= radius) return true;
    }
    return false;
}

function updateFlight(time) {
    const deltaTime = Math.min((time - previousTime) / 1000, 0.05);
    previousTime = time;
    if (launching) {
        requestAnimationFrame(updateFlight);
        return;
    }

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

    let reachLeft = Infinity, reachRight = -Infinity;
    let reachTop = Infinity, reachBottom = -Infinity;
    for (const [hx, hy] of hull) {
        const px = hx * cos - hy * sin;
        const py = hx * sin + hy * cos;
        reachLeft = Math.min(reachLeft, px);
        reachRight = Math.max(reachRight, px);
        reachTop = Math.min(reachTop, py);
        reachBottom = Math.max(reachBottom, py);
    }

    const centerX = x + width / 2;
    const centerY = y + height / 2;
    const pageWidth = document.documentElement.clientWidth;

    if (centerX + reachLeft < 0) {
        x -= centerX + reachLeft;
        vx = Math.abs(vx) * bounce;
    } else if (centerX + reachRight > pageWidth) {
        x -= centerX + reachRight - pageWidth;
        vx = -Math.abs(vx) * bounce;
    }
    if (centerY + reachTop < 0) {
        y -= centerY + reachTop;
        vy = Math.abs(vy) * bounce;
    } else if (centerY + reachBottom > worldHeight) {
        y -= centerY + reachBottom - worldHeight;
        vy = -Math.abs(vy) * bounce;
    }


    ship.style.left = `${x}px`;
    ship.style.top = `${y}px`;
    ship.style.transform = `rotate(${angle}deg)`

    activePlanet = null;
    if (document.activeElement === ship && demo.hidden) {
        activePlanet = destinations.find(destination => 
            touchesPlanet(destination, width, height, sin, cos)
        ) ?? null;
    }

    planetPrompt.hidden = !activePlanet;
    if (activePlanet) {
        const action = activePlanet.element.tagName === "A" ? "visit" : "enter";
        planetPrompt.textContent = `Press F to ${action} ${activePlanet.name}`;
    }

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

window.addEventListener("keydown", event => {
    if (event.key !== "Enter" || event.repeat || event.defaultPrevented) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (launching || !demo.hidden || ship.style.visibility !== "visible") return;
    event.preventDefault();
    keys.clear()
    ship.focus({ preventScroll: true });
})