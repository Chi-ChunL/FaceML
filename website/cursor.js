import "./cursor.css";
import cursorSheet from "./assets/cursor/cursor-sheet.png"

async function StartCursor() {
    const hasMouse = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

    if (!hasMouse) return;

    const image = new Image();
    image.src = cursorSheet;
    await image.decode();

    const cursor = document.createElement("div");
    cursor.className = "space-cursor";
    cursor.setAttribute("aria-hidden", "true");
    cursor.style.backgroundImage = `url("${cursorSheet}")`;
    document.body.appendChild(cursor);

    const root = document.documentElement;

    function hideCursor() {
        cursor.style.visibility = "hidden";
        root.classList.remove("custom-cursor-active");
    }

    window.addEventListener("pointermove", event => {
        if (event.pointerType !== "mouse") {
            hideCursor();
            return;
        }

        cursor.style.transform = `translate3d(${event.clientX - 10}px, ${event.clientY - 5}px, 0)`;
        cursor.style.visibility = "visible";
        root.classList.add("custom-cursor-active");
    });

    root.addEventListener("pointerleave", hideCursor);
    window.addEventListener("blur", hideCursor);
}

StartCursor().catch(error => {
    console.error("Couldn't load cursor:", error);
});