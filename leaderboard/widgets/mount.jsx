import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "../src/index.css";
import "./widget.css";

export function mountWidget(Component, id) {
    const container = document.getElementById("root");
    createRoot(container).render(
        <StrictMode>
            <Component />
        </StrictMode>,
    );

    if (window.parent === window) return;

    let lastHeight = 0;
    const post = () => {
        const height = Math.ceil(document.documentElement.scrollHeight);
        if (height === lastHeight) return;
        lastHeight = height;
        window.parent.postMessage({ type: "mu-bench-widget-height", id, height }, "*");
    };

    const observer = new ResizeObserver(post);
    observer.observe(document.body);
    window.addEventListener("load", post);
    post();
}
