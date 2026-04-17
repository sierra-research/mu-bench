import { defineConfig } from "vite";
import { resolve } from "path";
import react from "@vitejs/plugin-react";

export default defineConfig({
    plugins: [react()],
    base: process.env.NODE_ENV === "production" ? process.env.GITHUB_PAGES_BASE || "/" : "/",
    build: {
        outDir: "dist",
        rollupOptions: {
            input: {
                main: resolve(__dirname, "index.html"),
                "widget-globe": resolve(__dirname, "widgets/globe/index.html"),
                "widget-audio-samples": resolve(__dirname, "widgets/audio-samples/index.html"),
                "widget-coverage-matrix": resolve(__dirname, "widgets/coverage-matrix/index.html"),
                "widget-heatmap": resolve(__dirname, "widgets/heatmap/index.html"),
                "widget-radar": resolve(__dirname, "widgets/radar/index.html"),
                "widget-significance": resolve(__dirname, "widgets/significance/index.html"),
                "widget-error-examples": resolve(__dirname, "widgets/error-examples/index.html"),
                "widget-dataset-stats": resolve(__dirname, "widgets/dataset-stats/index.html"),
            },
        },
    },
});
