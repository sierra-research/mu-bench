import { useState } from "react";
import { SHOW_QUALITY } from "../utils/metrics.js";
import "./PaperHeatmap.css";

const LOCALES = ["en-US", "es-MX", "tr-TR", "vi-VN", "zh-CN"];
const LOCALE_FLAGS = {
    "en-US": "\u{1F1FA}\u{1F1F8}",
    "es-MX": "\u{1F1F2}\u{1F1FD}",
    "tr-TR": "\u{1F1F9}\u{1F1F7}",
    "vi-VN": "\u{1F1FB}\u{1F1F3}",
    "zh-CN": "\u{1F1E8}\u{1F1F3}",
};

const PROVIDERS = [
    { id: "google-chirp3", label: "Google Chirp-3" },
    { id: "elevenlabs-scribe-v2", label: "ElevenLabs Scribe v2" },
    { id: "deepgram-nova3", label: "Deepgram Nova-3" },
    { id: "openai-gpt4o-mini-transcribe", label: "OpenAI GPT-4o Mini" },
    { id: "azure", label: "Microsoft Azure" },
];

const SCORES = {
    "google-chirp3": {
        "en-US": { wer: 0.053, sigWer: 0.046, quality: 2.9, latencyP50: 1236, latency: 2146 },
        "es-MX": { wer: 0.11, sigWer: 0.087, quality: 2.86, latencyP50: 1192, latency: 1975 },
        "tr-TR": { wer: 0.114, sigWer: 0.084, quality: 2.84, latencyP50: 1244, latency: 2041 },
        "vi-VN": { wer: 0.1, sigWer: 0.087, quality: 2.82, latencyP50: 1254, latency: 2165 },
        "zh-CN": { wer: 0.16, sigWer: 0.225, quality: 2.31, latencyP50: 1140, latency: 1715 },
    },
    "deepgram-nova3": {
        "en-US": { wer: 0.073, sigWer: 0.069, quality: 2.83, latencyP50: 254, latency: 655 },
        "es-MX": { wer: 0.123, sigWer: 0.122, quality: 2.72, latencyP50: 286, latency: 640 },
        "tr-TR": { wer: 0.127, sigWer: 0.144, quality: 2.53, latencyP50: 248, latency: 677 },
        "vi-VN": { wer: 0.448, sigWer: 0.546, quality: 1.55, latencyP50: 241, latency: 946 },
        "zh-CN": { wer: 0.254, sigWer: 0.368, quality: 1.84, latencyP50: 202, latency: 1329 },
    },
    "elevenlabs-scribe-v2": {
        "en-US": { wer: 0.129, sigWer: 0.051, quality: 2.85, latencyP50: 432, latency: 899 },
        "es-MX": { wer: 0.192, sigWer: 0.063, quality: 2.8, latencyP50: 429, latency: 828 },
        "tr-TR": { wer: 0.135, sigWer: 0.079, quality: 2.75, latencyP50: 456, latency: 903 },
        "vi-VN": { wer: 0.284, sigWer: 0.148, quality: 2.62, latencyP50: 432, latency: 821 },
        "zh-CN": { wer: 0.177, sigWer: 0.209, quality: 2.28, latencyP50: 376, latency: 560 },
    },
    "openai-gpt4o-mini-transcribe": {
        "en-US": { wer: 0.039, sigWer: 0.051, quality: 2.87, latencyP50: 664, latency: 1328 },
        "es-MX": { wer: 0.111, sigWer: 0.145, quality: 2.78, latencyP50: 659, latency: 1375 },
        "tr-TR": { wer: 0.113, sigWer: 0.178, quality: 2.66, latencyP50: 696, latency: 1652 },
        "vi-VN": { wer: 0.209, sigWer: 0.319, quality: 2.59, latencyP50: 662, latency: 1281 },
        "zh-CN": { wer: 0.165, sigWer: 0.225, quality: 2.19, latencyP50: 566, latency: 1116 },
    },
    azure: {
        "en-US": { wer: 0.046, sigWer: 0.033, quality: 2.92, latencyP50: 276, latency: 638 },
        "es-MX": { wer: 0.134, sigWer: 0.101, quality: 2.77, latencyP50: 243, latency: 476 },
        "tr-TR": { wer: 0.137, sigWer: 0.176, quality: 2.58, latencyP50: 968, latency: 1815 },
        "vi-VN": { wer: 0.272, sigWer: 0.256, quality: 2.5, latencyP50: 277, latency: 586 },
        "zh-CN": { wer: 0.207, sigWer: 0.219, quality: 2.26, latencyP50: 263, latency: 424 },
    },
};

const ALL_METRICS = [
    { key: "wer", label: "WER", format: (v) => `${(v * 100).toFixed(1)}%`, lower: true },
    { key: "sigWer", label: "UER", format: (v) => `${(v * 100).toFixed(1)}%`, lower: true },
    { key: "quality", label: "Quality", format: (v) => v.toFixed(2), lower: false },
    { key: "latencyP50", label: "Latency (p50)", format: (v) => `${Math.round(v)} ms`, lower: true },
    { key: "latency", label: "Latency (p95)", format: (v) => `${Math.round(v)} ms`, lower: true },
];

const METRICS = SHOW_QUALITY ? ALL_METRICS : ALL_METRICS.filter((m) => m.key !== "quality");

const COLOR_RANGES = {
    wer: { min: 0, max: 0.55 },
    sigWer: { min: 0, max: 0.55 },
    quality: { min: 1.5, max: 3.0 },
    latencyP50: { min: 200, max: 1400 },
    latency: { min: 400, max: 2500 },
};

function getCellColor(value, metric) {
    const range = COLOR_RANGES[metric.key] || { min: 0, max: 1 };
    if (metric.lower) {
        const t = Math.min(Math.max((value - range.min) / (range.max - range.min), 0), 1);
        const r = Math.round(34 + t * 200);
        const g = Math.round(160 - t * 100);
        const b = Math.round(70 - t * 40);
        return `rgba(${r}, ${g}, ${b}, ${0.12 + t * 0.18})`;
    } else {
        const t = Math.min(Math.max((value - range.min) / (range.max - range.min), 0), 1);
        const r = Math.round(200 - t * 170);
        const g = Math.round(60 + t * 100);
        const b = Math.round(30 + t * 40);
        return `rgba(${r}, ${g}, ${b}, ${0.12 + t * 0.18})`;
    }
}

export default function PaperHeatmap() {
    const [metricIdx, setMetricIdx] = useState(0);
    const metric = METRICS[metricIdx];
    const [hoveredCell, setHoveredCell] = useState(null);

    return (
        <div className="hm-widget">
            <div className="hm-metric-tabs">
                {METRICS.map((m, i) => (
                    <button
                        key={m.key}
                        className={`hm-tab ${i === metricIdx ? "hm-tab--active" : ""}`}
                        onClick={() => setMetricIdx(i)}
                    >
                        {m.label}
                    </button>
                ))}
            </div>
            <div className="hm-table-wrap">
                <table className="hm-table">
                    <thead>
                        <tr>
                            <th className="hm-th-provider">Provider</th>
                            {LOCALES.map((l) => (
                                <th key={l} className="hm-th-locale">
                                    {LOCALE_FLAGS[l]} {l}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {PROVIDERS.map((p) => (
                            <tr key={p.id}>
                                <td className="hm-td-provider">{p.label}</td>
                                {LOCALES.map((l) => {
                                    const val = SCORES[p.id][l][metric.key];
                                    const isHovered = hoveredCell === `${p.id}-${l}`;
                                    return (
                                        <td
                                            key={l}
                                            className={`hm-td-cell ${isHovered ? "hm-td-cell--hover" : ""}`}
                                            style={{ background: getCellColor(val, metric) }}
                                            onMouseEnter={() => setHoveredCell(`${p.id}-${l}`)}
                                            onMouseLeave={() => setHoveredCell(null)}
                                        >
                                            {metric.format(val)}
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <p className="hm-legend">{metric.lower ? "Greener = better (lower)" : "Greener = better (higher)"}</p>
        </div>
    );
}
