import { useState } from "react";
import { PROVIDERS as DATA_PROVIDERS } from "../data/data.js";
import "./PaperHeatmap.css";

const LOCALES = ["en-US", "es-MX", "tr-TR", "vi-VN", "zh-CN"];
const LOCALE_FLAGS = {
    "en-US": "\u{1F1FA}\u{1F1F8}",
    "es-MX": "\u{1F1F2}\u{1F1FD}",
    "tr-TR": "\u{1F1F9}\u{1F1F7}",
    "vi-VN": "\u{1F1FB}\u{1F1F3}",
    "zh-CN": "\u{1F1E8}\u{1F1F3}",
};

const PROVIDER_LABELS = {
    "google-chirp3": "Google Chirp-3",
    "elevenlabs-scribe-v2": "ElevenLabs Scribe v2",
    "deepgram-nova3": "Deepgram Nova-3",
    "openai-gpt4o-mini-transcribe": "OpenAI GPT-4o Mini",
    azure: "Microsoft Azure",
};

const PROVIDER_ORDER = [
    "google-chirp3",
    "elevenlabs-scribe-v2",
    "deepgram-nova3",
    "openai-gpt4o-mini-transcribe",
    "azure",
];
const PROVIDERS = PROVIDER_ORDER.map((id) => ({ id, label: PROVIDER_LABELS[id] || id }));

function buildScores() {
    const scores = {};
    for (const p of DATA_PROVIDERS) {
        scores[p.id] = {};
        for (const [locale, lr] of Object.entries(p.localeResults)) {
            scores[p.id][locale] = {
                wer: lr.wer,
                sigWer: lr.significantWer,
                latencyP50: lr.latencyP50Ms,
                latency: lr.latencyP95Ms,
            };
        }
    }
    return scores;
}
const SCORES = buildScores();

const METRICS = [
    { key: "sigWer", label: "UER", format: (v) => `${(v * 100).toFixed(1)}%`, lower: true },
    { key: "wer", label: "WER", format: (v) => `${(v * 100).toFixed(1)}%`, lower: true },
    { key: "latencyP50", label: "Latency (p50)", format: (v) => `${Math.round(v)} ms`, lower: true },
    { key: "latency", label: "Latency (p95)", format: (v) => `${Math.round(v)} ms`, lower: true },
];

const COLOR_RANGES = {
    wer: { min: 0, max: 0.55 },
    sigWer: { min: 0, max: 0.55 },
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
