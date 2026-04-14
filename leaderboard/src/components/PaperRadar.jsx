import { useState } from "react";
import "./PaperRadar.css";

const LOCALES = ["en-US", "es-MX", "tr-TR", "vi-VN", "zh-CN"];
const LOCALE_LABELS = {
    "en-US": "English",
    "es-MX": "Spanish",
    "tr-TR": "Turkish",
    "vi-VN": "Vietnamese",
    "zh-CN": "Chinese",
};

const PROVIDERS = [
    { id: "google-chirp3", label: "Google Chirp-3", color: "#059669" },
    { id: "elevenlabs-scribe-v2", label: "ElevenLabs Scribe v2", color: "#7c3aed" },
    { id: "deepgram-nova3", label: "Deepgram Nova-3", color: "#2563eb" },
    { id: "openai-gpt4o-transcribe", label: "OpenAI GPT-4o", color: "#dc2626" },
    { id: "azure", label: "Microsoft Azure", color: "#d97706" },
];

const SCORES = {
    "google-chirp3": {
        "en-US": { wer: 0.053, sigWer: 0.046, quality: 2.9, latency: 2146 },
        "es-MX": { wer: 0.11, sigWer: 0.087, quality: 2.86, latency: 1975 },
        "tr-TR": { wer: 0.114, sigWer: 0.084, quality: 2.84, latency: 2041 },
        "vi-VN": { wer: 0.1, sigWer: 0.087, quality: 2.82, latency: 2165 },
        "zh-CN": { wer: 0.16, sigWer: 0.225, quality: 2.31, latency: 1715 },
    },
    "deepgram-nova3": {
        "en-US": { wer: 0.073, sigWer: 0.069, quality: 2.83, latency: 655 },
        "es-MX": { wer: 0.123, sigWer: 0.122, quality: 2.72, latency: 640 },
        "tr-TR": { wer: 0.127, sigWer: 0.144, quality: 2.53, latency: 677 },
        "vi-VN": { wer: 0.448, sigWer: 0.546, quality: 1.55, latency: 946 },
        "zh-CN": { wer: 0.254, sigWer: 0.368, quality: 1.84, latency: 1329 },
    },
    "elevenlabs-scribe-v2": {
        "en-US": { wer: 0.129, sigWer: 0.051, quality: 2.85, latency: 899 },
        "es-MX": { wer: 0.192, sigWer: 0.063, quality: 2.8, latency: 828 },
        "tr-TR": { wer: 0.135, sigWer: 0.079, quality: 2.75, latency: 903 },
        "vi-VN": { wer: 0.284, sigWer: 0.148, quality: 2.62, latency: 821 },
        "zh-CN": { wer: 0.177, sigWer: 0.209, quality: 2.28, latency: 560 },
    },
    "openai-gpt4o-transcribe": {
        "en-US": { wer: 0.076, sigWer: 0.072, quality: 2.87, latency: 980 },
        "es-MX": { wer: 0.108, sigWer: 0.101, quality: 2.78, latency: 1075 },
        "tr-TR": { wer: 0.133, sigWer: 0.142, quality: 2.66, latency: 1159 },
        "vi-VN": { wer: 0.172, sigWer: 0.229, quality: 2.59, latency: 1042 },
        "zh-CN": { wer: 0.202, sigWer: 0.262, quality: 2.19, latency: 936 },
    },
    azure: {
        "en-US": { wer: 0.046, sigWer: 0.033, quality: 2.92, latency: 638 },
        "es-MX": { wer: 0.134, sigWer: 0.101, quality: 2.77, latency: 476 },
        "tr-TR": { wer: 0.137, sigWer: 0.176, quality: 2.58, latency: 1815 },
        "vi-VN": { wer: 0.272, sigWer: 0.256, quality: 2.5, latency: 586 },
        "zh-CN": { wer: 0.207, sigWer: 0.219, quality: 2.26, latency: 424 },
    },
};

const METRICS = [
    { key: "wer", label: "WER", max: 0.55, invert: true },
    { key: "sigWer", label: "UER", max: 0.6, invert: true },
    { key: "quality", label: "Quality", max: 3.0, invert: false },
    { key: "latency", label: "Latency (p95)", max: 2500, invert: true },
];

const CX = 200;
const CY = 200;
const R = 150;
const RINGS = 4;

function polarToXY(angle, radius) {
    const rad = (angle - 90) * (Math.PI / 180);
    return [CX + radius * Math.cos(rad), CY + radius * Math.sin(rad)];
}

export default function PaperRadar() {
    const [metricIdx, setMetricIdx] = useState(0);
    const [hovered, setHovered] = useState(null);
    const metric = METRICS[metricIdx];

    const angleStep = 360 / LOCALES.length;
    const axes = LOCALES.map((_, i) => i * angleStep);

    function getRadius(value) {
        let normalized = value / metric.max;
        if (metric.invert) normalized = 1 - normalized;
        return Math.max(0, Math.min(1, normalized)) * R;
    }

    return (
        <div className="radar-widget">
            <div className="radar-metric-tabs">
                {METRICS.map((m, i) => (
                    <button
                        key={m.key}
                        className={`radar-tab ${i === metricIdx ? "radar-tab--active" : ""}`}
                        onClick={() => setMetricIdx(i)}
                    >
                        {m.label}
                    </button>
                ))}
            </div>
            <div className="radar-content">
                <svg viewBox="0 0 400 400" className="radar-svg">
                    {Array.from({ length: RINGS }, (_, ring) => {
                        const r = ((ring + 1) / RINGS) * R;
                        const points = axes.map((a) => polarToXY(a, r).join(",")).join(" ");
                        return <polygon key={ring} points={points} fill="none" stroke="#e2e8f0" strokeWidth="1" />;
                    })}
                    {axes.map((angle, i) => {
                        const [x, y] = polarToXY(angle, R);
                        const [lx, ly] = polarToXY(angle, R + 24);
                        return (
                            <g key={i}>
                                <line x1={CX} y1={CY} x2={x} y2={y} stroke="#e2e8f0" strokeWidth="1" />
                                <text
                                    x={lx}
                                    y={ly}
                                    textAnchor="middle"
                                    dominantBaseline="middle"
                                    fontSize="11"
                                    fontWeight="600"
                                    fill="#4a5568"
                                >
                                    {LOCALE_LABELS[LOCALES[i]]}
                                </text>
                            </g>
                        );
                    })}
                    {PROVIDERS.map((provider) => {
                        const scores = SCORES[provider.id];
                        const points = LOCALES.map((locale, i) => {
                            const val = scores[locale][metric.key];
                            const r = getRadius(val);
                            return polarToXY(axes[i], r).join(",");
                        }).join(" ");
                        const isActive = hovered === null || hovered === provider.id;
                        return (
                            <polygon
                                key={provider.id}
                                points={points}
                                fill={provider.color}
                                fillOpacity={isActive ? 0.12 : 0.03}
                                stroke={provider.color}
                                strokeWidth={isActive ? 2.5 : 1}
                                strokeOpacity={isActive ? 1 : 0.2}
                                style={{ transition: "all 0.2s ease" }}
                            />
                        );
                    })}
                    {PROVIDERS.map((provider) => {
                        const scores = SCORES[provider.id];
                        const isActive = hovered === null || hovered === provider.id;
                        if (!isActive) return null;
                        return LOCALES.map((locale, i) => {
                            const val = scores[locale][metric.key];
                            const r = getRadius(val);
                            const [x, y] = polarToXY(axes[i], r);
                            return (
                                <circle
                                    key={`${provider.id}-${locale}`}
                                    cx={x}
                                    cy={y}
                                    r="4"
                                    fill={provider.color}
                                    stroke="white"
                                    strokeWidth="1.5"
                                />
                            );
                        });
                    })}
                </svg>
                <div className="radar-legend">
                    {PROVIDERS.map((p) => (
                        <button
                            key={p.id}
                            className={`radar-legend-item ${hovered === p.id ? "radar-legend-item--active" : ""}`}
                            onMouseEnter={() => setHovered(p.id)}
                            onMouseLeave={() => setHovered(null)}
                        >
                            <span className="radar-legend-dot" style={{ background: p.color }} />
                            {p.label}
                        </button>
                    ))}
                </div>
            </div>
            <p className="radar-note">
                {metric.invert
                    ? `Larger area = better (lower ${metric.key === "latency" ? "latency" : "error rate"})`
                    : "Larger area = better (higher score)"}
            </p>
        </div>
    );
}
