import { useState } from "react";
import { PROVIDERS as DATA_PROVIDERS } from "../data/data.js";
import "./PaperRadar.css";

const LOCALES = ["en-US", "es-MX", "tr-TR", "vi-VN", "zh-CN"];
const LOCALE_LABELS = {
    "en-US": "English",
    "es-MX": "Spanish",
    "tr-TR": "Turkish",
    "vi-VN": "Vietnamese",
    "zh-CN": "Chinese",
};

const PROVIDER_COLORS = {
    "google-chirp3": "#059669",
    "elevenlabs-scribe-v2": "#7c3aed",
    "deepgram-nova3": "#2563eb",
    "openai-gpt4o-mini-transcribe": "#dc2626",
    azure: "#d97706",
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
const PROVIDERS = PROVIDER_ORDER.map((id) => ({
    id,
    label: PROVIDER_LABELS[id] || id,
    color: PROVIDER_COLORS[id] || "#888",
}));

function buildScores() {
    const scores = {};
    for (const p of DATA_PROVIDERS) {
        scores[p.id] = {};
        for (const [locale, lr] of Object.entries(p.localeResults)) {
            scores[p.id][locale] = {
                wer: lr.wer,
                sigWer: lr.significantWer,
                latency: lr.completeP95Ms,
            };
        }
    }
    return scores;
}
const SCORES = buildScores();

const METRICS = [
    { key: "sigWer", label: "UER", max: 0.6, invert: true },
    { key: "wer", label: "WER", max: 0.55, invert: true },
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
