import { useState } from "react";
import "./PaperSignificance.css";

const PROVIDERS = [
    { id: "google-chirp3", name: "Google Chirp-3" },
    { id: "elevenlabs-scribe-v2", name: "ElevenLabs Scribe v2" },
    { id: "azure", name: "Microsoft Azure" },
    { id: "openai-gpt4o-mini-transcribe", name: "OpenAI GPT-4o Mini" },
    { id: "deepgram-nova3", name: "Deepgram Nova-3" },
];

const SIG_DATA = {
    significantWer: {
        means: {
            "google-chirp3": 0.1049,
            "elevenlabs-scribe-v2": 0.1631,
            azure: 0.1988,
            "openai-gpt4o-mini-transcribe": 0.1889,
            "deepgram-nova3": 0.2572,
        },
        pairwise: [
            [null, 0.0, 0.0, 0.0, 0.0],
            [1.0, null, 0.0, 0.0, 0.0],
            [1.0, 1.0, null, 0.929, 0.0],
            [1.0, 1.0, 0.073, null, 0.0],
            [1.0, 1.0, 1.0, 1.0, null],
        ],
        numConversations: 250,
        numIterations: 10000,
    },
    wer: {
        means: {
            "google-chirp3": 0.0596,
            "elevenlabs-scribe-v2": 0.0789,
            azure: 0.09,
            "openai-gpt4o-mini-transcribe": 0.0928,
            "deepgram-nova3": 0.1149,
        },
        pairwise: [
            [null, 0.0, 0.0, 0.0, 0.0],
            [1.0, null, 0.0, 0.0, 0.0],
            [1.0, 1.0, null, 0.069, 0.0],
            [1.0, 1.0, 0.925, null, 0.0],
            [1.0, 1.0, 1.0, 1.0, null],
        ],
        numConversations: 250,
        numIterations: 10000,
    },
};

// Means come from the latest scoring run. The std values were measured
// across 4 independent re-runs of the full normalization+scoring pipeline
// and reflect LLM-scoring stability, which is not affected by the WER
// aggregation method, so they're carried forward as a representative
// indication of run-to-run variability.
const VARIANCE = {
    "google-chirp3": {
        "en-US": { wer: { mean: 0.0449, std: 0.0005 }, significantWer: { mean: 0.0509, std: 0.0036 } },
        "zh-CN": { wer: { mean: 0.0718, std: 0.0008 }, significantWer: { mean: 0.1645, std: 0.0023 } },
    },
    "elevenlabs-scribe-v2": {
        "en-US": { wer: { mean: 0.0417, std: 0.0007 }, significantWer: { mean: 0.0644, std: 0.0032 } },
        "zh-CN": { wer: { mean: 0.1109, std: 0.0017 }, significantWer: { mean: 0.2337, std: 0.0088 } },
    },
    azure: {
        "en-US": { wer: { mean: 0.0364, std: 0.0002 }, significantWer: { mean: 0.0334, std: 0.0018 } },
        "zh-CN": { wer: { mean: 0.1271, std: 0.0015 }, significantWer: { mean: 0.2524, std: 0.0136 } },
    },
    "openai-gpt4o-mini-transcribe": {
        "en-US": { wer: { mean: 0.037, std: 0.0004 }, significantWer: { mean: 0.0484, std: 0.001 } },
        "zh-CN": { wer: { mean: 0.131, std: 0.0021 }, significantWer: { mean: 0.2137, std: 0.0022 } },
    },
    "deepgram-nova3": {
        "en-US": { wer: { mean: 0.0456, std: 0.0005 }, significantWer: { mean: 0.0498, std: 0.0025 } },
        "zh-CN": { wer: { mean: 0.154, std: 0.0034 }, significantWer: { mean: 0.282, std: 0.0161 } },
    },
};

const METRICS = [
    { key: "significantWer", label: "UER" },
    { key: "wer", label: "WER" },
];

function pLabel(p) {
    if (p === null) return "";
    if (p < 0.001) return "<.001";
    if (p < 0.01) return p.toFixed(3);
    return p.toFixed(2);
}

function cellColor(p, isLowerTriangle) {
    if (p === null) return "transparent";
    if (!isLowerTriangle) {
        if (p < 0.05) return "rgba(16, 185, 129, 0.18)";
        return "rgba(156, 163, 175, 0.12)";
    }
    if (p < 0.05) return "rgba(16, 185, 129, 0.18)";
    return "rgba(156, 163, 175, 0.12)";
}

function SignificanceMatrix({ metric }) {
    const data = SIG_DATA[metric];
    const [hoveredCell, setHoveredCell] = useState(null);

    return (
        <div className="sig-matrix-wrap">
            <table className="sig-matrix">
                <thead>
                    <tr>
                        <th className="sig-corner"></th>
                        {PROVIDERS.map((p, j) => (
                            <th key={p.id} className="sig-col-header">
                                <span className="sig-col-name">{p.name}</span>
                                <span className="sig-col-score">{(data.means[p.id] * 100).toFixed(1)}%</span>
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {PROVIDERS.map((rowP, i) => (
                        <tr key={rowP.id}>
                            <td className="sig-row-header">
                                <span className="sig-rank">{i + 1}</span>
                                {rowP.name}
                            </td>
                            {PROVIDERS.map((colP, j) => {
                                const p = data.pairwise[i][j];
                                const isLower = i > j;
                                const isDiag = i === j;
                                const isSig = p !== null && p < 0.05;
                                const isHovered = hoveredCell && hoveredCell[0] === i && hoveredCell[1] === j;
                                return (
                                    <td
                                        key={colP.id}
                                        className={`sig-cell ${isDiag ? "sig-cell--diag" : ""} ${isSig && !isDiag ? "sig-cell--sig" : ""} ${!isSig && !isDiag ? "sig-cell--ns" : ""} ${isHovered ? "sig-cell--hover" : ""}`}
                                        style={{
                                            background: isDiag ? "#f3f4f6" : cellColor(p, isLower),
                                        }}
                                        onMouseEnter={() => setHoveredCell([i, j])}
                                        onMouseLeave={() => setHoveredCell(null)}
                                    >
                                        {isDiag ? "—" : pLabel(p)}
                                    </td>
                                );
                            })}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

function VarianceChart({ metric, locale }) {
    const maxVal = Math.max(
        ...PROVIDERS.map(
            (p) => (VARIANCE[p.id]?.[locale]?.[metric]?.mean || 0) + (VARIANCE[p.id]?.[locale]?.[metric]?.std || 0),
        ),
    );
    const scale = 100 / (maxVal * 1.15);

    return (
        <div className="var-chart">
            <div className="var-bars">
                {PROVIDERS.map((p) => {
                    const v = VARIANCE[p.id]?.[locale]?.[metric];
                    if (!v) return null;
                    const barH = v.mean * scale;
                    const errH = v.std * scale;
                    return (
                        <div key={p.id} className="var-bar-group">
                            <div className="var-bar-container" style={{ height: "100px" }}>
                                <div
                                    className="var-whisker"
                                    style={{ bottom: `${barH - errH}%`, height: `${errH * 2}%` }}
                                />
                                <div className="var-bar" style={{ height: `${barH}%` }} />
                                <span className="var-label">{(v.mean * 100).toFixed(1)}%</span>
                            </div>
                            <span className="var-provider">{p.name.split(" ")[0]}</span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

export default function PaperSignificance() {
    const [metric, setMetric] = useState("significantWer");
    const metricLabel = METRICS.find((m) => m.key === metric)?.label;

    return (
        <div className="sig-widget">
            <div className="sig-tabs">
                {METRICS.map((m) => (
                    <button
                        key={m.key}
                        className={`sig-tab ${metric === m.key ? "sig-tab--active" : ""}`}
                        onClick={() => setMetric(m.key)}
                    >
                        {m.label}
                    </button>
                ))}
            </div>

            <div className="sig-content">
                <div className="sig-section">
                    <h4 className="sig-section-title">Pairwise significance — {metricLabel}</h4>
                    <SignificanceMatrix metric={metric} />
                    <p className="sig-caption">
                        Paired bootstrap resampling ({SIG_DATA[metric].numIterations.toLocaleString()} iterations,{" "}
                        {SIG_DATA[metric].numConversations.toLocaleString()} conversations). Resampled at the
                        conversation level to account for within-conversation correlation. p-value = P(row ≥ column).{" "}
                        <span className="sig-legend-sig">Green</span> = significant (p &lt; 0.05). Gray = not
                        distinguishable.
                    </p>
                </div>

                <div className="sig-section">
                    <h4 className="sig-section-title">Scorer stability — {metricLabel} across 4 independent runs</h4>
                    <div className="var-locale-row">
                        <div className="var-locale-col">
                            <span className="var-locale-label">en-US</span>
                            <VarianceChart metric={metric} locale="en-US" />
                        </div>
                        <div className="var-locale-col">
                            <span className="var-locale-label">zh-CN</span>
                            <VarianceChart metric={metric} locale="zh-CN" />
                        </div>
                    </div>
                    <p className="sig-caption">
                        Error bars show ±1 standard deviation across 4 full pipeline re-runs. Rankings never changed
                        between runs.
                    </p>
                </div>
            </div>
        </div>
    );
}
