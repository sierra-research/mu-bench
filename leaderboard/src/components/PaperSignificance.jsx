import { useState } from "react";
import { PROVIDERS as LB_PROVIDERS, SIGNIFICANCE, SIGNIFICANCE_PROVIDERS, VARIANCE } from "../data/data.js";
import "./PaperSignificance.css";

// Build display-name lookup from the leaderboard provider list, then
// compose the SignificanceMatrix's row order from significance.json's
// providers field (which is sorted best -> worst by the metric the
// JSON was last written under). For the VarianceChart we use the same
// order so the two side-by-side widgets stay aligned.
const PROVIDER_NAMES = Object.fromEntries(
    (LB_PROVIDERS || []).map((p) => [p.id, p.model || p.id]),
);
const PROVIDERS = (SIGNIFICANCE_PROVIDERS || []).map((p) => ({
    id: p.id,
    name: PROVIDER_NAMES[p.id] || p.name || p.id,
}));

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
    const data = SIGNIFICANCE[metric];
    const [hoveredCell, setHoveredCell] = useState(null);

    if (!data || !data.pairwise || !data.means) {
        return (
            <div className="sig-matrix-wrap">
                <p className="sig-caption">No significance data for {metric} yet.</p>
            </div>
        );
    }

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

function variancePresent(metric) {
    return Object.values(VARIANCE).some(
        (locales) =>
            locales &&
            Object.values(locales).some((m) => m && m[metric] && typeof m[metric].mean === "number"),
    );
}

function nWaves() {
    // Use the first provider/locale's _n_waves to label the chart.
    const firstProv = Object.values(VARIANCE)[0];
    if (!firstProv) return 4;
    const firstLoc = Object.values(firstProv)[0];
    return firstLoc?._n_waves ?? 4;
}

export default function PaperSignificance() {
    const [metric, setMetric] = useState("significantWer");
    const metricLabel = METRICS.find((m) => m.key === metric)?.label;
    const sig = SIGNIFICANCE[metric] || {};
    const iters = sig.numIterations ? sig.numIterations.toLocaleString() : "?";
    const convs = sig.numConversations ? sig.numConversations.toLocaleString() : "?";
    const N = nWaves();
    const showVariance = variancePresent(metric);

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
                        Paired bootstrap resampling ({iters} iterations, {convs} conversations). Resampled at the
                        conversation level to account for within-conversation correlation. p-value = P(row ≥ column).{" "}
                        <span className="sig-legend-sig">Green</span> = significant (p &lt; 0.05). Gray = not
                        distinguishable.
                    </p>
                </div>

                {showVariance && (
                    <div className="sig-section">
                        <h4 className="sig-section-title">
                            Scorer stability — {metricLabel} across {N} independent runs
                        </h4>
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
                            Error bars show ±1 standard deviation across {N} full pipeline re-runs. Rankings never
                            changed between runs.
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}
