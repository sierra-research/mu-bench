import { useState, useEffect, useRef, useMemo } from "react";
import { LOCALES, PROVIDERS, SAMPLE_UTTERANCES, getProviderScore } from "../data/data.js";
import { METRICS, VISIBLE_METRIC_KEYS, getSortDirection } from "../utils/metrics.js";
import "./Leaderboard.css";
import sampleEnUS_0 from "../samples/en-US/conv-39-turn-8.wav";
import sampleEsMX_0 from "../samples/es-MX/conv-49-turn-5.wav";
import sampleTrTR_0 from "../samples/tr-TR/conv-41-turn-11.wav";
import sampleViVN_0 from "../samples/vi-VN/conv-20-turn-14.wav";
import sampleZhCN_0 from "../samples/zh-CN/conv-4-turn-12.wav";

const LOCALE_OPTIONS = [{ code: "overall", label: "Overall", flag: "\u{1F4CA}" }, ...LOCALES];

const SAMPLES = [
    { src: sampleEnUS_0, locale: "en-US", label: "English (US)" },
    { src: sampleEsMX_0, locale: "es-MX", label: "Spanish (MX)" },
    { src: sampleTrTR_0, locale: "tr-TR", label: "Turkish" },
    { src: sampleViVN_0, locale: "vi-VN", label: "Vietnamese" },
    { src: sampleZhCN_0, locale: "zh-CN", label: "Chinese (CN)" },
];

export default function Leaderboard() {
    const [locale, setLocale] = useState(() => {
        return localStorage.getItem("lb-locale") || "overall";
    });

    const [sortBy, setSortBy] = useState("significantWer");

    const [sampleIndex, setSampleIndex] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const audioRef = useRef(null);
    const canvasRef = useRef(null);
    const audioContextRef = useRef(null);
    const analyserRef = useRef(null);
    const animFrameRef = useRef(null);
    const [selectedProvider, setSelectedProvider] = useState(null);

    useEffect(() => {
        localStorage.setItem("lb-locale", locale);
    }, [locale]);

    useEffect(() => {
        if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current.load();
            setIsPlaying(false);
        }
    }, [sampleIndex]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext("2d");
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const w = rect.width;
        const h = rect.height;
        const barW = 2;
        const gap = 1;
        const step = barW + gap;
        const barCount = Math.floor(w / step);

        if (!isPlaying) {
            for (let i = 0; i < barCount; i++) {
                ctx.fillStyle = "#a0aec0";
                ctx.fillRect(i * step, (h - 2) / 2, barW, 2);
            }
            return;
        }

        const analyser = analyserRef.current;
        if (!analyser) return;

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        function draw() {
            animFrameRef.current = requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);

            ctx.clearRect(0, 0, w, h);

            for (let i = 0; i < barCount; i++) {
                const idx = Math.round((i / barCount) * bufferLength * 0.4);
                const value = dataArray[idx] / 255;
                const barH = Math.max(2, value * h);
                const x = i * step;
                const y = (h - barH) / 2;

                ctx.fillStyle = "#2d3748";
                ctx.fillRect(x, y, barW, barH);
            }
        }

        animFrameRef.current = requestAnimationFrame(draw);
        return () => {
            if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
        };
    }, [isPlaying]);

    const togglePlay = () => {
        const audio = audioRef.current;
        if (!audio) return;

        if (!audioContextRef.current) {
            const ctx = new AudioContext();
            const analyser = ctx.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.85;
            const source = ctx.createMediaElementSource(audio);
            source.connect(analyser);
            analyser.connect(ctx.destination);
            audioContextRef.current = ctx;
            analyserRef.current = analyser;
        }
        if (audioContextRef.current.state === "suspended") {
            audioContextRef.current.resume();
        }

        if (isPlaying) {
            audio.pause();
            setIsPlaying(false);
        } else {
            audio.play();
            setIsPlaying(true);
        }
    };

    const rankedProviders = useMemo(() => {
        const withScores = PROVIDERS.map((p) => ({
            ...p,
            sigWer: getProviderScore(p, "significantWer", locale),
            latency: getProviderScore(p, "latencyP95Ms", locale),
        })).filter((p) => p.sigWer !== null);

        const dir = getSortDirection(sortBy);
        withScores.sort((a, b) => {
            const aVal = sortBy === "latencyP95Ms" ? a.latency : a.sigWer;
            const bVal = sortBy === "latencyP95Ms" ? b.latency : b.sigWer;
            if (aVal === null && bVal === null) return 0;
            if (aVal === null) return 1;
            if (bVal === null) return -1;
            return dir === "asc" ? aVal - bVal : bVal - aVal;
        });

        return withScores.map((p, i) => ({
            ...p,
            rank: i + 1,
        }));
    }, [sortBy, locale]);

    const sigWerMetric = METRICS.significantWer;
    const latencyMetric = METRICS.latencyP95Ms;

    return (
        <div className="leaderboard">
            <div className="leaderboard-inner">
                {SAMPLES.length > 0 && (
                    <div className="sample-player-compact">
                        <span className="sample-label">Listen to samples</span>
                        <div className="waveform-player">
                            <button
                                className="waveform-play-btn"
                                onClick={togglePlay}
                                aria-label={isPlaying ? "Pause" : "Play"}
                            >
                                {isPlaying ? "\u23F8" : "\u25B6"}
                            </button>
                            <canvas className="waveform-canvas" ref={canvasRef} />
                        </div>
                        <audio
                            ref={audioRef}
                            src={SAMPLES[sampleIndex]?.src}
                            onEnded={() => {
                                setIsPlaying(false);
                                setSampleIndex((prev) => (prev + 1) % SAMPLES.length);
                            }}
                        />
                        <div className="sample-dots">
                            {SAMPLES.map((_, i) => (
                                <button
                                    key={i}
                                    className={`sample-dot ${i === sampleIndex ? "active" : ""}`}
                                    onClick={() => setSampleIndex(i)}
                                    aria-label={`Sample ${i + 1}`}
                                />
                            ))}
                        </div>
                    </div>
                )}

                <h2 className="leaderboard-title">Leaderboard</h2>

                <div className="controls">
                    <div className="control-group">
                        <label className="control-label">Locale</label>
                        <div className="locale-pills">
                            {LOCALE_OPTIONS.map((l) => (
                                <button
                                    key={l.code}
                                    className={`pill ${locale === l.code ? "active" : ""}`}
                                    onClick={() => setLocale(l.code)}
                                >
                                    <span className="pill-flag">{l.flag}</span>
                                    <span className="pill-code">{l.code === "overall" ? "Overall" : l.code}</span>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="table-wrapper">
                    <table className="lb-table">
                        <thead>
                            <tr>
                                <th className="col-rank">Rank</th>
                                <th className="col-model">Model</th>
                                <th className="col-org">Organization</th>
                                <th className="col-date">Date</th>
                                <th
                                    className={`col-sig-wer sortable ${sortBy === "significantWer" ? "sorted" : ""}`}
                                    onClick={() => setSortBy("significantWer")}
                                >
                                    Utterance Error Rate {sortBy === "significantWer" ? "▲" : ""}
                                    <span className="col-sig-wer-footnote">*</span>
                                </th>
                                <th
                                    className={`col-latency sortable ${sortBy === "latencyP95Ms" ? "sorted" : ""}`}
                                    onClick={() => setSortBy("latencyP95Ms")}
                                >
                                    Latency (p95) {sortBy === "latencyP95Ms" ? "▲" : ""}
                                </th>
                                <th className="col-arrow"></th>
                            </tr>
                        </thead>
                        <tbody>
                            {rankedProviders.length === 0 && (
                                <tr>
                                    <td colSpan="7" className="empty-state">
                                        No results available{locale !== "overall" ? ` for ${locale}` : ""}. Submissions
                                        are being processed.
                                    </td>
                                </tr>
                            )}
                            {rankedProviders.map((provider) => (
                                <ProviderRow
                                    key={provider.id}
                                    provider={provider}
                                    sigWerMetric={sigWerMetric}
                                    latencyMetric={latencyMetric}
                                    onRowClick={() => setSelectedProvider(provider.id)}
                                />
                            ))}
                        </tbody>
                    </table>
                </div>

                <p className="uer-footnote">
                    * Fraction of utterances containing at least one meaning-changing transcription error.
                </p>

                <div className="action-cards">
                    <div className="action-card">
                        <span className="action-card-title">View the Dataset</span>
                        <span className="action-card-desc">
                            This dataset contains 250 real phone calls spanning 5.1 hours of audio, in which users
                            across five locales spoke to an AI banking agent about their credit card details.
                        </span>
                        <a
                            className="action-card-btn"
                            href="https://huggingface.co/datasets/sierra-research/mu-bench"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            Download on Hugging Face
                        </a>
                    </div>
                    <div className="action-card">
                        <span className="action-card-title">Add Your Model</span>
                        <span className="action-card-desc">
                            Submit your model's transcripts and they will be automatically scored and ranked. See the
                            submission guide for formatting and step-by-step instructions.
                        </span>
                        <a
                            className="action-card-btn"
                            href="https://github.com/sierra-research/mu-bench/blob/main/submissions/SUBMITTING.md"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            Submission Instructions
                        </a>
                    </div>
                    <div className="action-card">
                        <span className="action-card-title">Read About Our Metrics</span>
                        <span className="action-card-desc">
                            We focus on the metrics that matter in production: latency and semantic accuracy,
                            emphasizing meaning-changing transcription errors over surface-level formatting.
                        </span>
                        <a className="action-card-btn" href="#paper">
                            Read the Sierra Blog
                        </a>
                    </div>
                </div>
            </div>

            {selectedProvider && (
                <ProviderDetail
                    providerId={selectedProvider}
                    providers={PROVIDERS}
                    locale={locale}
                    onClose={() => setSelectedProvider(null)}
                />
            )}
        </div>
    );
}

function ProviderRow({ provider, sigWerMetric, latencyMetric, onRowClick }) {
    return (
        <tr className="provider-row" onClick={onRowClick}>
            <td className="col-rank">
                <span
                    className={`rank-number ${provider.rank === 1 ? "rank-gold" : provider.rank === 2 ? "rank-silver" : provider.rank === 3 ? "rank-bronze" : ""}`}
                >
                    #{provider.rank}
                </span>
            </td>
            <td className="col-model">
                <span className="model-name">{provider.model}</span>
            </td>
            <td className="col-org">{provider.organization}</td>
            <td className="col-date">{provider.modelDate || "\u2014"}</td>
            <td className="col-sig-wer">
                {provider.sigWer !== null ? (
                    <span className="score-value">{sigWerMetric.format(provider.sigWer)}</span>
                ) : (
                    <span className="no-data">&mdash;</span>
                )}
            </td>
            <td className="col-latency">
                {provider.latency !== null ? (
                    <span className="score-value">{latencyMetric.format(provider.latency)}</span>
                ) : (
                    <span className="no-data">&mdash;</span>
                )}
            </td>
            <td className="col-arrow">
                <span className="row-arrow">{"\u203A"}</span>
            </td>
        </tr>
    );
}

function ProviderDetailOverall({ provider, providerId }) {
    const [showScores, setShowScores] = useState(false);
    const [showUtterances, setShowUtterances] = useState(false);

    const sigWerMetric = METRICS.significantWer;
    const latencyMetric = METRICS.latencyP95Ms;

    const scoreRows = useMemo(() => {
        const overallSigWer = getProviderScore(provider, "significantWer", "overall");
        const overallLatency = getProviderScore(provider, "latencyP95Ms", "overall");
        const rows = [{ label: "Overall", sigWer: overallSigWer, latency: overallLatency, isOverall: true }];
        for (const l of LOCALES) {
            rows.push({
                label: l.code,
                sigWer: getProviderScore(provider, "significantWer", l.code),
                latency: getProviderScore(provider, "latencyP95Ms", l.code),
                isOverall: false,
            });
        }
        return rows;
    }, [provider]);

    const utterances = useMemo(() => {
        return LOCALES.map((l) => (SAMPLE_UTTERANCES[l.code] || [])[0]).filter(Boolean);
    }, []);

    return (
        <div className="provider-detail-sections">
            <div className="collapsible-section">
                <div className="collapsible-toggle" onClick={() => setShowScores(!showScores)}>
                    <span className={`collapsible-caret ${showScores ? "open" : ""}`}>{"\u25B6"}</span>
                    <span className="collapsible-label">Metrics by Locale</span>
                </div>
                {showScores && (
                    <div className="collapsible-content">
                        <table className="scores-table">
                            <thead>
                                <tr>
                                    <th>Locale</th>
                                    <th>UER {"\u2193"}</th>
                                    <th>Latency (p95) {"\u2193"}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {scoreRows.map((row) => (
                                    <tr key={row.label} className={row.isOverall ? "scores-row-overall" : ""}>
                                        <td>
                                            {row.isOverall ? (
                                                row.label
                                            ) : (
                                                <span className="utterance-locale-badge">{row.label}</span>
                                            )}
                                        </td>
                                        <td>{row.sigWer !== null ? sigWerMetric.format(row.sigWer) : "\u2014"}</td>
                                        <td>{row.latency !== null ? latencyMetric.format(row.latency) : "\u2014"}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            <div className="collapsible-section">
                <div className="collapsible-toggle" onClick={() => setShowUtterances(!showUtterances)}>
                    <span className={`collapsible-caret ${showUtterances ? "open" : ""}`}>{"\u25B6"}</span>
                    <span className="collapsible-label">Compare transcripts to ground truth</span>
                </div>
                {showUtterances && (
                    <div className="collapsible-content">
                        <table className="utterance-table">
                            <thead>
                                <tr>
                                    <th className="utt-col-locale">Locale</th>
                                    <th className="utt-col-gt">Ground Truth</th>
                                    <th className="utt-col-sub">{provider.model}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {utterances.map((u) => {
                                    const sub = u.submissions?.[providerId] ?? "";
                                    const matches = sub === u.transcript;
                                    return (
                                        <tr key={u.id}>
                                            <td className="utt-col-locale">
                                                <span className="utterance-locale-badge">{u.locale}</span>
                                            </td>
                                            <td className="utt-col-gt">{u.transcript}</td>
                                            <td className={`utt-col-sub ${matches ? "" : "utt-diff"}`}>
                                                {sub || <span className="no-data">(empty)</span>}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
}

function ProviderDetailLocale({ provider, locale }) {
    return (
        <table className="scores-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {VISIBLE_METRIC_KEYS.map((mk) => {
                    const val = getProviderScore(provider, mk, locale);
                    return (
                        <tr key={mk}>
                            <td>
                                {METRICS[mk].fullLabel} {METRICS[mk].lowerIsBetter ? "\u2193" : "\u2191"}
                            </td>
                            <td>{val !== null ? METRICS[mk].format(val) : "\u2014"}</td>
                        </tr>
                    );
                })}
            </tbody>
        </table>
    );
}

function ProviderDetail({ providerId, providers, locale, onClose }) {
    const provider = providers.find((p) => p.id === providerId);

    if (!provider) return null;

    return (
        <div className="provider-detail-overlay" onClick={onClose}>
            <div className="provider-detail-modal" onClick={(e) => e.stopPropagation()}>
                <div className="provider-detail-header">
                    <h2 className="provider-detail-title">
                        {locale === "overall" ? "Scoring Breakdown" : "Additional Metrics"} for {provider.model}
                    </h2>
                    <button className="provider-detail-close" onClick={onClose}>
                        &times;
                    </button>
                </div>

                {locale === "overall" ? (
                    <ProviderDetailOverall provider={provider} providerId={providerId} />
                ) : (
                    <ProviderDetailLocale provider={provider} locale={locale} />
                )}

                <div className="provider-detail-footer">
                    <a
                        href="https://github.com/sierra-research/mu-bench/blob/main/submissions/SUBMITTING.md"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="provider-detail-link"
                    >
                        Submit your own model &rarr;
                    </a>
                </div>
            </div>
        </div>
    );
}
