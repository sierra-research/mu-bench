import { useState, useRef } from "react";
import "./PaperCoverageMatrix.css";

const ALL_LOCALES = [
    { code: "ar-AE", status: "completed" },
    { code: "ar-BH", status: "completed" },
    { code: "ar-KW", status: "completed" },
    { code: "ar-QA", status: "completed" },
    { code: "ar-SA", status: "completed" },
    { code: "az-AZ", status: "completed" },
    { code: "bg-BG", status: "completed" },
    { code: "bn-BD", status: "completed" },
    { code: "bn-IN", status: "completed" },
    { code: "ca-ES", status: "completed" },
    { code: "cs-CZ", status: "completed" },
    { code: "da-DK", status: "completed" },
    { code: "de-AT", status: "completed" },
    { code: "de-CH", status: "completed" },
    { code: "de-DE", status: "completed" },
    { code: "de-LU", status: "ongoing" },
    { code: "el-GR", status: "completed" },
    { code: "en-AU", status: "completed" },
    { code: "en-GB", status: "completed" },
    { code: "en-GB-birmingham", status: "completed" },
    { code: "en-GB-irish-english", status: "completed" },
    { code: "en-GB-liverpool", status: "completed" },
    { code: "en-GB-london", status: "completed" },
    { code: "en-GB-manchester", status: "completed" },
    { code: "en-GB-midlands", status: "completed" },
    { code: "en-GB-newcastle", status: "completed" },
    { code: "en-GB-northern-england", status: "completed" },
    { code: "en-GB-northern-irish", status: "completed" },
    { code: "en-GB-scottish", status: "completed" },
    { code: "en-GB-southern-english", status: "completed" },
    { code: "en-GB-wales", status: "completed" },
    { code: "en-GB-yorkshire", status: "completed" },
    { code: "en-NZ", status: "completed" },
    { code: "en-SG", status: "completed" },
    { code: "en-US", status: "completed", open: true },
    { code: "es-CO", status: "ongoing" },
    { code: "es-ES", status: "completed" },
    { code: "es-MX", status: "completed", open: true },
    { code: "es-US", status: "completed" },
    { code: "eu-ES", status: "completed" },
    { code: "fi-FI", status: "completed" },
    { code: "fil-PH", status: "completed" },
    { code: "fr-BE", status: "completed" },
    { code: "fr-CA", status: "completed" },
    { code: "fr-CH", status: "ongoing" },
    { code: "fr-FR", status: "completed" },
    { code: "fr-LU", status: "ongoing" },
    { code: "gl-ES", status: "completed" },
    { code: "he-IL", status: "completed" },
    { code: "hi-IN", status: "completed" },
    { code: "hr-HR", status: "completed" },
    { code: "hu-HU", status: "completed" },
    { code: "id-ID", status: "completed" },
    { code: "is-IS", status: "ongoing" },
    { code: "it-CH", status: "ongoing" },
    { code: "it-IT", status: "completed" },
    { code: "ja-JP", status: "completed" },
    { code: "ko-KR", status: "completed" },
    { code: "lt-LT", status: "ongoing" },
    { code: "lv-LV", status: "completed" },
    { code: "ms-MY", status: "completed" },
    { code: "nb-NO", status: "completed" },
    { code: "nl-BE", status: "ongoing" },
    { code: "nl-NL", status: "completed" },
    { code: "pl-PL", status: "completed" },
    { code: "pt-BR", status: "completed" },
    { code: "pt-PT", status: "completed" },
    { code: "ro-RO", status: "completed" },
    { code: "ru-KZ", status: "completed" },
    { code: "ru-RU", status: "completed" },
    { code: "sk-SK", status: "ongoing" },
    { code: "sv-SE", status: "completed" },
    { code: "th-TH", status: "completed" },
    { code: "tl-PH", status: "completed" },
    { code: "tr-TR", status: "completed", open: true },
    { code: "uk-UA", status: "completed" },
    { code: "vi-VN", status: "completed", open: true },
    { code: "zh-CN", status: "completed", open: true },
    { code: "zh-HK", status: "completed" },
];

const PROVIDERS = [
    { id: "deepgram", label: "Deepgram" },
    { id: "google", label: "Google" },
    { id: "azure", label: "Azure" },
    { id: "speechmatics", label: "Speechmatics" },
    { id: "fano", label: "Fano" },
    { id: "openai", label: "OpenAI" },
    { id: "elevenlabs", label: "ElevenLabs" },
    { id: "assemblyai", label: "AssemblyAI" },
    { id: "gemini", label: "Gemini" },
    { id: "voxtral", label: "Voxtral" },
    { id: "parakeet", label: "Parakeet" },
    { id: "amivoice", label: "AmiVoice" },
    { id: "gpt-realtime", label: "GPT Realtime" },
    { id: "kotoba", label: "Kotoba" },
];

const OPEN_PROVIDERS = new Set(["deepgram", "google", "azure", "openai", "elevenlabs"]);

const SIGWER_SCORES = {
    "en-US": { deepgram: 0.053, google: 0.056, azure: 0.039, openai: 0.062, elevenlabs: 0.062 },
    "es-MX": { deepgram: 0.158, google: 0.09, azure: 0.134, openai: 0.129, elevenlabs: 0.13 },
    "tr-TR": { deepgram: 0.183, google: 0.101, azure: 0.231, openai: 0.181, elevenlabs: 0.124 },
    "vi-VN": { deepgram: 0.576, google: 0.11, azure: 0.289, openai: 0.274, elevenlabs: 0.226 },
    "zh-CN": { deepgram: 0.549, google: 0.365, azure: 0.418, openai: 0.436, elevenlabs: 0.429 },
};

function getScoreColor(val) {
    const t = Math.min(Math.max(val / 0.55, 0), 1);
    const r = Math.round(34 + t * 200);
    const g = Math.round(160 - t * 100);
    const b = Math.round(70 - t * 40);
    return `rgb(${r}, ${g}, ${b})`;
}

// Total speaker estimates per base language (L1+L2, millions, Ethnologue 2025)
const LANG_TOTAL_SPEAKERS_M = {
    ar: 380,
    az: 33,
    bg: 9,
    bn: 300,
    ca: 10,
    cs: 13,
    da: 6,
    de: 135,
    el: 13,
    en: 1500,
    es: 560,
    eu: 1,
    fi: 6,
    fil: 82,
    fr: 310,
    gl: 3,
    he: 10,
    hi: 600,
    hr: 7,
    hu: 13,
    id: 275,
    is: 0.4,
    it: 85,
    ja: 125,
    ko: 82,
    lt: 3,
    lv: 2,
    ms: 200,
    nb: 5,
    nl: 30,
    pl: 45,
    pt: 265,
    ro: 26,
    ru: 260,
    sk: 5,
    sv: 13,
    th: 61,
    tl: 28,
    tr: 88,
    uk: 45,
    vi: 86,
    zh: 1120,
};
const WORLD_POP_M = 8100;

function computeSpeakerCoverage() {
    const seen = new Set();
    let total = 0;
    for (const loc of ALL_LOCALES) {
        const lang = loc.code.split("-")[0];
        if (seen.has(lang)) continue;
        // fil/tl overlap, ms/id overlap -- skip duplicates
        if (lang === "tl" && seen.has("fil")) continue;
        if (lang === "fil" && seen.has("tl")) continue;
        if (lang === "ms" && seen.has("id")) continue;
        if (lang === "id" && seen.has("ms")) continue;
        seen.add(lang);
        total += LANG_TOTAL_SPEAKERS_M[lang] || 0;
    }
    return Math.round((total / WORLD_POP_M) * 100);
}

const SPEAKER_COVERAGE_PCT = computeSpeakerCoverage();

const SORTED_LOCALES = [...ALL_LOCALES.filter((l) => l.open), ...ALL_LOCALES.filter((l) => !l.open)];

const SORTED_PROVIDERS = [
    ...PROVIDERS.filter((p) => OPEN_PROVIDERS.has(p.id)),
    ...PROVIDERS.filter((p) => !OPEN_PROVIDERS.has(p.id)),
];
const CELL = 7;
const GAP = 1.5;

export default function PaperCoverageMatrix() {
    const [tooltip, setTooltip] = useState(null);
    const wrapRef = useRef(null);

    const totalCells = ALL_LOCALES.length * PROVIDERS.length;
    const openCells = ALL_LOCALES.filter((l) => l.open).length * OPEN_PROVIDERS.size;

    const gridW = SORTED_LOCALES.length * (CELL + GAP) - GAP;
    const gridH = SORTED_PROVIDERS.length * (CELL + GAP) - GAP;

    function handleCellEnter(e, locale, provider) {
        const rect = wrapRef.current.getBoundingClientRect();
        const cellRect = e.currentTarget.getBoundingClientRect();
        const score = SIGWER_SCORES[locale.code]?.[provider.id];
        setTooltip({
            x: cellRect.left - rect.left + cellRect.width / 2,
            y: cellRect.top - rect.top - 6,
            locale: locale.code,
            provider: provider.label,
            score,
            isOpen: locale.open && OPEN_PROVIDERS.has(provider.id),
        });
    }

    return (
        <div className="cm-widget" ref={wrapRef}>
            <div className="cm-layout">
                <div className="cm-grid-side">
                    <svg className="cm-grid-svg" viewBox={`0 0 ${gridW} ${gridH}`}>
                        {SORTED_PROVIDERS.map((prov, pi) =>
                            SORTED_LOCALES.map((loc, li) => {
                                const isOpen = loc.open && OPEN_PROVIDERS.has(prov.id);
                                const score = isOpen ? SIGWER_SCORES[loc.code]?.[prov.id] : null;
                                const x = li * (CELL + GAP);
                                const y = pi * (CELL + GAP);
                                return (
                                    <rect
                                        key={`${loc.code}-${prov.id}`}
                                        x={x}
                                        y={y}
                                        width={CELL}
                                        height={CELL}
                                        rx={1}
                                        fill={score != null ? getScoreColor(score) : "#eaedf0"}
                                        stroke={isOpen ? "rgba(16,185,129,0.5)" : "none"}
                                        strokeWidth={isOpen ? 0.5 : 0}
                                        className="cm-pixel"
                                        onMouseEnter={(e) => handleCellEnter(e, loc, prov)}
                                        onMouseLeave={() => setTooltip(null)}
                                    />
                                );
                            }),
                        )}
                    </svg>
                    <div className="cm-axis-row">
                        <span className="cm-axis-label">
                            {ALL_LOCALES.length} locales &times; {PROVIDERS.length} providers
                        </span>
                        <span className="cm-legend">
                            <span className="cm-legend-item">
                                <span className="cm-legend-swatch cm-legend-swatch--open" />
                                Open-sourced
                            </span>
                            <span className="cm-legend-item">
                                <span className="cm-legend-swatch cm-legend-swatch--closed" />
                                Internal only
                            </span>
                        </span>
                    </div>
                </div>
                <div className="cm-info-side">
                    <div className="cm-stats-row">
                        <div className="cm-stat">
                            <span className="cm-stat-num">{totalCells.toLocaleString()}</span>
                            <span className="cm-stat-label">pairs evaluated internally</span>
                        </div>
                        <span className="cm-stat-sep">/</span>
                        <div className="cm-stat cm-stat--highlight">
                            <span className="cm-stat-num">{openCells}</span>
                            <span className="cm-stat-label">open-sourced</span>
                        </div>
                        <span className="cm-stat-sep">/</span>
                        <div className="cm-stat">
                            <span className="cm-stat-num">~{SPEAKER_COVERAGE_PCT}%</span>
                            <span className="cm-stat-label">of world population covered</span>
                        </div>
                    </div>
                </div>
            </div>
            {tooltip && (
                <div className="cm-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
                    <div className="cm-tooltip-locale">{tooltip.locale}</div>
                    <div className="cm-tooltip-provider">{tooltip.provider}</div>
                    {tooltip.isOpen && tooltip.score != null && (
                        <div className="cm-tooltip-score">UER: {(tooltip.score * 100).toFixed(1)}%</div>
                    )}
                </div>
            )}
        </div>
    );
}
