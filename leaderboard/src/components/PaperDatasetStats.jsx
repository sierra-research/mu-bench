import "./PaperDatasetStats.css";

const STATS = [
    {
        locale: "en-US",
        flag: "\u{1F1FA}\u{1F1F8}",
        label: "English (US)",
        utterances: 817,
        duration: 55.5,
        avgLen: 4.1,
    },
    {
        locale: "es-MX",
        flag: "\u{1F1F2}\u{1F1FD}",
        label: "Spanish (MX)",
        utterances: 792,
        duration: 60.5,
        avgLen: 4.6,
    },
    { locale: "tr-TR", flag: "\u{1F1F9}\u{1F1F7}", label: "Turkish", utterances: 846, duration: 70.5, avgLen: 5.0 },
    { locale: "vi-VN", flag: "\u{1F1FB}\u{1F1F3}", label: "Vietnamese", utterances: 975, duration: 72.4, avgLen: 4.5 },
    {
        locale: "zh-CN",
        flag: "\u{1F1E8}\u{1F1F3}",
        label: "Chinese (CN)",
        utterances: 840,
        duration: 45.7,
        avgLen: 3.3,
    },
];

const MAX_UTTS = Math.max(...STATS.map((s) => s.utterances));
const MAX_DUR = Math.max(...STATS.map((s) => s.duration));
const TOTAL_UTTS = STATS.reduce((a, s) => a + s.utterances, 0);
const TOTAL_DUR = STATS.reduce((a, s) => a + s.duration, 0);

export default function PaperDatasetStats() {
    return (
        <div className="ds-widget">
            <div className="ds-chart">
                {STATS.map((s) => (
                    <div className="ds-row" key={s.locale}>
                        <div className="ds-locale">
                            <span className="ds-flag">{s.flag}</span>
                            <span className="ds-locale-label">{s.label}</span>
                        </div>
                        <div className="ds-bars">
                            <div className="ds-bar-track">
                                <div
                                    className="ds-bar ds-bar--utts"
                                    style={{ width: `${(s.utterances / MAX_UTTS) * 100}%` }}
                                >
                                    <span className="ds-bar-value">{s.utterances} utterances</span>
                                </div>
                            </div>
                            <div className="ds-bar-track">
                                <div
                                    className="ds-bar ds-bar--dur"
                                    style={{ width: `${(s.duration / MAX_DUR) * 100}%` }}
                                >
                                    <span className="ds-bar-value">{s.duration} min</span>
                                </div>
                            </div>
                        </div>
                        <div className="ds-avg">{s.avgLen}s avg</div>
                    </div>
                ))}
            </div>
            <div className="ds-totals">
                <span className="ds-total-item">
                    <strong>{TOTAL_UTTS.toLocaleString()}</strong> utterances
                </span>
                <span className="ds-total-sep">/</span>
                <span className="ds-total-item">
                    <strong>{TOTAL_DUR.toFixed(1)}</strong> min ({(TOTAL_DUR / 60).toFixed(1)} hrs)
                </span>
                <span className="ds-total-sep">/</span>
                <span className="ds-total-item">
                    <strong>250</strong> conversations
                </span>
            </div>
        </div>
    );
}
