import { useState } from "react";
import "./PaperErrorExamples.css";

const EXAMPLES = [
    {
        id: "surface-level",
        title: "Surface-Level Errors (High WER, Low UER)",
        description:
            "Filler words and rephrasing inflate WER significantly, but the caller\u2019s intent and every key detail are preserved.",
        gold: "I\u2019d like to dispute a charge on my account for fifty dollars.",
        predictions: {
            "Provider A": "Uh, so I\u2019d like to, um, dispute a charge on my account for, like, fifty dollars, yeah.",
            "Provider B": "Oh, I\u2019d like to, uh, dispute like a charge on my account for fifty dollars I think.",
        },
        locale: "en-US",
    },
    {
        id: "significant",
        title: "Significant Errors (Low WER, High UER)",
        description:
            "Only one word differs in each transcription, but the person\u2019s name is wrong \u2014 a single substitution that changes meaning entirely.",
        gold: "The name on the account is Mason Lee.",
        predictions: {
            "Provider A": "The name on the account is Jason Lee.",
            "Provider B": "The name on the account is Mason Li.",
        },
        locale: "en-US",
    },
];

function diffWords(gold, pred) {
    const gTokens = gold.split(/(\s+)/);
    const pTokens = pred.split(/(\s+)/);
    const m = gTokens.length;
    const n = pTokens.length;

    const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = m - 1; i >= 0; i--)
        for (let j = n - 1; j >= 0; j--)
            dp[i][j] = gTokens[i] === pTokens[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);

    const result = [];
    let i = 0,
        j = 0;
    while (i < m || j < n) {
        if (i < m && j < n && gTokens[i] === pTokens[j]) {
            result.push({ text: pTokens[j], type: "match" });
            i++;
            j++;
        } else if (j < n && (i >= m || dp[i][j + 1] >= dp[i + 1][j])) {
            result.push({ text: pTokens[j], type: "insert" });
            j++;
        } else {
            result.push({ text: gTokens[i], type: "delete" });
            i++;
        }
    }
    return result;
}

function DiffView({ gold, predicted }) {
    const diff = diffWords(gold, predicted);
    return (
        <div className="err-diff">
            <div className="err-diff-row">
                <span className="err-diff-label">Gold</span>
                <span className="err-diff-text">{gold}</span>
            </div>
            <div className="err-diff-row">
                <span className="err-diff-label">Predicted</span>
                <span className="err-diff-text">
                    {diff.map((d, i) =>
                        d.type === "match" ? (
                            <span key={i}>{d.text}</span>
                        ) : d.type === "insert" ? (
                            <mark key={i} className="err-ins">
                                {d.text}
                            </mark>
                        ) : d.type === "delete" ? (
                            <mark key={i} className="err-del">
                                {d.text}
                            </mark>
                        ) : (
                            <mark key={i} className="err-sub" title={`was: ${d.original}`}>
                                {d.text}
                            </mark>
                        ),
                    )}
                </span>
            </div>
        </div>
    );
}

function ExampleCard({ example }) {
    const providers = Object.keys(example.predictions);
    const [providerIdx, setProviderIdx] = useState(0);

    return (
        <div className="err-section">
            <h4 className="err-title">{example.title}</h4>
            <p className="err-desc">{example.description}</p>
            <div className="err-widget">
                <div className="err-provider-tabs">
                    {providers.map((p, i) => (
                        <button
                            key={p}
                            className={`err-tab ${i === providerIdx ? "err-tab--active" : ""}`}
                            onClick={() => setProviderIdx(i)}
                        >
                            {p}
                        </button>
                    ))}
                </div>
                <div className="err-body">
                    <DiffView gold={example.gold} predicted={example.predictions[providers[providerIdx]]} />
                </div>
            </div>
        </div>
    );
}

export default function PaperErrorExamples() {
    return (
        <>
            <p className="err-see-below">See example below.</p>
            {EXAMPLES.map((ex) => (
                <ExampleCard key={ex.id} example={ex} />
            ))}
        </>
    );
}
