import { useState } from "react";
import "./PaperErrorExamples.css";

const EXAMPLES = [
    {
        id: "cosmetic",
        title: "Cosmetic Errors (High WER, Low UER)",
        description:
            "Filler words and minor verb-form differences inflate WER but preserve the caller\u2019s name and intent.",
        gold: "Okay. Hopefully, I get all that information for you. Let\u2019s see here. My full name for this account would be John Brown.",
        predictions: {
            Deepgram:
                "Oh, okay. Hopefully, I got all that information for you. Let\u2019s see here. My full name for this account would be John Brown.",
            ElevenLabs:
                "Uh, okay. Hopefully, I get all that information for you. Let\u2019s see here. Uh, my full name for this account would be John Brown.",
            Google: "Okay. Hopefully, I got all that information for you. Let\u2019s see here. My full name for this account would be John Brown.",
            Azure: "Okay. Hopefully, I got all that information for you. Let\u2019s see here. My full name for this account would be John Brown.",
        },
        locale: "en-US",
    },
    {
        id: "significant",
        title: "Significant Errors (Low WER, High UER)",
        description:
            "Only one or two characters differ, but they change the person\u2019s name entirely \u2014 \u7F8E\u2192\u6885, \u73B2\u2192\u94C3.",
        gold: "\u662F\u7F8E\u70B9\u73B2 at email dot com\u3002",
        predictions: {
            Deepgram: "\u662F\u6885\u70B9\u94C3 at email dot com.",
            ElevenLabs: "\u662F\u7F8E\u70B9\u73B2 at email dot com.",
            Google: "\u662F\u6CA1\u70B9\u96F6 at email dot com.",
            Azure: "She made Dian Ling at email dot com.",
        },
        locale: "zh-CN",
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
