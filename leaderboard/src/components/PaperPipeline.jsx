import { useState, useRef } from "react";
import sampleTrTR from "../samples/tr-TR/conv-41-turn-11.wav";
import sampleViVN from "../samples/vi-VN/conv-20-turn-14.wav";
import sampleZhCN from "../samples/zh-CN/conv-4-turn-12.wav";
import "./PaperPipeline.css";

const EXAMPLES = [
    {
        locale: "zh-CN",
        flag: "\u{1F1E8}\u{1F1F3}",
        label: "Name correction (Mandarin)",
        src: sampleZhCN,
        gold: "\u4f60\u8fd8\u6ca1\u542c\u5230\u6211\u7684\u6b63\u786e\u7684\u59d3\u540d\uff0c\u6211\u7684\u59d3\u540d\u662f\u541b\u7075\u3002",
        providers: [
            {
                name: "Google Chirp-3",
                text: "\u4f60\u8fd8\u6ca1\u542c\u5230\u6211\u7684\u6b63\u786e\u7684\u59d3\u540d\uff0c\u6211\u7684\u59d3\u540d\u662f\u6da6\u9716\u3002",
            },
            {
                name: "Azure",
                text: "\u4f60\u8fd8\u6ca1\u542c\u5230\u6211\u7684\u6b63\u786e\u7684\u59d3\u540d\uff0c\u6211\u7684\u59d3\u540d\u662f\u541b\u73b2\u3002",
            },
            {
                name: "ElevenLabs",
                text: "\u4f60\u8fd8\u6ca1\u542c\u5230\u6211\u7684\u6b63\u786e\u7684\u59d3\u540d\u3002\u6211\u7684\u59d3\u540d\u662f\u5a1f\u73b2\u3002",
            },
            {
                name: "OpenAI",
                text: "\u4f60\u9084\u6c92\u807d\u5230\u6211\u7684\u6b63\u5f0f\u7684\u59d3\u540d\u3002\u6211\u7684\u59d3\u540d\u662fDream\u73b2\u3002",
            },
            {
                name: "Deepgram",
                text: "\u4f60\u8fd8\u6ca1\u542c\u5230\u6211\u7684\u6b63\u786e\u7684\u59d3\u540d\u6211\u7684\u5211\u547d\u662fdreamlin",
            },
        ],
        note: "The caller says their name is \u541b\u7075 (J\u016bn L\u00edng). Every provider gets the sentence structure right but the name wrong \u2014 five different guesses, all incorrect. This is a significant error: the downstream system records the wrong identity.",
    },
    {
        locale: "tr-TR",
        flag: "\u{1F1F9}\u{1F1F7}",
        label: "Phone number (Turkish)",
        src: sampleTrTR,
        gold: "Tamam, dinlersen s\u00f6yleyece\u011fim Anna. Art\u0131 90 123 0 90 45 67.",
        providers: [
            { name: "Google Chirp-3", text: "Tamam, dinler misin? Payla\u015fam\u0131yorum ben. +90 123 090 45 67" },
            { name: "ElevenLabs", text: "Tamam, dinlersen payla\u015faca\u011f\u0131m ama. +90 123 090 45 67" },
            {
                name: "Deepgram",
                text: "Tamam, dinlersin payla\u015faca\u011f\u0131m ondan. Art\u0131 doksan, y\u00fcz yirmi \u00fc\u00e7, s\u0131f\u0131r doksan, k\u0131rk be\u015f, altm\u0131\u015f yedi.",
            },
            {
                name: "OpenAI",
                text: "Tamam, dinle a\u00e7\u0131m payla\u015fay\u0131m onlar. Art\u0131 doksan y\u00fcz yirmi \u00fc\u00e7 s\u0131f\u0131r doksan k\u0131rk be\u015f altm\u0131\u015f yedi.",
            },
            { name: "Azure", text: "Tamam, binlerce mallar+ 90 123 s\u0131f\u0131r, 90 45 67." },
        ],
        note: "The caller dictates a phone number. Some providers write digits, others spell them out in Turkish \u2014 both are valid. But the conversational preamble is garbled by most: \u201Cdinlersen s\u00f6yleyece\u011fim Anna\u201D becomes \u201Cbinlerce mallar\u201D (Azure) or \u201Cdinle a\u00e7\u0131m payla\u015fay\u0131m onlar\u201D (OpenAI).",
    },
    {
        locale: "vi-VN",
        flag: "\u{1F1FB}\u{1F1F3}",
        label: "Email dictation (Vietnamese)",
        src: sampleViVN,
        gold: "Kh\u00f4ng ph\u1ea3i. Minh, minh l\u00e0 t\u00ean \u0111\u00f3. Tr\u1ea7n l\u00e0 h\u1ecd \u0111\u00f3. Minh ch\u1ea5m tr\u1ea7n a c\u1ed9ng email ch\u1ea5m com.",
        providers: [
            {
                name: "ElevenLabs",
                text: "Kh\u00f4ng ph\u1ea3i. Minh, Minh l\u00e0 t\u00ean \u0111\u00f3, Tr\u1ea7n l\u00e0 h\u1ecd \u0111\u00f3. minh.tran@email.com",
            },
            {
                name: "Google Chirp-3",
                text: "Kh\u00f4ng ph\u1ea3i, Minh Minh l\u00e0 t\u00ean \u0111\u00f3, Tr\u1ea7n l\u00e0 h\u1ecd \u0111\u00f3, Minh ch\u1ea5m Tr\u1ea7n a c\u00f2ng email ch\u1ea5m com.",
            },
            {
                name: "OpenAI",
                text: "Kh\u00f4ng ph\u1ea3i, Minh l\u00e0 t\u00ean \u0111\u00f3, Tr\u1ea7n l\u00e0 h\u1ecd \u0111\u00f3, minh.tran@gmail.com.",
            },
            {
                name: "Deepgram",
                text: "Kh\u00f4ng ph\u1ea3i Minh Minh l\u00e0 t\u00ean \u0111\u00f3 Tr\u1ea7n l\u00e0 h\u1ecd \u0111\u00f3 m\u00ecnh ch\u1ea5m tr\u1ea7ngmail ch\u1ea5m com",
            },
            {
                name: "Azure",
                text: "Kh\u00f4ng ph\u1ea3i minh minh l\u00e0 t\u00ean \u0111\u00f3 tr\u1ea7n l\u00e0 h\u1ecd \u0111\u00f3 minh. Tr\u1ea7n a. C\u00f2ngemail.com.",
            },
        ],
        note: "The caller spells out their email: \u201Cminh ch\u1ea5m tr\u1ea7n a c\u1ed9ng email ch\u1ea5m com\u201D (minh dot tran at email dot com). ElevenLabs reconstructs it as minh.tran@email.com. OpenAI hallucinates gmail.com. Azure and Deepgram garble the punctuation.",
    },
];

export default function PaperPipeline() {
    const [active, setActive] = useState(0);
    const [playing, setPlaying] = useState(false);
    const audioRef = useRef(null);
    const ex = EXAMPLES[active];

    const toggle = () => {
        if (!audioRef.current) return;
        if (playing) audioRef.current.pause();
        else audioRef.current.play();
    };

    const switchExample = (i) => {
        if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current.currentTime = 0;
        }
        setPlaying(false);
        setActive(i);
    };

    return (
        <div className="pl-widget">
            <div className="pl-tabs">
                {EXAMPLES.map((e, i) => (
                    <button
                        key={e.locale}
                        className={`pl-tab ${i === active ? "pl-tab--on" : ""}`}
                        onClick={() => switchExample(i)}
                    >
                        <span className="pl-tab-flag">{e.flag}</span>
                        <span className="pl-tab-label">{e.label}</span>
                    </button>
                ))}
            </div>
            <div className="pl-body">
                <audio
                    ref={audioRef}
                    src={ex.src}
                    onPlay={() => setPlaying(true)}
                    onPause={() => setPlaying(false)}
                    onEnded={() => setPlaying(false)}
                />
                <div className="pl-gold">
                    <div className="pl-row-header">
                        <span className="pl-badge pl-badge--gold">Ground truth</span>
                        <button
                            className={`pl-play ${playing ? "pl-play--on" : ""}`}
                            onClick={toggle}
                            aria-label={playing ? "Pause" : "Play"}
                        >
                            {playing ? "\u23F8" : "\u25B6\uFE0E"} Listen
                        </button>
                    </div>
                    <p className="pl-transcript pl-transcript--gold">{ex.gold}</p>
                </div>
                <div className="pl-providers">
                    {ex.providers.map((p) => (
                        <div key={p.name} className="pl-provider">
                            <span className="pl-badge">{p.name}</span>
                            <p className="pl-transcript">{p.text}</p>
                        </div>
                    ))}
                </div>
                <p className="pl-note">{ex.note}</p>
            </div>
        </div>
    );
}
