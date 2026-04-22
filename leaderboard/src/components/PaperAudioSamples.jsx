import { useState, useRef, useEffect } from "react";
import sampleEnUS from "../samples/en-US/conv-39-turn-8.wav";
import sampleEsMX from "../samples/es-MX/conv-49-turn-5.wav";
import sampleTrTR from "../samples/tr-TR/conv-41-turn-11.wav";
import sampleViVN from "../samples/vi-VN/conv-20-turn-14.wav";
import sampleZhCN from "../samples/zh-CN/conv-35-turn-1.wav";
import "./PaperAudioSamples.css";

const SAMPLES = [
    {
        locale: "en-US",
        flag: "\u{1F1FA}\u{1F1F8}",
        label: "English (US)",
        src: sampleEnUS,
        transcript: "It\u2019d be under my name, Ashley Brown. And do you want my email address?",
    },
    {
        locale: "es-MX",
        flag: "\u{1F1F2}\u{1F1FD}",
        label: "Spanish (MX)",
        src: sampleEsMX,
        transcript: "Es Juan punto L\u00f3pez arroba email punto com.",
    },
    {
        locale: "tr-TR",
        flag: "\u{1F1F9}\u{1F1F7}",
        label: "Turkish",
        src: sampleTrTR,
        transcript: "Tamam, dinlersen s\u00f6yleyece\u011fim Anna. Art\u0131 90 123 0 90 45 67.",
    },
    {
        locale: "vi-VN",
        flag: "\u{1F1FB}\u{1F1F3}",
        label: "Vietnamese",
        src: sampleViVN,
        transcript:
            "Kh\u00f4ng ph\u1ea3i. Minh, minh l\u00e0 t\u00ean \u0111\u00f3. Tr\u1ea7n l\u00e0 h\u1ecd \u0111\u00f3. Minh ch\u1ea5m tr\u1ea7n a c\u00f2ng email ch\u1ea5m com.",
    },
    {
        locale: "zh-CN",
        flag: "\u{1F1E8}\u{1F1F3}",
        label: "Chinese (CN)",
        src: sampleZhCN,
        transcript: "\u6211\u60f3\u54a8\u8be2\u672a\u6210\u5e74\u80fd\u529e\u5361\u5417\uff1f",
    },
];

export default function PaperAudioSamples() {
    const [active, setActive] = useState(0);
    const [playing, setPlaying] = useState(false);
    const audioRef = useRef(null);
    const canvasRef = useRef(null);
    const audioContextRef = useRef(null);
    const analyserRef = useRef(null);
    const animFrameRef = useRef(null);

    const sample = SAMPLES[active];

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

        if (!playing) {
            for (let i = 0; i < barCount; i++) {
                ctx.fillStyle = "#d1d5db";
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

                ctx.fillStyle = "#374151";
                ctx.fillRect(x, y, barW, barH);
            }
        }

        animFrameRef.current = requestAnimationFrame(draw);
        return () => {
            if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
        };
    }, [playing]);

    const toggle = () => {
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

        if (playing) {
            audio.pause();
        } else {
            audio.play();
        }
    };

    const switchLocale = (i) => {
        if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current.currentTime = 0;
        }
        setPlaying(false);
        setActive(i);
    };

    return (
        <>
            <div className="as-widget">
                <div className="as-tabs">
                    {SAMPLES.map((s, i) => (
                        <button
                            key={s.locale}
                            className={`as-tab ${i === active ? "as-tab--on" : ""}`}
                            onClick={() => switchLocale(i)}
                        >
                            <span className="as-tab-flag">{s.flag}</span>
                            <span className="as-tab-label">{s.label}</span>
                        </button>
                    ))}
                </div>
                <div className="as-body">
                    <audio
                        ref={audioRef}
                        src={sample.src}
                        onPlay={() => setPlaying(true)}
                        onPause={() => setPlaying(false)}
                        onEnded={() => setPlaying(false)}
                    />
                    <div className={`as-player ${playing ? "as-player--on" : ""}`}>
                        <button className="as-play" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
                            {playing ? "\u23F8" : "\u25B6"}
                        </button>
                        <canvas className="as-canvas" ref={canvasRef} />
                    </div>
                    <p className={`as-transcript${sample.locale === "zh-CN" ? " as-transcript--zh" : ""}`}>
                        {sample.transcript}
                    </p>
                </div>
            </div>
            <p className="as-caption">Sample utterances from each locale.</p>
        </>
    );
}
