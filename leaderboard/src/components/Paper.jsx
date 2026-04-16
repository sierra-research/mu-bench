import { useState, useEffect } from "react";
import Markdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";
import PaperAudioSamples from "./PaperAudioSamples.jsx";
import PaperDatasetStats from "./PaperDatasetStats.jsx";
import PaperErrorExamples from "./PaperErrorExamples.jsx";
import PaperHeatmap from "./PaperHeatmap.jsx";
import PaperRadar from "./PaperRadar.jsx";
import PaperGlobe from "./PaperGlobe.jsx";
import PaperCoverageMatrix from "./PaperCoverageMatrix.jsx";
import PaperSignificance from "./PaperSignificance.jsx";
import "./Paper.css";

const WIDGETS = {
    "audio-samples": PaperAudioSamples,
    "dataset-stats": PaperDatasetStats,
    "error-examples": PaperErrorExamples,
    heatmap: PaperHeatmap,
    radar: PaperRadar,
    globe: PaperGlobe,
    "coverage-matrix": PaperCoverageMatrix,
    significance: PaperSignificance,
};

const WIDGET_RE = /<!-- widget:(\S+) -->/;

function slugify(text) {
    return text
        .toLowerCase()
        .replace(/[^\w\s-]/g, "")
        .replace(/\s+/g, "-");
}

function HeadingWithId({ node, children, ...props }) {
    const Tag = `h${node.tagName.charAt(1)}`;
    const text = typeof children === "string" ? children : String(children);
    return (
        <Tag id={slugify(text)} {...props}>
            {children}
        </Tag>
    );
}

const MD_COMPONENTS = {
    h1: HeadingWithId,
    h2: HeadingWithId,
    h3: HeadingWithId,
    h4: HeadingWithId,
};

function splitByWidgets(md) {
    const lines = md.split("\n");
    const sections = [];
    let buf = [];

    for (const line of lines) {
        const match = line.match(WIDGET_RE);
        if (match) {
            if (buf.length) {
                sections.push({ type: "md", content: buf.join("\n") });
                buf = [];
            }
            sections.push({ type: "widget", id: match[1] });
        } else {
            buf.push(line);
        }
    }
    if (buf.length) {
        sections.push({ type: "md", content: buf.join("\n") });
    }
    return sections;
}

export default function Paper() {
    const [content, setContent] = useState("");

    useEffect(() => {
        fetch(`${import.meta.env.BASE_URL}paper.md`)
            .then((r) => r.text())
            .then(setContent)
            .catch(() => setContent("Failed to load paper content."));
    }, []);

    if (!content) {
        return <div className="paper-loading">Loading...</div>;
    }

    const sections = splitByWidgets(content);

    return (
        <article className="paper">
            {sections.map((section, i) => {
                if (section.type === "widget") {
                    const Widget = WIDGETS[section.id];
                    return Widget ? <Widget key={i} /> : null;
                }
                return (
                    <Markdown
                        key={i}
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeRaw]}
                        components={MD_COMPONENTS}
                    >
                        {section.content}
                    </Markdown>
                );
            })}
        </article>
    );
}
