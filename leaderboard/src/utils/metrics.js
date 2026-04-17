export const METRICS = {
    wer: {
        key: "wer",
        label: "WER",
        fullLabel: "Word Error Rate",
        lowerIsBetter: true,
        format: (v) => `${(v * 100).toFixed(1)}%`,
        unit: "%",
        description: "Word Error Rate \u2014 percentage of words incorrectly transcribed",
    },
    significantWer: {
        key: "significantWer",
        label: "UER",
        fullLabel: "Utterance Error Rate",
        lowerIsBetter: true,
        format: (v) => `${(v * 100).toFixed(1)}%`,
        unit: "%",
        description:
            "Utterance Error Rate \u2014 fraction of utterances containing at least one meaning-changing error",
    },
    latencyP95Ms: {
        key: "latencyP95Ms",
        label: "Latency",
        fullLabel: "Latency (p95)",
        lowerIsBetter: true,
        format: (v) => `${Math.round(v)} ms`,
        unit: "ms",
        description:
            "95th percentile (p95) API response time per utterance in milliseconds. Median (p50) latency is also recorded but p95 better reflects worst-case production performance.",
    },
};

export const METRIC_KEYS = Object.keys(METRICS);

export const VISIBLE_METRIC_KEYS = METRIC_KEYS;

export function getSortDirection(metricKey) {
    return METRICS[metricKey].lowerIsBetter ? "asc" : "desc";
}
