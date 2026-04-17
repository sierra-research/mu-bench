export const METRICS = {
    wer: {
        key: "wer",
        label: "WER",
        fullLabel: "Word Error Rate",
        lowerIsBetter: true,
        format: (v) => `${(v * 100).toFixed(1)}%`,
        unit: "%",
        description:
            "Corpus Word Error Rate \u2014 total word edits across the locale divided by total reference words. Overall is the unweighted mean of the five per-locale corpus WERs.",
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
            "95th percentile (p95) time to complete transcript, per utterance, in milliseconds. For batch providers this is request-to-response round-trip; for streaming providers it is send to final transcript. Median (p50) latency is also recorded but p95 better reflects worst-case production performance.",
    },
};

/**
 * Return the unified sortable latency p95 for a provider/locale combo,
 * preferring the new `completeP95Ms` field (batch + streaming) and
 * falling back to the legacy `latencyP95Ms` when a score.json hasn't
 * been re-scored under the new latency schema yet.
 */
export function pickLatencyP95(localeResult) {
    if (!localeResult) return null;
    const c = localeResult.completeP95Ms;
    if (c !== null && c !== undefined) return c;
    const legacy = localeResult.latencyP95Ms;
    return legacy !== null && legacy !== undefined ? legacy : null;
}

/**
 * Return the streaming TTFT p95 when the provider reports streaming
 * measurements; null otherwise. Used to render the `+TTFT` annotation
 * on streaming rows without influencing the sort.
 */
export function pickTtftP95(localeResult) {
    if (!localeResult) return null;
    const t = localeResult.ttftP95Ms;
    return t !== null && t !== undefined ? t : null;
}

export const METRIC_KEYS = Object.keys(METRICS);

export const VISIBLE_METRIC_KEYS = METRIC_KEYS;

export function getSortDirection(metricKey) {
    return METRICS[metricKey].lowerIsBetter ? "asc" : "desc";
}
