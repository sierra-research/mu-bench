"""Paired bootstrap significance tests for provider ranking comparisons.

Loads per-utterance detail files from results/<provider>/details/<locale>/,
runs paired bootstrap resampling to determine whether ranking differences
between providers are statistically significant.

Usage:
    python scripts/significance_test.py
    python scripts/significance_test.py --metric significantWer --locales en-US zh-CN
    python scripts/significance_test.py --iterations 50000
"""

import argparse
import json
from pathlib import Path

import numpy as np

TARGET_LOCALES = ["en-US", "es-MX", "tr-TR", "vi-VN", "zh-CN"]

PROVIDERS = [
    "deepgram-nova3",
    "azure",
    "google-chirp3",
    "openai-gpt4o-mini-transcribe",
    "elevenlabs-scribe-v2",
]

SHORT_NAMES = {
    "deepgram-nova3": "Deepgram",
    "azure": "Azure",
    "google-chirp3": "Google",
    "openai-gpt4o-mini-transcribe": "OpenAI",
    "elevenlabs-scribe-v2": "ElevenLabs",
}


def load_utterance_scores(
    results_dir: Path, provider: str, locales: list[str], metric: str
) -> dict[str, tuple[float, float]]:
    """Load per-utterance (numerator, denominator) pairs for a provider.

    Returns ``{utterance_key: (num, denom)}`` where ``utterance_key`` is
    ``"<locale>/<utterance_id>"``. Both metrics are stored as a (num, denom)
    pair so the caller can aggregate by sum-of-num / sum-of-denom under
    paired bootstrap resampling — the same form WER and UER take in the
    published leaderboard:

    - For ``metric == "wer"``: (werEdits, werRefWords). Sum-ratio = corpus WER.
    - For ``metric == "significantWer"``: (1 if majorErrorsCount > 0 else 0, 1).
      Sum-ratio = utterance error rate (fraction of utterances with any
      major error), matching ``scoring.score``'s flush.

    Skips utterances where the metric is None (unintelligible, etc.).
    """
    scores: dict[str, tuple[float, float]] = {}
    for locale in locales:
        detail_dir = results_dir / provider / "details" / locale
        if not detail_dir.exists():
            continue
        for f in detail_dir.glob("*.json"):
            utterance_id = f.stem
            with open(f, "r") as fh:
                detail = json.load(fh)

            if metric == "significantWer":
                sig = detail.get("significantWer")
                if sig is None:
                    continue
                major = detail.get("majorErrorsCount", 0) or 0
                scores[f"{locale}/{utterance_id}"] = (1.0 if major > 0 else 0.0, 1.0)
            elif metric == "wer":
                edits = detail.get("werEdits")
                ref_words = detail.get("werRefWords")
                if edits is None or ref_words is None or ref_words <= 0:
                    continue
                scores[f"{locale}/{utterance_id}"] = (float(edits), float(ref_words))
            else:
                val = detail.get(metric)
                if val is not None:
                    # Generic fallback: treat metric as a per-utterance rate
                    # with implicit denominator 1.
                    scores[f"{locale}/{utterance_id}"] = (float(val), 1.0)
    return scores


def utterance_key_to_conversation(key: str) -> str:
    """Map ``"<locale>/conv-N-turn-M"`` to ``"<locale>/conv-N"``.

    Conversation-level resampling treats utterances within the same
    conversation as a single sampling unit, because they share speaker,
    audio quality, environment, and conversational context — i.e. they
    are not independent draws.
    """
    locale, utt = key.split("/", 1)
    if "-turn-" in utt:
        conv = utt.rsplit("-turn-", 1)[0]
    else:
        conv = utt
    return f"{locale}/{conv}"


def aggregate_per_conversation(
    scores: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Sum (num, denom) within each conversation.

    Returns ``{conversation_key: (sum_num, sum_denom)}``. Used as the
    sampling unit for the conversation-level paired bootstrap.
    """
    out: dict[str, list[float]] = {}
    for utt_key, (num, denom) in scores.items():
        conv_key = utterance_key_to_conversation(utt_key)
        if conv_key not in out:
            out[conv_key] = [0.0, 0.0]
        out[conv_key][0] += num
        out[conv_key][1] += denom
    return {k: (v[0], v[1]) for k, v in out.items()}


def paired_bootstrap_ratio(
    num_a: np.ndarray,
    den_a: np.ndarray,
    num_b: np.ndarray,
    den_b: np.ndarray,
    n_iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Paired bootstrap test on the ratio of sums, conversation-level.

    Inputs are aligned arrays of ``(numerator, denominator)`` aggregated
    **per conversation** (not per utterance) — see
    ``aggregate_per_conversation``. For each of ``n_iterations`` rounds we
    resample N conversation indices with replacement, then compute
    ``sum(num[idx]) / sum(den[idx])`` for each provider and take the
    difference as the test statistic.

    Sampling at the conversation level rather than the utterance level
    avoids over-claiming independence: utterances within the same
    conversation share speaker, audio quality, environment, and topic, so
    they are not independent draws. Conversation-level resampling gives
    more conservative (wider) confidence intervals.

    Both metrics in this script flow through this function:
    - WER: num=edits, den=ref_words → corpus WER per resample.
    - UER: num=(0/1 has-major-error), den=1 → utterance error rate per
      resample.

    Returns ``(p_value, mean_diff, mean_diff_std)`` where ``p_value`` is
    the fraction of bootstrap samples where A >= B (low p means A is
    significantly *better*, i.e. lower error rate, than B).
    """
    n = len(num_a)
    indices = rng.integers(0, n, size=(n_iterations, n))
    boot_a = num_a[indices].sum(axis=1) / den_a[indices].sum(axis=1)
    boot_b = num_b[indices].sum(axis=1) / den_b[indices].sum(axis=1)
    diffs = boot_a - boot_b
    p_value = (diffs >= 0).mean()
    return float(p_value), float(diffs.mean()), float(diffs.std())


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap significance tests")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory containing per-provider detail files",
    )
    parser.add_argument(
        "--metric",
        default="significantWer",
        choices=["wer", "significantWer"],
        help="Metric to test (default: significantWer)",
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        default=None,
        help="Locales to include (default: all)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of bootstrap iterations (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Optional path to results/significance.json. If provided, merges the "
            "computed means + pairwise matrix into the file's ``metrics[<metric>]`` "
            "block and also refreshes the top-level ``providers`` list. Preserves "
            "other keys (e.g. ``variance``, ``_note``)."
        ),
    )
    args = parser.parse_args()

    locales = args.locales or TARGET_LOCALES
    results_dir = args.results_dir.resolve()
    rng = np.random.default_rng(args.seed)

    print(f"Metric: {args.metric}")
    print(f"Locales: {locales}")
    print(f"Iterations: {args.iterations}")
    print(f"Results dir: {results_dir}")
    print()

    # Load scores for all providers
    all_scores: dict[str, dict[str, float]] = {}
    for provider in PROVIDERS:
        scores = load_utterance_scores(results_dir, provider, locales, args.metric)
        if scores:
            all_scores[provider] = scores
            print(f"  {SHORT_NAMES[provider]}: {len(scores)} utterances")
        else:
            print(f"  {SHORT_NAMES[provider]}: NO DATA FOUND")

    available = [p for p in PROVIDERS if p in all_scores]
    if len(available) < 2:
        print("Need at least 2 providers with data")
        return

    # Find common utterances across all providers
    common_keys = set.intersection(*(set(all_scores[p].keys()) for p in available))
    print(f"\nCommon utterances across all providers: {len(common_keys)}")

    if not common_keys:
        print("No common utterances found")
        return

    sorted_keys = sorted(common_keys)

    # Aggregate per-utterance (num, denom) into per-conversation (sum_num,
    # sum_denom) so the bootstrap resamples conversations, not utterances.
    # A conversation is the natural sampling unit because utterances within
    # a conversation share speaker, audio, environment, and topic.
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    means: dict[str, float] = {}
    conv_keys: list[str] | None = None
    for p in available:
        per_utt = {k: all_scores[p][k] for k in sorted_keys}
        per_conv = aggregate_per_conversation(per_utt)
        if conv_keys is None:
            conv_keys = sorted(per_conv.keys())
        else:
            # Sanity: same conversation set across providers (we already
            # filtered to common utterances above).
            assert sorted(per_conv.keys()) == conv_keys, "conversation set mismatch across providers"
        nums = np.array([per_conv[c][0] for c in conv_keys], dtype=float)
        dens = np.array([per_conv[c][1] for c in conv_keys], dtype=float)
        arrays[p] = (nums, dens)
        means[p] = nums.sum() / dens.sum() if dens.sum() > 0 else float("nan")
    print(f"\nConversation-level sampling units: {len(conv_keys or [])}")

    # Overall summary
    print(f"\n=== Overall {args.metric} ===")
    ranked = sorted(available, key=lambda p: means[p])
    for i, p in enumerate(ranked):
        print(f"  {i + 1}. {SHORT_NAMES[p]:<15} {means[p]:.4f}")

    # Pairwise bootstrap tests
    print(f"\n=== Pairwise Bootstrap Significance ({args.iterations} iterations) ===")
    print("p-value = P(row provider >= col provider), low p = row is significantly better\n")

    # Header
    name_width = 12
    col_width = 12
    header = " " * name_width
    for p in ranked:
        header += f"{SHORT_NAMES[p]:>{col_width}}"
    print(header)
    print("-" * len(header))

    sig_pairs = []
    # Pairwise matrix ordered by ``ranked`` (best → worst).
    # ``pairwise_matrix[i][j] = P(ranked[i] >= ranked[j])``; diagonal is None.
    pairwise_matrix: list[list[float | None]] = []
    for p_row in ranked:
        row = f"{SHORT_NAMES[p_row]:<{name_width}}"
        matrix_row: list[float | None] = []
        for p_col in ranked:
            if p_row == p_col:
                row += f"{'--':>{col_width}}"
                matrix_row.append(None)
            else:
                n_a, d_a = arrays[p_row]
                n_b, d_b = arrays[p_col]
                p_val, _, _ = paired_bootstrap_ratio(n_a, d_a, n_b, d_b, args.iterations, rng)
                matrix_row.append(round(p_val, 4))
                if p_val < 0.001:
                    cell = "p<0.001"
                else:
                    cell = f"p={p_val:.3f}"
                row += f"{cell:>{col_width}}"
                if p_val < 0.05 and means[p_row] < means[p_col]:
                    sig_pairs.append((p_row, p_col, p_val))
        pairwise_matrix.append(matrix_row)
        print(row)

    # Summary
    print("\n=== Significant differences (p < 0.05) ===")
    if sig_pairs:
        for p_a, p_b, p_val in sig_pairs:
            p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
            print(f"  {SHORT_NAMES[p_a]} < {SHORT_NAMES[p_b]}: {means[p_a]:.4f} vs {means[p_b]:.4f} ({p_str})")
    else:
        print("  No significant differences found")

    # Per-locale breakdown
    if len(locales) > 1:
        print("\n=== Per-Locale Rankings ===")
        for locale in locales:
            locale_scores: dict[str, float] = {}
            prefix = f"{locale}/"
            for p in available:
                nums = np.array([all_scores[p][k][0] for k in sorted_keys if k.startswith(prefix)], dtype=float)
                dens = np.array([all_scores[p][k][1] for k in sorted_keys if k.startswith(prefix)], dtype=float)
                if dens.sum() > 0:
                    locale_scores[p] = float(nums.sum() / dens.sum())
            if locale_scores:
                loc_ranked = sorted(locale_scores.keys(), key=lambda p: locale_scores[p])
                ranking = ", ".join(
                    f"{i + 1}.{SHORT_NAMES[p]}({locale_scores[p]:.4f})" for i, p in enumerate(loc_ranked)
                )
                print(f"  {locale}: {ranking}")

    # Optional: merge into results/significance.json
    if args.output_json is not None:
        out_path = args.output_json.resolve()
        if out_path.is_file():
            with out_path.open("r", encoding="utf-8") as f:
                doc = json.load(f)
        else:
            doc = {}
        doc["providers"] = [{"id": p, "name": SHORT_NAMES[p]} for p in ranked]
        doc.setdefault("metrics", {})
        doc["metrics"][args.metric] = {
            "means": {p: round(means[p], 4) for p in ranked},
            "pairwise": pairwise_matrix,
            "numUtterances": len(sorted_keys),
            "numConversations": len(conv_keys or []),
            "numIterations": args.iterations,
            "samplingUnit": "conversation",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\nMerged {args.metric} results into {out_path}")


if __name__ == "__main__":
    main()
