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
) -> dict[str, tuple[float, float] | float]:
    """Load per-utterance scores for a provider.

    Returns {utterance_key: value} where utterance_key is 'locale/utterance_id'.
    Skips utterances where the metric is None (unintelligible, etc.).

    For metric == "wer", returns (edits, ref_words) tuples so the caller can
    compute WER as a ratio of sums under bootstrap resampling.

    For metric == "significantWer", returns the binary "has any major error"
    (1.0 or 0.0) to match the leaderboard's aggregation (fraction of utterances
    with errors), NOT the per-utterance error rate.
    """
    scores: dict[str, tuple[float, float] | float] = {}
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
                scores[f"{locale}/{utterance_id}"] = 1.0 if major > 0 else 0.0
            elif metric == "wer":
                edits = detail.get("werEdits")
                ref_words = detail.get("werRefWords")
                if edits is None or ref_words is None or ref_words <= 0:
                    continue
                scores[f"{locale}/{utterance_id}"] = (float(edits), float(ref_words))
            else:
                val = detail.get(metric)
                if val is not None:
                    scores[f"{locale}/{utterance_id}"] = val
    return scores


def paired_bootstrap_mean(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Paired bootstrap test on means: is mean(A) < mean(B)?

    Returns (p_value, mean_diff, mean_diff_std) where p_value is the
    fraction of bootstrap samples where A >= B (i.e., low p means A is
    significantly better than B).
    """
    n = len(scores_a)
    indices = rng.integers(0, n, size=(n_iterations, n))
    boot_a = scores_a[indices].mean(axis=1)
    boot_b = scores_b[indices].mean(axis=1)
    diffs = boot_a - boot_b
    p_value = (diffs >= 0).mean()
    return float(p_value), float(diffs.mean()), float(diffs.std())


def paired_bootstrap_ratio(
    edits_a: np.ndarray,
    refs_a: np.ndarray,
    edits_b: np.ndarray,
    refs_b: np.ndarray,
    n_iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Paired bootstrap test on ratio of sums.

    For each resample of utterance indices, computes
        sum(edits[idx]) / sum(ref_words[idx])
    per provider and uses the difference as the test statistic. Reflects the
    real per-locale WER aggregation.
    """
    n = len(edits_a)
    indices = rng.integers(0, n, size=(n_iterations, n))
    boot_a = edits_a[indices].sum(axis=1) / refs_a[indices].sum(axis=1)
    boot_b = edits_b[indices].sum(axis=1) / refs_b[indices].sum(axis=1)
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
    is_wer = args.metric == "wer"

    # Build per-provider arrays (or array pairs for WER ratio)
    arrays: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]] = {}
    means: dict[str, float] = {}
    for p in available:
        if is_wer:
            edits = np.array([all_scores[p][k][0] for k in sorted_keys], dtype=float)
            refs = np.array([all_scores[p][k][1] for k in sorted_keys], dtype=float)
            arrays[p] = (edits, refs)
            means[p] = edits.sum() / refs.sum() if refs.sum() > 0 else float("nan")
        else:
            vals = np.array([all_scores[p][k] for k in sorted_keys], dtype=float)
            arrays[p] = vals
            means[p] = float(vals.mean())

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
    for p_row in ranked:
        row = f"{SHORT_NAMES[p_row]:<{name_width}}"
        for p_col in ranked:
            if p_row == p_col:
                row += f"{'--':>{col_width}}"
            else:
                if is_wer:
                    e_a, r_a = arrays[p_row]
                    e_b, r_b = arrays[p_col]
                    p_val, _, _ = paired_bootstrap_ratio(e_a, r_a, e_b, r_b, args.iterations, rng)
                else:
                    p_val, _, _ = paired_bootstrap_mean(arrays[p_row], arrays[p_col], args.iterations, rng)
                if p_val < 0.001:
                    cell = "p<0.001"
                else:
                    cell = f"p={p_val:.3f}"
                row += f"{cell:>{col_width}}"
                if p_val < 0.05 and means[p_row] < means[p_col]:
                    sig_pairs.append((p_row, p_col, p_val))
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
                if is_wer:
                    e = np.array([all_scores[p][k][0] for k in sorted_keys if k.startswith(prefix)], dtype=float)
                    r = np.array([all_scores[p][k][1] for k in sorted_keys if k.startswith(prefix)], dtype=float)
                    if r.sum() > 0:
                        locale_scores[p] = float(e.sum() / r.sum())
                else:
                    vals = [all_scores[p][k] for k in sorted_keys if k.startswith(prefix)]
                    if vals:
                        locale_scores[p] = float(np.mean(vals))
            if locale_scores:
                loc_ranked = sorted(locale_scores.keys(), key=lambda p: locale_scores[p])
                ranking = ", ".join(
                    f"{i + 1}.{SHORT_NAMES[p]}({locale_scores[p]:.4f})" for i, p in enumerate(loc_ranked)
                )
                print(f"  {locale}: {ranking}")


if __name__ == "__main__":
    main()
