"""Aggregate per-utterance scoring details across waves into the
N=4 variance block of ``results/significance.json``.

Inputs (per provider, per locale, one detail JSON per utterance):
  * Wave A (published):    results/<provider>/details/<locale>/*.json
  * Wave B (variance run): tmp/variance_runs/wave-b/_results/<provider>/details/<locale>/*.json
  * Wave C (variance run): tmp/variance_runs/wave-c/_results/<provider>/details/<locale>/*.json
  * Wave D (variance run): tmp/variance_runs/wave-d/_results/<provider>/details/<locale>/*.json

Per detail file we read:
  * ``werEdits`` + ``werRefWords`` -> corpus WER per locale
    = sum(werEdits) / sum(werRefWords) across the locale
  * ``majorErrorsCount`` + ``totalWordsCount`` -> corpus sigWER per locale
    = sum(majorErrorsCount) / sum(totalWordsCount) across the locale
  * ``unintelligible`` rows are skipped (they have null edits / null
    sigWer in score.py).

For each (provider, locale) we get one corpus WER and one corpus sigWER
per wave -> four samples -> mean and std (ddof=1, sample std).

Writes the new ``variance`` block back into
``results/significance.json``, preserving the ``providers`` and
``metrics`` blocks (refreshed by ``scripts/significance_test.py`` in a
separate step). Adds an inline ``_note`` calling out that the new
variance includes provider transcription noise.

Usage:
    .venv/bin/python scripts/aggregate_variance.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIG_PATH = REPO_ROOT / "results" / "significance.json"
DEFAULT_VARIANCE_LOCALES = ["en-US", "zh-CN"]
DEFAULT_PROVIDERS = [
    "deepgram-nova3",
    "google-chirp3",
    "azure",
    "elevenlabs-scribe-v2",
    "openai-gpt4o-mini-transcribe",
]

WAVE_PATHS = {
    "a": REPO_ROOT / "results",
    "b": REPO_ROOT / "tmp" / "variance_runs" / "wave-b" / "_results",
    "c": REPO_ROOT / "tmp" / "variance_runs" / "wave-c" / "_results",
    "d": REPO_ROOT / "tmp" / "variance_runs" / "wave-d" / "_results",
}


def corpus_metrics_for_wave(
    wave_results_root: Path, provider: str, locale: str
) -> tuple[float | None, float | None, int]:
    """Return (corpus_wer, utterance_error_rate, n_utterances) for one wave/provider/locale.

    Must match the definitions used by ``scoring.score``'s flush so the
    variance block is directly comparable to the published overall /
    per-locale numbers:

    - ``corpus WER = sum(werEdits) / sum(werRefWords)`` (edits per reference word)
    - ``UER = n_utt_with_any_major_error / n_utt_scored`` (fraction of
      utterances where ``majorErrorsCount > 0``). This is the LEADERBOARD
      number under the field ``significantWer``.

    Returns (None, None, 0) if the details directory is missing.
    """
    details_dir = wave_results_root / provider / "details" / locale
    if not details_dir.is_dir():
        return None, None, 0

    edits_sum = 0
    ref_sum = 0
    sig_wer_scored = 0  # utterances with significantWer != None
    sig_wer_errorful = 0  # utterances with majorErrorsCount > 0
    n = 0
    for detail_path in sorted(details_dir.glob("*.json")):
        with detail_path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        n += 1
        if d.get("unintelligible"):
            continue
        edits = d.get("werEdits")
        ref_words = d.get("werRefWords")
        if edits is not None and ref_words is not None:
            edits_sum += edits
            ref_sum += ref_words
        if d.get("significantWer") is not None:
            sig_wer_scored += 1
            major = d.get("majorErrorsCount") or 0
            if major > 0:
                sig_wer_errorful += 1

    wer = (edits_sum / ref_sum) if ref_sum > 0 else None
    uer = (sig_wer_errorful / sig_wer_scored) if sig_wer_scored > 0 else None
    return wer, uer, n


def sample_mean_std(values: list[float]) -> tuple[float, float]:
    """Sample mean and std (ddof=1)."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def aggregate(
    providers: list[str],
    locales: list[str],
    waves: list[str],
    require_all_waves: bool,
) -> tuple[dict, list[str]]:
    """Build the variance dict. Returns (variance_block, warnings)."""
    variance: dict = {}
    warnings: list[str] = []
    for provider in providers:
        per_locale: dict = {}
        for locale in locales:
            wer_samples: list[float] = []
            sig_samples: list[float] = []
            wave_ns: dict[str, int] = {}
            for wave in waves:
                root = WAVE_PATHS[wave]
                wer, sig, n = corpus_metrics_for_wave(root, provider, locale)
                wave_ns[wave] = n
                if wer is None or sig is None:
                    msg = f"  WARN: missing wave {wave} details for {provider}/{locale} (root={root}, n_files={n})"
                    warnings.append(msg)
                    if require_all_waves:
                        continue
                else:
                    wer_samples.append(wer)
                    sig_samples.append(sig)

            wer_mean, wer_std = sample_mean_std(wer_samples)
            sig_mean, sig_std = sample_mean_std(sig_samples)
            per_locale[locale] = {
                "wer": {"mean": round(wer_mean, 4), "std": round(wer_std, 4)},
                "significantWer": {"mean": round(sig_mean, 4), "std": round(sig_std, 4)},
                "_n_waves": len(wer_samples),
                "_wave_n_utterances": wave_ns,
            }
        variance[provider] = per_locale
    return variance, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--significance-path",
        type=Path,
        default=DEFAULT_SIG_PATH,
        help=f"Path to results/significance.json (default: {DEFAULT_SIG_PATH.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=DEFAULT_PROVIDERS,
        help="Provider IDs to aggregate (default: all 5)",
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        default=DEFAULT_VARIANCE_LOCALES,
        help="Locales for the variance block (default: en-US zh-CN)",
    )
    parser.add_argument(
        "--waves",
        nargs="+",
        default=list(WAVE_PATHS.keys()),
        help="Waves to include (default: a b c d)",
    )
    parser.add_argument(
        "--allow-missing-waves",
        action="store_true",
        help="Compute mean/std on whatever waves are present (default: warn but continue)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the aggregated block but do not write significance.json",
    )
    args = parser.parse_args(argv)

    variance, warnings = aggregate(
        providers=args.providers,
        locales=args.locales,
        waves=args.waves,
        require_all_waves=False,
    )
    for w in warnings:
        print(w)

    if args.dry_run:
        print(json.dumps(variance, indent=2))
        return 0

    if not args.significance_path.is_file():
        print(f"ERROR: {args.significance_path} not found")
        return 2

    with args.significance_path.open("r", encoding="utf-8") as f:
        sig_doc = json.load(f)

    sig_doc["variance"] = variance
    sig_doc.setdefault("_note", {})
    sig_doc["_note"]["variance"] = (
        f"N={len(args.waves)} runs (waves {', '.join(args.waves)}); "
        "includes provider transcription noise (prior block held transcripts fixed)."
    )

    with args.significance_path.open("w", encoding="utf-8") as f:
        json.dump(sig_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote variance block to {args.significance_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
