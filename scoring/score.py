"""Calculate metrics for a submission by comparing against ground truth.

Computes WER, quality score, and significant WER for each utterance,
then aggregates per-locale and overall. Reads ground truth from manifest.json.

Quality is computed on raw submissions. WER and significant WER are computed
on LLM-normalized submissions (run scoring.normalize first).

Saves both per-utterance detail files and a scores.json with aggregated metrics.

Usage:
    # Step 1: Normalize submissions
    python -m scoring.normalize --submission-dir submissions/raw/deepgram-nova3

    # Step 2: Score (uses raw for quality, normalized for WER/sigWER)
    python -m scoring.score --submission-dir submissions/raw/deepgram-nova3

    # Or with simple WER (no LLM needed):
    python -m scoring.score --submission-dir submissions/raw/deepgram-nova3 --simple-wer
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from scoring.metrics import (
    TranscriptRow,
    compute_quality,
    compute_significant_wer,
    compute_simple_wer,
    compute_wer,
    is_unintelligible,
)
from scoring.normalize import load_ground_truth_from_manifest

TARGET_LOCALES = ["en-US", "es-MX", "tr-TR", "vi-VN", "zh-CN"]


def load_transcript_pairs(
    submission_dir: Path, ground_truth: dict[str, dict[str, str]]
) -> list[tuple[str, str, TranscriptRow]]:
    """Load paired ground truth and submission transcripts as TranscriptRows."""
    rows = []
    for locale in TARGET_LOCALES:
        if locale not in ground_truth:
            continue
        sub_locale_dir = submission_dir / locale
        if not sub_locale_dir.exists():
            continue

        for utterance_id, gold in sorted(ground_truth[locale].items()):
            sub_file = sub_locale_dir / f"{utterance_id}.txt"
            if not sub_file.exists():
                # Missing submission file — count as empty prediction
                predicted = ""
            else:
                predicted = sub_file.read_text(encoding="utf-8").strip()

            row = TranscriptRow(
                locale=locale,
                utterance_id=utterance_id,
                gold=gold,
                predicted=predicted,
            )
            rows.append((locale, utterance_id, row))

    return rows


def load_normalized_pairs_with_gold(
    normalized_dir: Path,
) -> list[tuple[str, str, TranscriptRow]]:
    """Load transcript pairs where both gold and predicted come from symmetric normalization.

    Reads normalized predicted from <utterance_id>.txt and normalized gold from
    <utterance_id>.gold.txt. Both sides have been symmetrically normalized.
    """
    rows = []
    for locale in TARGET_LOCALES:
        locale_dir = normalized_dir / locale
        if not locale_dir.exists():
            continue

        for txt_file in sorted(locale_dir.glob("*.txt")):
            if txt_file.name.endswith(".gold.txt"):
                continue
            utterance_id = txt_file.stem
            gold_file = locale_dir / f"{utterance_id}.gold.txt"
            if not gold_file.exists():
                continue

            predicted = txt_file.read_text(encoding="utf-8").strip()
            gold = gold_file.read_text(encoding="utf-8").strip()

            row = TranscriptRow(
                locale=locale,
                utterance_id=utterance_id,
                gold=gold,
                predicted=predicted,
            )
            rows.append((locale, utterance_id, row))

    return rows


def load_existing_detail(detail_path: Path) -> dict | None:
    """Load an existing detail JSON file if it exists."""
    if detail_path.exists():
        with open(detail_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Calculate metrics for a submission")
    parser.add_argument("--submission-dir", type=Path, required=True, help="Raw submission directory")
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=None,
        help="LLM-normalized submission directory (default: submissions/normalized/<name>). "
        "Used for WER and significant WER. Quality always uses raw submissions.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifest.json"),
        help="Path to manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for metrics (default: results/<submission-name>)",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel workers for LLM calls")
    parser.add_argument("--max-utterances", type=int, default=None, help="Limit utterances for testing")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["wer", "quality", "significantWer"],
        help="Metrics to compute. Options: wer, quality, significantWer",
    )
    parser.add_argument(
        "--simple-wer",
        action="store_true",
        help="Use simple normalization for WER instead of LLM-based (no OpenAI needed)",
    )
    parser.add_argument("--locales", nargs="+", default=None, help="Limit to specific locales (e.g. en-US zh-CN)")
    args = parser.parse_args()

    if args.locales:
        global TARGET_LOCALES
        TARGET_LOCALES = [loc for loc in TARGET_LOCALES if loc in args.locales]

    submission_dir = args.submission_dir.resolve()
    manifest_path = args.manifest.resolve()

    # Determine normalized directory
    submission_name = submission_dir.name
    if args.normalized_dir:
        normalized_dir = args.normalized_dir.resolve()
    else:
        normalized_dir = (Path("submissions/normalized") / submission_name).resolve()

    # Determine output directory
    if args.output_dir:
        metrics_dir = args.output_dir.resolve()
    else:
        metrics_dir = (Path("results") / submission_name).resolve()

    # Load metadata
    metadata = {}
    for meta_name in ["metadata.yaml", "metadata.json"]:
        meta_path = submission_dir / meta_name
        if meta_path.exists():
            if meta_name.endswith(".yaml"):
                try:
                    import yaml

                    with open(meta_path, "r") as f:
                        metadata = yaml.safe_load(f) or {}
                except ImportError:
                    pass
            else:
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
            break

    # Load ground truth from manifest
    ground_truth = load_ground_truth_from_manifest(manifest_path)

    # Load raw transcript pairs (used for quality scoring)
    print("Loading raw transcript pairs...")
    raw_pairs = load_transcript_pairs(submission_dir, ground_truth)
    print(f"Loaded {len(raw_pairs)} raw utterance pairs")

    # Load normalized transcript pairs (used for WER/sigWER)
    has_normalized = normalized_dir.exists()
    if has_normalized and not args.simple_wer:
        print(f"Loading normalized transcripts from {normalized_dir}...")
        # Try loading with .gold.txt files first (symmetric normalization)
        normalized_pairs = load_normalized_pairs_with_gold(normalized_dir)
        if normalized_pairs:
            print(f"Loaded {len(normalized_pairs)} normalized utterance pairs (both sides normalized)")
        else:
            normalized_pairs = load_transcript_pairs(normalized_dir, ground_truth)
            print(f"Loaded {len(normalized_pairs)} normalized utterance pairs")
    else:
        if not args.simple_wer and ("wer" in args.metrics or "significantWer" in args.metrics):
            print(f"WARNING: Normalized directory not found at {normalized_dir}")
            print("  WER/sigWER will use raw transcripts. Run 'python -m scoring.normalize' first for best results.")
        normalized_pairs = raw_pairs

    # Use raw pairs as the primary index
    pairs = raw_pairs

    if args.max_utterances:
        pairs = pairs[: args.max_utterances]
        print(f"Limited to {len(pairs)} utterances for testing")

    if not pairs:
        print("No matching utterances found")
        return

    all_raw_rows = [row for _, _, row in pairs]

    # Build normalized rows aligned to raw pairs by utterance_id
    normalized_by_key = {(loc, uid): row for loc, uid, row in normalized_pairs}
    all_normalized_rows = []
    for locale, utterance_id, raw_row in pairs:
        norm_row = normalized_by_key.get((locale, utterance_id), raw_row)
        all_normalized_rows.append(norm_row)
    metrics_to_run = set(args.metrics)
    print(f"Metrics to compute: {metrics_to_run}")

    def load_all_existing_details():
        details: list[dict | None] = [None] * len(pairs)
        for i, (locale, utterance_id, _) in enumerate(pairs):
            details[i] = load_existing_detail(metrics_dir / "details" / locale / f"{utterance_id}.json")
        return details

    def flush_details_to_disk():
        """Write current state of all detail files and return locale stats."""
        locale_stats: dict[str, dict] = {}
        for i, (locale, utterance_id, row) in enumerate(pairs):
            detail_dir = metrics_dir / "details" / locale
            detail_path = detail_dir / f"{utterance_id}.json"

            existing = load_existing_detail(detail_path)

            detail = {
                "gold": row.gold,
                "predicted": row.predicted,
                "unintelligible": is_unintelligible(row.gold),
            }
            for key in [
                "wer",
                "werNormalizedGold",
                "werNormalizedPredicted",
                "qualityScore",
                "significantWer",
                "majorErrorsCount",
                "totalWordsCount",
                "sigWerNormalizedGold",
                "sigWerNormalizedPredicted",
                "sigWerErrors",
            ]:
                detail[key] = (
                    computed_values[i].get(key)
                    if computed_values[i].get(key) is not None
                    else (existing or {}).get(key)
                )

            detail_dir.mkdir(parents=True, exist_ok=True)
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(detail, f, indent=2, ensure_ascii=False)

            if locale not in locale_stats:
                locale_stats[locale] = {
                    "wer_sum": 0.0,
                    "wer_count": 0,
                    "quality_sum": 0,
                    "quality_count": 0,
                    "sig_wer_has_error": 0,
                    "sig_wer_total": 0,
                    "utterance_count": 0,
                    "unintelligible_count": 0,
                }
            stats = locale_stats[locale]
            stats["utterance_count"] += 1
            if detail["unintelligible"]:
                stats["unintelligible_count"] += 1
            if detail["wer"] is not None:
                stats["wer_sum"] += detail["wer"]
                stats["wer_count"] += 1
            if detail["qualityScore"] is not None:
                stats["quality_sum"] += detail["qualityScore"]
                stats["quality_count"] += 1
            if detail["significantWer"] is not None:
                stats["sig_wer_total"] += 1
                if detail.get("majorErrorsCount") and detail["majorErrorsCount"] > 0:
                    stats["sig_wer_has_error"] += 1
        return locale_stats

    computed_values: list[dict] = [{} for _ in pairs]

    # --- Significant WER (run first to validate quickly) ---
    if "significantWer" in metrics_to_run:
        existing_details = load_all_existing_details()
        needs_sig_wer = [
            i
            for i in range(len(pairs))
            if existing_details[i] is None or existing_details[i].get("significantWer") is None
        ]
        for i in range(len(pairs)):
            if i not in set(needs_sig_wer) and existing_details[i] is not None:
                computed_values[i]["significantWer"] = existing_details[i].get("significantWer")
                computed_values[i]["majorErrorsCount"] = existing_details[i].get("majorErrorsCount")
                computed_values[i]["totalWordsCount"] = existing_details[i].get("totalWordsCount")
                computed_values[i]["sigWerNormalizedGold"] = existing_details[i].get("sigWerNormalizedGold")
                computed_values[i]["sigWerNormalizedPredicted"] = existing_details[i].get("sigWerNormalizedPredicted")
                computed_values[i]["sigWerErrors"] = existing_details[i].get("sigWerErrors")

        if not needs_sig_wer:
            print(f"Significant WER: all {len(pairs)} utterances already computed, skipping")
        else:
            print(f"Computing significant WER for {len(needs_sig_wer)} utterances (normalized transcripts)...")
            rows_to_compute = [all_normalized_rows[i] for i in needs_sig_wer]
            sig_wer_results = compute_significant_wer(
                rows_to_compute,
                num_workers=args.num_workers,
            )
            for idx, i in enumerate(needs_sig_wer):
                computed_values[i]["significantWer"] = sig_wer_results[idx].major_error_rate
                computed_values[i]["majorErrorsCount"] = sig_wer_results[idx].major_errors_count
                computed_values[i]["totalWordsCount"] = sig_wer_results[idx].total_words_count
                computed_values[i]["sigWerNormalizedGold"] = sig_wer_results[idx].normalized_gold
                computed_values[i]["sigWerNormalizedPredicted"] = sig_wer_results[idx].normalized_predicted
                computed_values[i]["sigWerErrors"] = sig_wer_results[idx].all_errors_with_scores

        print("Flushing significant WER results to disk...")
        flush_details_to_disk()

    # --- WER ---
    if "wer" in metrics_to_run:
        existing_details = load_all_existing_details()
        needs_wer = [
            i for i in range(len(pairs)) if existing_details[i] is None or existing_details[i].get("wer") is None
        ]
        for i in range(len(pairs)):
            if i not in set(needs_wer) and existing_details[i] is not None:
                computed_values[i]["wer"] = existing_details[i].get("wer")
                computed_values[i]["werNormalizedGold"] = existing_details[i].get("werNormalizedGold")
                computed_values[i]["werNormalizedPredicted"] = existing_details[i].get("werNormalizedPredicted")

        if not needs_wer:
            print(f"WER: all {len(pairs)} utterances already computed, skipping")
        else:
            if len(needs_wer) < len(pairs):
                print(f"WER: {len(pairs) - len(needs_wer)}/{len(pairs)} cached, computing {len(needs_wer)} remaining")

            rows_to_compute = [all_normalized_rows[i] for i in needs_wer]
            if args.simple_wer:
                print(f"Computing WER (simple normalization) for {len(needs_wer)} utterances...")
                wer_results = compute_simple_wer(rows_to_compute)
            else:
                print(f"Computing WER (LLM normalization) for {len(needs_wer)} utterances...")
                wer_results = compute_wer(rows_to_compute)

            for idx, i in enumerate(needs_wer):
                computed_values[i]["wer"] = wer_results[idx].wer
                computed_values[i]["werNormalizedGold"] = wer_results[idx].normalized_gold
                computed_values[i]["werNormalizedPredicted"] = wer_results[idx].normalized_predicted

        print("Flushing WER results to disk...")
        flush_details_to_disk()

    # --- Quality ---
    if "quality" in metrics_to_run:
        existing_details = load_all_existing_details()
        needs_quality = [
            i
            for i in range(len(pairs))
            if existing_details[i] is None or existing_details[i].get("qualityScore") is None
        ]
        for i in range(len(pairs)):
            if i not in set(needs_quality) and existing_details[i] is not None:
                computed_values[i]["qualityScore"] = existing_details[i].get("qualityScore")

        if not needs_quality:
            print(f"Quality: all {len(pairs)} utterances already computed, skipping")
        else:
            print(f"Computing quality scores for {len(needs_quality)} utterances (raw transcripts)...")
            rows_to_compute = [all_raw_rows[i] for i in needs_quality]
            quality_results = compute_quality(rows_to_compute, num_workers=args.num_workers)
            for idx, i in enumerate(needs_quality):
                computed_values[i]["qualityScore"] = quality_results[idx].score

        print("Flushing quality results to disk...")
        flush_details_to_disk()

    # Final flush to collect locale stats for summary
    locale_stats = flush_details_to_disk()

    # Build summary
    def avg(s, c):
        return round(s / c, 4) if c > 0 else None

    summary_locales = {}
    total_wer_sum, total_wer_count = 0.0, 0
    total_quality_sum, total_quality_count = 0, 0
    total_sig_wer_has_error, total_sig_wer_total = 0, 0
    total_utterances, total_unintelligible = 0, 0

    for locale in TARGET_LOCALES:
        if locale not in locale_stats:
            continue
        s = locale_stats[locale]
        summary_locales[locale] = {
            "wer": avg(s["wer_sum"], s["wer_count"]),
            "qualityScore": avg(s["quality_sum"], s["quality_count"]),
            "significantWer": avg(s["sig_wer_has_error"], s["sig_wer_total"]),
            "utteranceCount": s["utterance_count"],
            "unintelligibleCount": s["unintelligible_count"],
        }
        total_wer_sum += s["wer_sum"]
        total_wer_count += s["wer_count"]
        total_quality_sum += s["quality_sum"]
        total_quality_count += s["quality_count"]
        total_sig_wer_has_error += s["sig_wer_has_error"]
        total_sig_wer_total += s["sig_wer_total"]
        total_utterances += s["utterance_count"]
        total_unintelligible += s["unintelligible_count"]

    # Overall only when all locales present
    all_locales_present = all(loc in locale_stats for loc in TARGET_LOCALES)
    if all_locales_present:
        overall = {
            "wer": avg(total_wer_sum, total_wer_count),
            "qualityScore": avg(total_quality_sum, total_quality_count),
            "significantWer": avg(total_sig_wer_has_error, total_sig_wer_total),
            "utteranceCount": total_utterances,
            "unintelligibleCount": total_unintelligible,
        }
    else:
        missing = [loc for loc in TARGET_LOCALES if loc not in locale_stats]
        print(f"Overall metrics not computed — missing locales: {missing}")
        overall = None

    summary = {
        "model": metadata.get("model", ""),
        "organization": metadata.get("organization", metadata.get("company", "")),
        "date": metadata.get("date", metadata.get("modelVersion", "")),
        "locales": summary_locales,
        "overall": overall,
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / "scores.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\nMetrics written to: {metrics_dir}")
    for locale, scores in summary_locales.items():
        print(
            f"  {locale}: WER={scores['wer']}, Quality={scores['qualityScore']}, "
            f"Sig.WER={scores['significantWer']} ({scores['utteranceCount']} utterances)"
        )
    if overall:
        print(
            f"  Overall: WER={overall['wer']}, Quality={overall['qualityScore']}, "
            f"Sig.WER={overall['significantWer']} ({overall['utteranceCount']} utterances)"
        )


if __name__ == "__main__":
    main()
