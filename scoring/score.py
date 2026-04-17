"""Calculate metrics for a submission by comparing against ground truth.

Computes WER and significant WER for each utterance, then aggregates per-locale
and overall. Reads ground truth from manifest.json.

WER is reported as **corpus WER**: per utterance we record the edit count
(substitutions + deletions + insertions) and reference word count. Per-locale
WER is sum(edits) / sum(ref_words); overall WER is the unweighted mean of
the per-locale corpus WERs (macro across locales).

WER and significant WER are computed on LLM-normalized submissions (run
scoring.normalize first).

Saves both per-utterance detail files and a scores.json with aggregated metrics.

Usage:
    # Step 1: Normalize submissions
    python -m scoring.normalize --submission-dir submissions/raw/deepgram-nova3

    # Step 2: Score
    python -m scoring.score --submission-dir submissions/raw/deepgram-nova3

    # Or with simple WER (no LLM needed):
    python -m scoring.score --submission-dir submissions/raw/deepgram-nova3 --simple-wer
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from scoring.llm import JUDGE_CONFIG, prompt_sha
from scoring.metrics import (
    TranscriptRow,
    compute_significant_wer,
    compute_simple_wer,
    compute_wer,
    is_unintelligible,
)
from scoring.normalize import load_ground_truth_from_manifest
from scoring.normalize_gold import (
    GOLD_CACHE_DIR,
    compute_manifest_gold_hash,
    load_manifest_gold,
    read_cached_gold_hash,
)

TARGET_LOCALES = ["en-US", "es-MX", "tr-TR", "vi-VN", "zh-CN"]


# Required metadata.yaml config keys. Kept here so scoring.score and
# scoring.validate agree on the schema without a circular import.
CONFIG_BLOCK_KEYS = (
    "beamSize",
    "languageHint",
    "customVocabulary",
    "noiseSuppression",
    "domainAdaptation",
    "keywordBoosting",
)


def _collect_judge_block() -> dict:
    """Build the ``judge`` block for scores.json.

    Records the pinned model / temperature / seed from scoring.llm plus the
    (truncated) SHA of each prompt constant in scoring.prompts. Missing
    prompts record an empty SHA so the drift-checker still catches the
    gap instead of silently omitting the field.
    """
    import datetime

    try:
        from scoring import prompts as _prompts  # type: ignore[attr-defined]
    except Exception:
        _prompts = None  # pragma: no cover

    def _sha(name: str) -> str:
        if _prompts is None:
            return ""
        val = getattr(_prompts, name, None)
        if val is None:
            return ""
        return prompt_sha(val)

    return {
        "model": JUDGE_CONFIG["model"],
        "modelSnapshot": JUDGE_CONFIG["model"],
        "temperature": JUDGE_CONFIG["temperature"],
        "seed": JUDGE_CONFIG["seed"],
        "normalizeGoldPromptSha": _sha("NORMALIZE_GOLD_PROMPT"),
        "normalizePredPromptSha": _sha("NORMALIZE_PRED_AGAINST_GOLD_PROMPT"),
        "significantErrorsPromptSha": _sha("SIGNIFICANT_WORD_ERRORS_PROMPT"),
        "scoredAt": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def _extract_config_block(metadata: dict) -> dict:
    """Pull the metadata.yaml ``config:`` block out for scores.json.

    Unknown keys are passed through (the validator rejects them at submit
    time). Missing ``config`` returns an empty dict so legacy submissions
    don't break scoring.
    """
    cfg = metadata.get("config") if isinstance(metadata, dict) else None
    if not isinstance(cfg, dict):
        return {}
    return {k: cfg.get(k, "default") for k in CONFIG_BLOCK_KEYS}


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
        "Used for WER and significant WER.",
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
        default=["wer", "significantWer"],
        help="Metrics to compute. Options: wer, significantWer",
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

    # Load raw transcript pairs (used as the master utterance index)
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
                "werEdits",
                "werRefWords",
                "werNormalizedGold",
                "werNormalizedPredicted",
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
                    "wer_edits_sum": 0,
                    "wer_ref_words_sum": 0,
                    "wer_count": 0,
                    "sig_wer_has_error": 0,
                    "sig_wer_total": 0,
                    "utterance_count": 0,
                    "unintelligible_count": 0,
                }
            stats = locale_stats[locale]
            stats["utterance_count"] += 1
            if detail["unintelligible"]:
                stats["unintelligible_count"] += 1
            edits = detail.get("werEdits")
            ref_words = detail.get("werRefWords")
            # Silence-hallucination fix (item 6): drop the ``ref_words > 0``
            # guard so silent-clip insertions flow into the corpus sums.
            # ``compute_wer`` reports (edits=hyp_words, ref_words=hyp_words)
            # for silent clips; a truly silent pair is (0, 0) and adds
            # nothing.
            if edits is not None and ref_words is not None:
                stats["wer_edits_sum"] += edits
                stats["wer_ref_words_sum"] += ref_words
                stats["wer_count"] += 1
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
                count = sig_wer_results[idx].major_errors_count
                computed_values[i]["significantWer"] = (
                    1 if (count is not None and count > 0) else (0 if count is not None else None)
                )
                computed_values[i]["majorErrorsCount"] = count
                computed_values[i]["totalWordsCount"] = sig_wer_results[idx].total_words_count
                computed_values[i]["sigWerNormalizedGold"] = sig_wer_results[idx].normalized_gold
                computed_values[i]["sigWerNormalizedPredicted"] = sig_wer_results[idx].normalized_predicted
                computed_values[i]["sigWerErrors"] = sig_wer_results[idx].all_errors_with_scores

        print("Flushing significant WER results to disk...")
        flush_details_to_disk()

    # --- WER ---
    if "wer" in metrics_to_run:
        existing_details = load_all_existing_details()

        # Treat missing edits/ref_words components as needs-recompute so old
        # detail files from the per-utterance-mean era get backfilled rather
        # than approximated.
        def _wer_cached(d: dict | None) -> bool:
            if d is None or d.get("wer") is None:
                return False
            if not is_unintelligible(d.get("gold", "") or ""):
                if d.get("werEdits") is None or d.get("werRefWords") is None:
                    return False
            return True

        needs_wer = [i for i in range(len(pairs)) if not _wer_cached(existing_details[i])]
        for i in range(len(pairs)):
            if i not in set(needs_wer) and existing_details[i] is not None:
                computed_values[i]["wer"] = existing_details[i].get("wer")
                computed_values[i]["werEdits"] = existing_details[i].get("werEdits")
                computed_values[i]["werRefWords"] = existing_details[i].get("werRefWords")
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
                computed_values[i]["werEdits"] = wer_results[idx].edits
                computed_values[i]["werRefWords"] = wer_results[idx].ref_words
                computed_values[i]["werNormalizedGold"] = wer_results[idx].normalized_gold
                computed_values[i]["werNormalizedPredicted"] = wer_results[idx].normalized_predicted

        print("Flushing WER results to disk...")
        flush_details_to_disk()

    # Final flush to collect locale stats for summary
    locale_stats = flush_details_to_disk()

    # Build summary
    def avg(s, c):
        return round(s / c, 4) if c > 0 else None

    def corpus_wer(s):
        """Per-locale corpus WER: sum(edits) / sum(ref_words)."""
        n, d = s["wer_edits_sum"], s["wer_ref_words_sum"]
        return round(n / d, 4) if d > 0 else None

    summary_locales = {}
    total_sig_wer_has_error, total_sig_wer_total = 0, 0
    total_utterances, total_unintelligible = 0, 0

    for locale in TARGET_LOCALES:
        if locale not in locale_stats:
            continue
        s = locale_stats[locale]
        summary_locales[locale] = {
            "wer": corpus_wer(s),
            "significantWer": avg(s["sig_wer_has_error"], s["sig_wer_total"]),
            "utteranceCount": s["utterance_count"],
            "unintelligibleCount": s["unintelligible_count"],
        }
        total_sig_wer_has_error += s["sig_wer_has_error"]
        total_sig_wer_total += s["sig_wer_total"]
        total_utterances += s["utterance_count"]
        total_unintelligible += s["unintelligible_count"]

    # Overall only when all locales present
    all_locales_present = all(loc in locale_stats for loc in TARGET_LOCALES)
    if all_locales_present:
        # Overall WER is the unweighted mean of per-locale corpus WERs
        # (macro across locales). Matches the leaderboard UI's overall column.
        per_locale_wers = [
            summary_locales[loc]["wer"] for loc in TARGET_LOCALES if summary_locales[loc]["wer"] is not None
        ]
        overall_wer = (
            round(sum(per_locale_wers) / len(per_locale_wers), 4)
            if len(per_locale_wers) == len(TARGET_LOCALES)
            else None
        )
        # Overall UER (item 3): switch from utterance-micro across all
        # locales to the unweighted mean of per-locale UERs (locale-macro),
        # symmetric with the WER overall above. Same field name, same shape
        # in scores.json — only the value changes.
        per_locale_uers = [
            summary_locales[loc]["significantWer"]
            for loc in TARGET_LOCALES
            if summary_locales[loc]["significantWer"] is not None
        ]
        overall_sig_wer = (
            round(sum(per_locale_uers) / len(per_locale_uers), 4)
            if len(per_locale_uers) == len(TARGET_LOCALES)
            else None
        )
        overall = {
            "wer": overall_wer,
            "significantWer": overall_sig_wer,
            "utteranceCount": total_utterances,
            "unintelligibleCount": total_unintelligible,
        }
    else:
        missing = [loc for loc in TARGET_LOCALES if loc not in locale_stats]
        print(f"Overall metrics not computed — missing locales: {missing}")
        overall = None

    # Reproducibility metadata: judge pin, prompt SHAs, manifest-gold
    # hash, and the declared inference config. These feed
    # scripts/check_judge_drift.py and the leaderboard's provider detail.
    try:
        manifest_gold_hash = compute_manifest_gold_hash(load_manifest_gold(manifest_path))
    except Exception as e:  # pragma: no cover — manifest issues surface elsewhere
        print(f"WARNING: could not compute manifest_gold_hash: {e}")
        manifest_gold_hash = ""
    cached_gold_hash = read_cached_gold_hash(GOLD_CACHE_DIR.resolve())

    summary = {
        "model": metadata.get("model", ""),
        "organization": metadata.get("organization", metadata.get("company", "")),
        "date": metadata.get("date", metadata.get("modelVersion", "")),
        "locales": summary_locales,
        "overall": overall,
        "judge": _collect_judge_block(),
        "meta": {
            "config": _extract_config_block(metadata),
            "manifestGoldHash": manifest_gold_hash,
            "canonicalGoldCacheHash": cached_gold_hash or "",
        },
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / "scores.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\nMetrics written to: {metrics_dir}")
    for locale, scores in summary_locales.items():
        print(
            f"  {locale}: WER={scores['wer']}, "
            f"Sig.WER={scores['significantWer']} ({scores['utteranceCount']} utterances)"
        )
    if overall:
        print(
            f"  Overall: WER={overall['wer']}, "
            f"Sig.WER={overall['significantWer']} ({overall['utteranceCount']} utterances)"
        )


if __name__ == "__main__":
    main()
