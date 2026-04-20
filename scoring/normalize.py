"""LLM-normalize submission predictions for fair WER comparison.

The gold side of the comparison is produced **once** from ``manifest.json``
by ``scoring.normalize_gold`` and lives under
``submissions/normalized/_gold/<locale>/<id>.gold.txt`` — it is the same
string every provider is scored against, so that provider-conditioned
normalization drift cannot leak into the leaderboard (item 1 of the
fairness-fixes plan).

This module normalizes only the **predicted** side. It reads the canonical
normalized gold for reference (so the LLM knows the target style), and
writes ``<id>.txt`` per utterance. The submitter-specific
``<id>.gold.txt`` files are no longer written here; consumers should read
the canonical cache instead.

Usage:
    python -m scoring.normalize --submission-dir submissions/raw/deepgram-nova3
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from scoring.llm import (
    NORMALIZE_PRED_SCHEMA,
    get_responses,
    load_responses,
)
from scoring.metrics import is_unintelligible
from scoring.normalize_gold import GOLD_CACHE_DIR

TARGET_LOCALES = ["en-US", "es-MX", "tr-TR", "vi-VN", "zh-CN"]


def load_ground_truth_from_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Load ground truth transcripts from manifest.json.

    Returns: {locale: {utterance_id: transcript}}
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    gt = {}
    for utt in manifest["utterances"]:
        locale = utt["locale"]
        if locale not in gt:
            gt[locale] = {}
        gt[locale][utt["id"]] = utt["transcript"]
    return gt


def load_canonical_gold(cache_dir: Path) -> dict[tuple[str, str], str]:
    """Load canonical normalized gold from ``submissions/normalized/_gold``.

    Returns a mapping ``{(locale, utterance_id): normalized_gold_text}``.
    Missing files are skipped (caller decides whether that's fatal).
    """
    gold: dict[tuple[str, str], str] = {}
    if not cache_dir.exists():
        return gold
    for locale_dir in cache_dir.iterdir():
        if not locale_dir.is_dir():
            continue
        locale = locale_dir.name
        for gf in locale_dir.glob("*.gold.txt"):
            uid = gf.name[: -len(".gold.txt")]
            gold[(locale, uid)] = gf.read_text(encoding="utf-8").strip()
    return gold


def load_transcript_pairs(
    submission_dir: Path, ground_truth: dict[str, dict[str, str]]
) -> list[tuple[str, str, str, str]]:
    """Load paired ground truth and submission transcripts.

    Returns list of (locale, utterance_id, gold_text, predicted_text).
    """
    rows = []
    for locale in TARGET_LOCALES:
        if locale not in ground_truth:
            continue
        sub_locale_dir = submission_dir / locale
        if not sub_locale_dir.exists():
            continue

        for txt_file in sorted(sub_locale_dir.glob("*.txt")):
            utterance_id = txt_file.stem
            if utterance_id not in ground_truth[locale]:
                continue

            gold = ground_truth[locale][utterance_id]
            predicted = txt_file.read_text(encoding="utf-8").strip()
            rows.append((locale, utterance_id, gold, predicted))

    return rows


def _load_pred_prompt() -> tuple[str, dict]:
    """Return ``(prompt_template, response_schema)`` for prediction normalization."""
    from scoring.prompts import NORMALIZE_PRED_AGAINST_GOLD_PROMPT  # type: ignore[attr-defined]

    return NORMALIZE_PRED_AGAINST_GOLD_PROMPT, NORMALIZE_PRED_SCHEMA


def main():
    parser = argparse.ArgumentParser(description="LLM-normalize submission predictions against canonical gold")
    parser.add_argument("--submission-dir", type=Path, required=True, help="Submission directory")
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
        help="Output directory (default: submissions/normalized/<name>)",
    )
    parser.add_argument(
        "--gold-cache-dir",
        type=Path,
        default=GOLD_CACHE_DIR,
        help=(
            "Canonical gold cache dir (default: submissions/normalized/_gold). "
            "Produced by `python -m scoring.normalize_gold`."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=15, help="Parallel LLM workers")
    parser.add_argument("--locales", nargs="+", default=None, help="Limit to specific locales (e.g. en-US zh-CN)")
    args = parser.parse_args()

    if args.locales:
        global TARGET_LOCALES
        TARGET_LOCALES = [loc for loc in TARGET_LOCALES if loc in args.locales]

    submission_dir = args.submission_dir.resolve()
    manifest_path = args.manifest.resolve()

    # Determine output directory
    if args.output_dir:
        normalized_dir = args.output_dir.resolve()
    else:
        submission_name = submission_dir.name
        normalized_dir = Path("submissions/normalized") / submission_name
        normalized_dir = normalized_dir.resolve()

    gold_cache_dir = args.gold_cache_dir.resolve()

    print(f"Submission dir: {submission_dir}")
    print(f"Output dir:     {normalized_dir}")
    print(f"Gold cache:     {gold_cache_dir}")

    # Load ground truth from manifest
    ground_truth = load_ground_truth_from_manifest(manifest_path)

    # Load the canonical gold cache (may be empty if normalize_gold hasn't run).
    canonical_gold = load_canonical_gold(gold_cache_dir)
    if canonical_gold:
        print(f"Loaded {len(canonical_gold)} canonical normalized gold entries.")
    else:
        print(
            "WARNING: canonical gold cache is empty. Run `python -m scoring.normalize_gold` "
            "first for item-1 fairness. Falling back to raw manifest gold in prompts."
        )

    # Load pairs
    print("Loading transcript pairs...")
    pairs = load_transcript_pairs(submission_dir, ground_truth)
    print(f"Loaded {len(pairs)} pairs")

    if not pairs:
        print("No pairs found")
        return

    # Exclusion tracking
    excluded_path = normalized_dir / "excluded_utterances.csv"
    excluded_exists = excluded_path.exists()
    existing_exclusions = set()
    if excluded_exists:
        with open(excluded_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_exclusions.add((row["locale"], row["filename"]))

    # Filter pairs
    to_normalize = []
    skipped = 0
    excluded_count = 0
    skipped_empty = 0

    normalized_dir.mkdir(parents=True, exist_ok=True)

    with open(excluded_path, "a", newline="", encoding="utf-8") as exc_file:
        exc_writer = csv.writer(exc_file)
        if not excluded_exists:
            exc_writer.writerow(["locale", "filename", "reason"])

        for locale, utterance_id, gold, predicted in pairs:
            out_file = normalized_dir / locale / f"{utterance_id}.txt"
            filename = f"{utterance_id}.txt"

            if out_file.exists():
                skipped += 1
                continue

            if is_unintelligible(gold):
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text(predicted, encoding="utf-8")
                if (locale, filename) not in existing_exclusions:
                    exc_writer.writerow([locale, filename, "unintelligible_ground_truth"])
                    existing_exclusions.add((locale, filename))
                excluded_count += 1
                continue

            if not predicted:
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text("", encoding="utf-8")
                skipped_empty += 1
                continue

            to_normalize.append((locale, utterance_id, gold, predicted))

    if skipped > 0:
        print(f"Skipping {skipped} already-normalized files")
    if excluded_count > 0:
        print(f"Excluded {excluded_count} unintelligible utterances")
    if skipped_empty > 0:
        print(f"Skipped {skipped_empty} empty predictions (wrote empty files)")

    if not to_normalize:
        print("All files already normalized")
        return

    prompt_template, response_schema = _load_pred_prompt()

    total = len(to_normalize)
    batch_size = 100
    print(f"Normalizing {total} pairs with {args.num_workers} workers in batches of {batch_size}...")

    # Comparison CSV for visual review
    comparison_path = normalized_dir / "comparison.csv"
    comparison_exists = comparison_path.exists()

    saved = 0
    failed = 0

    with open(comparison_path, "a", newline="", encoding="utf-8") as comp_file:
        comp_writer = csv.writer(comp_file)
        if not comparison_exists:
            comp_writer.writerow(
                [
                    "locale",
                    "filename",
                    "ground_truth",
                    "raw_prediction",
                    "normalized_prediction",
                ]
            )

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = to_normalize[batch_start:batch_end]
            print(f"Batch {batch_start + 1}-{batch_end} / {total}...")

            prompts = []
            for locale, uid, gold, predicted in batch:
                # Prefer the canonical normalized gold as the reference
                # string passed to the LLM; fall back to the raw manifest
                # gold if the cache isn't populated yet.
                gold_ref = canonical_gold.get((locale, uid), gold)
                prompts.append(
                    prompt_template.format(
                        expected_transcript=gold_ref,
                        actual_transcript=predicted,
                        locale=locale,
                    )
                )

            try:
                responses = get_responses(
                    prompts,
                    num_workers=args.num_workers,
                    response_format=response_schema,
                )
                loaded = load_responses(responses)
            except Exception as e:
                print(f"Batch failed: {e}. Skipping batch, will retry on next run.")
                failed += len(batch)
                continue

            for i, (locale, utterance_id, gold, predicted) in enumerate(batch):
                filename = f"{utterance_id}.txt"
                out_file = normalized_dir / locale / filename

                norm_pred = None
                if i < len(loaded) and isinstance(loaded[i], dict):
                    norm_pred = loaded[i].get("normalized_actual")

                if norm_pred is None:
                    failed += 1
                    print(f"  Failed: {locale}/{utterance_id}")
                    continue

                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text(norm_pred, encoding="utf-8")

                comp_writer.writerow([locale, filename, gold, predicted, norm_pred])
                saved += 1

            print(f"  Progress: {saved + failed}/{total} done ({saved} saved, {failed} failed)")

    print(f"\nSaved {saved} normalized files to {normalized_dir}")
    print(f"Failed: {failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
