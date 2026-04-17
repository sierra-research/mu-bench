"""Build the canonical gold normalization cache.

Reads ``manifest.json`` and LLM-normalizes each ground-truth transcript
**blind to any submission's prediction**. The output is committed to the
repo under ``submissions/normalized/_gold/<locale>/<utterance_id>.gold.txt``
and reused by every submission's scoring run, so provider A and provider B
are always compared against the same reference string (item 1 of the
fairness-fixes plan).

A ``submissions/normalized/_gold/manifest_gold_hash.txt`` file records a
SHA-256 over the manifest gold tuples; ``scoring.score`` reads it into
each ``scores.json`` and recomputes when it changes.

Usage:
    # Produce / refresh the canonical gold cache (requires OPENAI_API_KEY
    # and scoring/prompts.py — not run by the code-only PR).
    python -m scoring.normalize_gold --manifest manifest.json
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from scoring.llm import (
    NORMALIZE_GOLD_SCHEMA,
    NORMALIZE_SCHEMA,
    get_responses,
    load_responses,
)
from scoring.metrics import _load_gold_prompt, is_unintelligible

TARGET_LOCALES = ["en-US", "es-MX", "tr-TR", "vi-VN", "zh-CN"]

GOLD_CACHE_DIR = Path("submissions/normalized/_gold")
GOLD_HASH_FILENAME = "manifest_gold_hash.txt"


def load_manifest_gold(manifest_path: Path) -> list[tuple[str, str, str]]:
    """Load ``(locale, utterance_id, gold)`` triples from the manifest in
    deterministic sorted order.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    rows = [(utt["locale"], utt["id"], utt["transcript"]) for utt in manifest["utterances"]]
    rows.sort()
    return rows


def compute_manifest_gold_hash(rows: list[tuple[str, str, str]]) -> str:
    """Return a sha256 hex digest over sorted manifest gold tuples.

    Used as the cache-invalidation key: scoring runs record this hash in
    ``scores.json`` and recompute when the manifest gold changes.
    """
    h = hashlib.sha256()
    for locale, uid, gold in rows:
        h.update(locale.encode("utf-8"))
        h.update(b"\0")
        h.update(uid.encode("utf-8"))
        h.update(b"\0")
        h.update((gold or "").encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def read_cached_gold_hash(cache_dir: Path) -> str | None:
    """Return the previously cached manifest-gold hash, or None."""
    hash_path = cache_dir / GOLD_HASH_FILENAME
    if not hash_path.exists():
        return None
    return hash_path.read_text(encoding="utf-8").strip() or None


def write_cached_gold_hash(cache_dir: Path, digest: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / GOLD_HASH_FILENAME).write_text(digest + "\n", encoding="utf-8")


def _format_gold_prompt(prompt_template: str, gold: str) -> str:
    """Format the gold-normalization prompt.

    Works with both the new single-input prompt (``NORMALIZE_GOLD_PROMPT``
    expects only ``{expected_transcript}``) and the legacy two-input prompt
    (``NORMALIZE_AGAINST_GOLD_PROMPT`` expects both ``{expected_transcript}``
    and ``{actual_transcript}`` — we pass an empty prediction). This lets
    the code-only PR land before the secret prompts file is updated.
    """
    try:
        return prompt_template.format(expected_transcript=gold)
    except KeyError:
        return prompt_template.format(expected_transcript=gold, actual_transcript="")


def _response_schema_for(prompt_template: str) -> dict:
    """Pick the JSON schema that matches the prompt template in use."""
    # The new single-input prompt only produces normalized_expected. The
    # legacy two-input prompt produces both expected and actual; we still
    # honor its schema so calls don't fail validation.
    if "{actual_transcript}" in prompt_template:
        return NORMALIZE_SCHEMA
    return NORMALIZE_GOLD_SCHEMA


def _extract_normalized_gold(resp: object) -> str | None:
    if not isinstance(resp, dict):
        return None
    return resp.get("normalized_expected")


def main():
    parser = argparse.ArgumentParser(description="LLM-normalize manifest gold into a canonical cache")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifest.json"),
        help="Path to manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GOLD_CACHE_DIR,
        help="Canonical gold cache directory (default: submissions/normalized/_gold)",
    )
    parser.add_argument("--num-workers", type=int, default=15, help="Parallel LLM workers")
    parser.add_argument(
        "--locales",
        nargs="+",
        default=None,
        help="Limit to specific locales (e.g. en-US zh-CN). Useful for partial rebuilds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-normalize all utterances even if the manifest-gold hash matches the cached value.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    cache_dir = args.output_dir.resolve()

    rows = load_manifest_gold(manifest_path)
    if args.locales:
        locales_filter = set(args.locales)
        rows = [(loc, uid, gold) for (loc, uid, gold) in rows if loc in locales_filter]

    new_hash = compute_manifest_gold_hash(rows)
    cached_hash = read_cached_gold_hash(cache_dir)

    print(f"Manifest gold: {len(rows)} utterances across {len({r[0] for r in rows})} locales")
    print(f"New hash:      {new_hash}")
    print(f"Cached hash:   {cached_hash or '<none>'}")

    if cached_hash == new_hash and not args.force:
        # Hash match — every utterance's gold is still canonical. Only
        # re-normalize utterances whose files are actually missing from disk
        # (so a partial rebuild is idempotent and fast).
        to_normalize = [
            (loc, uid, gold)
            for (loc, uid, gold) in rows
            if not is_unintelligible(gold) and gold and not (cache_dir / loc / f"{uid}.gold.txt").exists()
        ]
        if not to_normalize:
            print("Canonical gold cache is up to date, nothing to do.")
            return
        print(f"Hash matches but {len(to_normalize)} gold files are missing on disk; filling those only.")
    else:
        to_normalize = [(loc, uid, gold) for (loc, uid, gold) in rows if not is_unintelligible(gold) and gold]
        print(f"Hash changed (or --force) — re-normalizing {len(to_normalize)} gold entries.")

    # For unintelligible or empty golds, write an empty ``.gold.txt`` so
    # downstream code doesn't have to special-case missing files.
    skipped_special = 0
    for locale, uid, gold in rows:
        if is_unintelligible(gold) or not gold:
            out = cache_dir / locale / f"{uid}.gold.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                out.write_text("", encoding="utf-8")
                skipped_special += 1
    if skipped_special:
        print(f"Wrote {skipped_special} empty gold files for unintelligible / silent entries.")

    if not to_normalize:
        write_cached_gold_hash(cache_dir, new_hash)
        print("Nothing to LLM-normalize.")
        return

    prompt_template = _load_gold_prompt()
    response_format = _response_schema_for(prompt_template)

    total = len(to_normalize)
    batch_size = 100
    print(f"Normalizing {total} gold entries with {args.num_workers} workers in batches of {batch_size}...")

    audit_path = cache_dir / "gold_normalization.csv"
    audit_exists = audit_path.exists()
    cache_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    failed = 0

    with open(audit_path, "a", newline="", encoding="utf-8") as audit_file:
        audit_writer = csv.writer(audit_file)
        if not audit_exists:
            audit_writer.writerow(["locale", "filename", "raw_gold", "normalized_gold"])

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = to_normalize[batch_start:batch_end]
            print(f"Batch {batch_start + 1}-{batch_end} / {total}...")

            prompts = [_format_gold_prompt(prompt_template, gold) for _, _, gold in batch]
            try:
                responses = get_responses(
                    prompts,
                    num_workers=args.num_workers,
                    response_format=response_format,
                )
                loaded = load_responses(responses)
            except Exception as e:
                print(f"Batch failed: {e}. Skipping batch, will retry on next run.")
                failed += len(batch)
                continue

            for i, (locale, uid, gold) in enumerate(batch):
                norm_gold = None
                if i < len(loaded):
                    norm_gold = _extract_normalized_gold(loaded[i])
                if norm_gold is None:
                    failed += 1
                    print(f"  Failed: {locale}/{uid}")
                    continue

                out_file = cache_dir / locale / f"{uid}.gold.txt"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text(norm_gold, encoding="utf-8")
                audit_writer.writerow([locale, f"{uid}.gold.txt", gold, norm_gold])
                saved += 1

            print(f"  Progress: {saved + failed}/{total} done ({saved} saved, {failed} failed)")

    if failed == 0:
        write_cached_gold_hash(cache_dir, new_hash)
        print(f"\nSaved {saved} canonical gold files to {cache_dir}")
        print(f"Updated {cache_dir / GOLD_HASH_FILENAME} -> {new_hash}")
    else:
        print(f"\nSaved {saved} canonical gold files; {failed} failed. Hash not updated; re-run to finish.")


if __name__ == "__main__":
    main()
