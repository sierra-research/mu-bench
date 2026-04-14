#!/usr/bin/env python3
"""
Fix GT alignment: drop unintelligible turns and re-index.

Audio files and submission files have extra items at positions where
unintelligible turns were dropped from the manifest GT. This script
deletes those extra files and renames subsequent files to close the gaps,
restoring 1:1 alignment between manifest entries and audio/submission files.

The manifest itself does NOT change — only the underlying files shift.

Usage:
    python scripts/fix_alignment.py \
        --mapping ~/Downloads/mapping.json \
        --manifest manifest.json \
        --legacy-dir legacy/audio \
        --submissions-dir submissions \
        --results-dir results \
        --audio-dir audio \
        --dry-run
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher

ORG_ID = "org-01JNM7NWHE7MSR8H5409BBGTPN"
PROVIDERS = [
    "azure",
    "deepgram-nova3",
    "elevenlabs-scribe-v2",
    "google-chirp3",
    "openai-gpt4o-transcribe",
]


def normalize_for_comparison(s):
    s = s.lower().strip()
    s = re.sub(r"<unintelligible>", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_for_comparison(a), normalize_for_comparison(b)).ratio()


def load_legacy_csv(csv_path):
    """Load a legacy annotation CSV, returning list of turn dicts."""
    turns = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            transcript = row[2].strip()
            inaudible_flag = row[3].strip().upper() if len(row) > 3 else ""
            turns.append(
                {
                    "transcript": transcript,
                    "fully_inaudible": inaudible_flag == "TRUE",
                    "has_unintelligible": "<unintelligible>" in transcript.lower(),
                }
            )
    return turns


def find_legacy_csv(legacy_dir, locale, audit_id):
    """Find the CSV file for a given audit ID."""
    audit_path = os.path.join(legacy_dir, locale, ORG_ID, audit_id)
    if not os.path.isdir(audit_path):
        return None
    for fname in os.listdir(audit_path):
        if fname.endswith(".csv"):
            return os.path.join(audit_path, fname)
    return None


def find_dropped_turns(legacy_turns, manifest_turns):
    """
    Align legacy CSV turns against manifest GT to find which legacy
    indices were dropped. Uses DP alignment maximizing total similarity.

    Returns list of legacy indices that were dropped, or empty list
    if alignment quality is too low (different conversation).
    """
    manifest_indices = sorted(manifest_turns.keys())
    manifest_texts = [manifest_turns[i] for i in manifest_indices]
    n_legacy = len(legacy_turns)
    n_manifest = len(manifest_texts)

    if n_legacy <= n_manifest:
        return []

    SIM_THRESHOLD = 0.4

    # Precompute similarity scores
    sim_cache = {}
    for li in range(n_legacy):
        for mi in range(n_manifest):
            s = similarity(legacy_turns[li]["transcript"], manifest_texts[mi])
            if s > SIM_THRESHOLD:
                sim_cache[(li, mi)] = s

    # DP maximizing total similarity score (not just count)
    # dp[li][mi] = best total similarity aligning legacy[li:] with manifest[mi:]
    dp = [[0.0] * (n_manifest + 1) for _ in range(n_legacy + 1)]
    for li in range(n_legacy - 1, -1, -1):
        for mi in range(n_manifest - 1, -1, -1):
            skip = dp[li + 1][mi]
            match_val = sim_cache.get((li, mi), 0.0)
            if match_val > 0:
                match_val += dp[li + 1][mi + 1]
            dp[li][mi] = max(skip, match_val)

    # Check alignment quality
    if dp[0][0] < n_manifest * 0.6:
        return []

    # Trace back to find which legacy indices are matched
    matched_legacy = set()
    li, mi = 0, 0
    while li < n_legacy and mi < n_manifest:
        s = sim_cache.get((li, mi), 0.0)
        skip_score = dp[li + 1][mi]
        match_score = s + dp[li + 1][mi + 1] if s > 0 else 0.0
        if match_score >= skip_score and s > 0:
            matched_legacy.add(li)
            li += 1
            mi += 1
        else:
            li += 1

    dropped = [i for i in range(n_legacy) if i not in matched_legacy]
    return dropped


def build_rename_map(n_legacy, dropped_indices):
    """
    Build mapping: old_index -> new_index (or None for deleted).
    Non-dropped files get sequential indices closing the gaps.
    """
    rename = {}
    new_idx = 0
    dropped_set = set(dropped_indices)
    for old_idx in range(n_legacy):
        if old_idx in dropped_set:
            rename[old_idx] = None
        else:
            rename[old_idx] = new_idx
            new_idx += 1
    return rename


def apply_file_renames(directory, locale, conv_id, rename_map, ext, dry_run):
    """
    Apply rename map to files matching conv-X-turn-Y.{ext} in directory/locale/.
    Uses a temp-name pass to avoid collisions.
    """
    locale_dir = os.path.join(directory, locale)
    if not os.path.isdir(locale_dir):
        return 0, 0

    prefix = f"{conv_id}-turn-"
    deletions = 0
    renames = 0

    existing = {}
    for fname in os.listdir(locale_dir):
        if fname.startswith(prefix) and fname.endswith(f".{ext}"):
            idx_str = fname[len(prefix) : -len(f".{ext}")]
            try:
                idx = int(idx_str)
                existing[idx] = os.path.join(locale_dir, fname)
            except ValueError:
                pass

    # Pass 1: rename to temp names to avoid collisions
    temp_map = {}
    for old_idx, path in sorted(existing.items()):
        new_idx = rename_map.get(old_idx)
        if new_idx is None:
            if not dry_run:
                os.remove(path)
            deletions += 1
        elif new_idx != old_idx:
            temp_name = path + ".tmp_reindex"
            if not dry_run:
                os.rename(path, temp_name)
            temp_map[temp_name] = os.path.join(locale_dir, f"{conv_id}-turn-{new_idx}.{ext}")
            renames += 1

    # Pass 2: rename from temp to final
    for temp_path, final_path in temp_map.items():
        if not dry_run:
            os.rename(temp_path, final_path)

    return deletions, renames


def update_latency_json(latency_path, locale, conv_id, rename_map, dry_run):
    """Update latency.json keys for affected conversation."""
    if not os.path.exists(latency_path):
        return 0, 0

    with open(latency_path) as f:
        latency = json.load(f)

    prefix = f"{locale}/{conv_id}-turn-"
    updated = {}
    deletions = 0
    renames = 0

    for key, value in latency.items():
        if key.startswith(prefix):
            idx_str = key[len(prefix) :]
            try:
                old_idx = int(idx_str)
            except ValueError:
                updated[key] = value
                continue

            new_idx = rename_map.get(old_idx)
            if new_idx is None:
                deletions += 1
            elif new_idx != old_idx:
                updated[f"{locale}/{conv_id}-turn-{new_idx}"] = value
                renames += 1
            else:
                updated[key] = value
        else:
            updated[key] = value

    if not dry_run and (deletions > 0 or renames > 0):
        with open(latency_path, "w") as f:
            json.dump(updated, f, indent=2)

    return deletions, renames


def update_normalized_csv(csv_path, conv_id, rename_map, dry_run):
    """Update filename references in comparison.csv or excluded_utterances.csv."""
    if not os.path.exists(csv_path):
        return False

    with open(csv_path, encoding="utf-8") as f:
        content = f.read()

    original = content
    prefix = f"{conv_id}-turn-"

    for old_idx in sorted(rename_map.keys(), reverse=True):
        new_idx = rename_map[old_idx]
        old_name = f"{prefix}{old_idx}.txt"
        if new_idx is None:
            content = "\n".join(
                line for line in content.split("\n") if old_name not in line or line.startswith("locale,")
            )
        elif new_idx != old_idx:
            new_name = f"{prefix}{new_idx}.txt"
            content = content.replace(old_name, new_name)

    if content != original:
        if not dry_run:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(content)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Fix GT alignment by re-indexing")
    parser.add_argument("--mapping", required=True, help="Path to mapping.json")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--legacy-dir", required=True, help="Path to legacy/audio/")
    parser.add_argument("--submissions-dir", required=True, help="Path to submissions/")
    parser.add_argument("--results-dir", required=True, help="Path to results/")
    parser.add_argument("--audio-dir", default=None, help="Path to audio/ (optional, may not exist)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without doing it",
    )
    args = parser.parse_args()

    mapping = json.load(open(args.mapping))
    manifest = json.load(open(args.manifest))

    manifest_convs = defaultdict(dict)
    for u in manifest["utterances"]:
        manifest_convs[(u["locale"], u["conversation_id"])][u["turn_index"]] = u["transcript"]

    hf_operations = {"delete": [], "rename": {}}
    total_affected = 0
    total_dropped = 0

    print("=" * 60)
    print("SCANNING FOR MISALIGNED CONVERSATIONS")
    print("=" * 60)

    all_rename_maps = {}

    for locale in sorted(mapping.keys()):
        for conv_id in sorted(mapping[locale].keys(), key=lambda x: int(x.split("-")[1])):
            audit_id = mapping[locale][conv_id]
            csv_path = find_legacy_csv(args.legacy_dir, locale, audit_id)

            if not csv_path:
                continue

            gt = manifest_convs.get((locale, conv_id), {})
            if not gt:
                continue

            legacy_turns = load_legacy_csv(csv_path)
            n_legacy = len(legacy_turns)
            n_manifest = len(gt)

            if n_legacy <= n_manifest:
                continue

            # Check if already fixed: count provider submission files
            check_dir = os.path.join(args.submissions_dir, "raw", PROVIDERS[0], locale)
            if os.path.isdir(check_dir):
                prefix = f"{conv_id}-turn-"
                existing_count = sum(1 for f in os.listdir(check_dir) if f.startswith(prefix) and f.endswith(".txt"))
                if existing_count == n_manifest:
                    continue

            dropped = find_dropped_turns(legacy_turns, gt)
            if not dropped:
                continue

            rename_map = build_rename_map(n_legacy, dropped)
            all_rename_maps[(locale, conv_id)] = rename_map
            total_affected += 1
            total_dropped += len(dropped)

            print(f"\n  {locale}/{conv_id}: {n_legacy} legacy -> {n_manifest} manifest")
            print(f"    Dropped turn indices: {dropped}")
            for di in dropped:
                t = legacy_turns[di]["transcript"][:70]
                print(f'      [{di}] "{t}"')

    if not all_rename_maps:
        print("\nNo misaligned conversations found. Nothing to do.")
        return

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_affected} conversations, {total_dropped} turns to drop")
    print(f"{'=' * 60}\n")

    # --- Apply renames ---
    print("APPLYING RENAMES" + (" (DRY RUN)" if args.dry_run else ""))
    print("-" * 60)

    for (locale, conv_id), rename_map in sorted(all_rename_maps.items()):
        print(f"\n  {locale}/{conv_id}:")

        # Audio files
        if args.audio_dir and os.path.isdir(args.audio_dir):
            d, r = apply_file_renames(args.audio_dir, locale, conv_id, rename_map, "wav", args.dry_run)
            if d or r:
                print(f"    audio: {d} deleted, {r} renamed")

        # Build HF operations regardless of local audio
        for old_idx, new_idx in sorted(rename_map.items()):
            old_path = f"{locale}/{conv_id}-turn-{old_idx}.wav"
            if new_idx is None:
                hf_operations["delete"].append(old_path)
            elif new_idx != old_idx:
                new_path = f"{locale}/{conv_id}-turn-{new_idx}.wav"
                hf_operations["rename"][old_path] = new_path

        # Raw submissions
        for provider in PROVIDERS:
            raw_dir = os.path.join(args.submissions_dir, "raw", provider)
            d, r = apply_file_renames(raw_dir, locale, conv_id, rename_map, "txt", args.dry_run)
            if d or r:
                print(f"    raw/{provider}: {d} deleted, {r} renamed")

            # Latency
            latency_path = os.path.join(raw_dir, "latency.json")
            ld, lr = update_latency_json(latency_path, locale, conv_id, rename_map, args.dry_run)
            if ld or lr:
                print(f"    raw/{provider}/latency.json: {ld} deleted, {lr} renamed")

        # Normalized submissions
        for provider in PROVIDERS:
            norm_dir = os.path.join(args.submissions_dir, "normalized", provider)
            d, r = apply_file_renames(norm_dir, locale, conv_id, rename_map, "txt", args.dry_run)
            if d or r:
                print(f"    normalized/{provider}: {d} deleted, {r} renamed")

            for csv_name in ["comparison.csv", "excluded_utterances.csv"]:
                csv_p = os.path.join(norm_dir, csv_name)
                if update_normalized_csv(csv_p, conv_id, rename_map, args.dry_run):
                    print(f"    normalized/{provider}/{csv_name}: updated")

        # Results
        for provider in PROVIDERS:
            details_dir = os.path.join(args.results_dir, provider, "details")
            d, r = apply_file_renames(details_dir, locale, conv_id, rename_map, "json", args.dry_run)
            if d or r:
                print(f"    results/{provider}/details: {d} deleted, {r} renamed")

    # Delete stale aggregated scores
    print(f"\n{'=' * 60}")
    print("DELETING STALE AGGREGATED SCORES")
    print("-" * 60)
    for provider in PROVIDERS:
        scores_path = os.path.join(args.results_dir, provider, "scores.json")
        if os.path.exists(scores_path):
            if not args.dry_run:
                os.remove(scores_path)
            print(f"  Deleted {scores_path}")

    leaderboard_path = os.path.join(args.results_dir, "leaderboard.json")
    if os.path.exists(leaderboard_path):
        if not args.dry_run:
            os.remove(leaderboard_path)
        print(f"  Deleted {leaderboard_path}")

    # Write HF operations
    hf_out = os.path.join(os.path.dirname(args.manifest), "audio_rename_map.json")
    with open(hf_out, "w") as f:
        json.dump(hf_operations, f, indent=2)
    print(f"\nHuggingFace operations written to {hf_out}")
    print(f"  {len(hf_operations['delete'])} deletions")
    print(f"  {len(hf_operations['rename'])} renames")

    # Verification
    print(f"\n{'=' * 60}")
    print("VERIFICATION")
    print("-" * 60)

    errors = 0
    for (locale, conv_id), rename_map in sorted(all_rename_maps.items()):
        gt = manifest_convs[(locale, conv_id)]
        expected_indices = sorted(gt.keys())

        for provider in PROVIDERS:
            raw_dir = os.path.join(args.submissions_dir, "raw", provider, locale)
            if not os.path.isdir(raw_dir):
                continue

            for turn_idx in expected_indices:
                expected = os.path.join(raw_dir, f"{conv_id}-turn-{turn_idx}.txt")
                if not args.dry_run and not os.path.exists(expected):
                    print(f"  MISSING: {expected}")
                    errors += 1

            max_expected = max(expected_indices)
            for fname in os.listdir(raw_dir):
                if not fname.startswith(f"{conv_id}-turn-"):
                    continue
                idx_str = fname.replace(f"{conv_id}-turn-", "").replace(".txt", "")
                try:
                    idx = int(idx_str)
                    if idx > max_expected:
                        print(f"  EXTRA: {os.path.join(raw_dir, fname)}")
                        errors += 1
                except ValueError:
                    pass

    if errors == 0:
        print("  All checks passed!")
    else:
        print(f"  {errors} errors found!")

    print("\nDone.")


if __name__ == "__main__":
    main()
