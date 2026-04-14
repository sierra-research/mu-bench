#!/usr/bin/env python3
"""Rebuild the audio dataset from legacy source recordings.

For every conversation in the manifest, uses mapping.json to find the legacy
audit directory, reads the annotation CSV (with timestamps), filters out
unintelligible rows, and clips per-turn WAV files from in.wav.

Each output turn also gets a .txt sidecar with the CSV ground-truth transcript
so it can be compared against the current manifest GT.

Usage:
    python scripts/rebuild_audio_dataset.py \
        --mapping ~/Downloads/mapping.json \
        --manifest manifest.json \
        --output-dir hf_new
"""

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

ORG = "org-01JNM7NWHE7MSR8H5409BBGTPN"
LEGACY_BASE = "legacy/audio"


def parse_timestamp(ts: str) -> float:
    parts = ts.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def clip_with_ffmpeg(in_path: str, out_path: str, start: float, end: float):
    duration = end - start
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            in_path,
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-c",
            "copy",
            out_path,
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Rebuild audio dataset from legacy sources")
    parser.add_argument("--mapping", required=True, help="Path to mapping.json")
    parser.add_argument("--manifest", default="manifest.json", help="Path to manifest.json")
    parser.add_argument("--output-dir", default="hf_new", help="Output directory")
    parser.add_argument("--legacy-dir", default=LEGACY_BASE, help="Legacy audio root")
    parser.add_argument("--locale", default=None, help="Process only this locale")
    args = parser.parse_args()

    with open(args.mapping) as f:
        mapping = json.load(f)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_convs = {}
    for u in manifest["utterances"]:
        key = (u["locale"], u["conversation_id"])
        manifest_convs.setdefault(key, {})[u["turn_index"]] = u["transcript"]

    locales = [args.locale] if args.locale else manifest["locales"]

    clipped = 0
    skipped = 0
    errors = 0
    gt_mismatches = 0

    for locale in locales:
        for conv_id in sorted(mapping.get(locale, {}).keys(), key=lambda x: int(x.split("-")[1])):
            audit_id = mapping[locale][conv_id]
            audit_dir = Path(args.legacy_dir) / locale / ORG / audit_id
            if not audit_dir.is_dir():
                print(f"  SKIP (no dir): {audit_dir}")
                skipped += 1
                continue

            in_wav = audit_dir / "in.wav"
            if not in_wav.exists():
                print(f"  SKIP (no in.wav): {audit_dir}")
                skipped += 1
                continue

            csv_files = [f for f in os.listdir(audit_dir) if f.endswith(".csv")]
            if not csv_files:
                print(f"  SKIP (no CSV): {audit_dir}")
                skipped += 1
                continue

            csv_path = audit_dir / csv_files[0]
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader)
                rows = list(reader)

            filtered = [(i, row) for i, row in enumerate(rows) if "<unintelligible>" not in row[2].lower()]

            gt = manifest_convs.get((locale, conv_id), {})
            if len(filtered) != len(gt):
                print(f"  ERROR {locale}/{conv_id}: filtered CSV rows ({len(filtered)}) != manifest turns ({len(gt)})")
                errors += 1
                continue

            out_locale_dir = Path(args.output_dir) / locale
            out_locale_dir.mkdir(parents=True, exist_ok=True)

            for (csv_idx, csv_row), (turn_idx, manifest_gt) in zip(filtered, sorted(gt.items())):
                start = parse_timestamp(csv_row[0])
                end = parse_timestamp(csv_row[1])
                csv_transcript = csv_row[2].strip()

                out_wav = out_locale_dir / f"{conv_id}-turn-{turn_idx}.wav"
                out_txt = out_locale_dir / f"{conv_id}-turn-{turn_idx}.txt"

                try:
                    clip_with_ffmpeg(str(in_wav), str(out_wav), start, end)
                    clipped += 1
                except subprocess.CalledProcessError as e:
                    print(f"  ERROR clipping {locale}/{conv_id}-turn-{turn_idx}: {e}")
                    errors += 1
                    continue

                out_txt.write_text(csv_transcript, encoding="utf-8")

                if csv_transcript != manifest_gt:
                    print(
                        f"  GT MISMATCH {locale}/{conv_id}-turn-{turn_idx}:\n"
                        f"    CSV: {csv_transcript[:70]}\n"
                        f"    GT:  {manifest_gt[:70]}"
                    )
                    gt_mismatches += 1

            if (clipped % 200 == 0) and clipped > 0:
                print(f"  Progress: {clipped} clipped, {errors} errors")

        print(f"{locale}: done ({clipped} clipped so far)")

    print(f"\nTotal: {clipped} clipped, {skipped} skipped, {errors} errors")
    print(f"GT mismatches: {gt_mismatches}")
    if gt_mismatches == 0:
        print("All CSV transcripts match manifest GT exactly.")


if __name__ == "__main__":
    main()
