#!/usr/bin/env python3
"""
Recompute duration_sec for every utterance in manifest.json from the actual audio files.

Usage:
    python scripts/fix_manifest_durations.py [--manifest manifest.json] [--dry-run]
"""

import argparse
import json
from pathlib import Path

import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="Fix manifest duration_sec from audio files")
    parser.add_argument("--manifest", default="manifest.json", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    changed = 0
    max_diff = 0.0
    max_diff_id = ""

    for u in manifest["utterances"]:
        audio_path = Path(u["audio_path"])
        if not audio_path.exists():
            print(f"  WARNING: missing {audio_path}")
            continue

        info = sf.info(str(audio_path))
        actual = round(info.frames / info.samplerate, 3)
        old = u["duration_sec"]
        diff = abs(actual - old)

        if diff > 0.0005:
            changed += 1
            if diff > max_diff:
                max_diff = diff
                max_diff_id = f"{u['locale']}/{u['id']}"
            u["duration_sec"] = actual

    print(f"Updated {changed}/{len(manifest['utterances'])} durations")
    if changed:
        print(f"Max diff: {max_diff:.4f}s ({max_diff_id})")

    if not args.dry_run:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Written to {manifest_path}")
    else:
        print("Dry run — no changes written")


if __name__ == "__main__":
    main()
