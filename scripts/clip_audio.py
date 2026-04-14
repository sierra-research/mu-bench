"""Clip full-call audio into per-turn WAV files using ground truth timestamps.

Reads call CSVs from the downloaded benchmark data and produces per-turn clips
matching the audio_path entries in manifest.json.

Usage:
    python scripts/clip_audio.py
    python scripts/clip_audio.py --locale vi-VN
"""

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


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
    parser = argparse.ArgumentParser(description="Clip audio into per-turn WAVs")
    parser.add_argument("--audio-dir", default="audio", help="Root audio directory")
    parser.add_argument("--output-dir", default="audio/clipped", help="Output for clipped files")
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--locale", default=None, help="Process only this locale")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    convos = {}
    for u in manifest["utterances"]:
        convos.setdefault(u["conversation_id"], []).append(u)

    org = "org-01JNM7NWHE7MSR8H5409BBGTPN"
    locales = [args.locale] if args.locale else manifest["locales"]

    clipped = 0
    skipped = 0
    errors = 0

    for locale in locales:
        locale_convos = {cid: utts for cid, utts in convos.items() if utts[0]["locale"] == locale}
        locale_dir = Path(args.audio_dir) / locale / org

        for conv_id, utts in sorted(locale_convos.items()):
            dirs = [d for d in os.listdir(locale_dir) if d.startswith(conv_id)]
            if not dirs:
                continue

            audit_dir = locale_dir / dirs[0]
            in_wav = audit_dir / "in.wav"
            if not in_wav.exists():
                print(f"  SKIP (no in.wav): {audit_dir}")
                continue

            csv_files = [f for f in os.listdir(audit_dir) if f.startswith("call_") and f.endswith(".csv")]
            if not csv_files:
                continue

            csv_path = audit_dir / csv_files[0]
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                rows = list(reader)

            sorted_utts = sorted(utts, key=lambda x: x["turn_index"])

            for csv_row, utt in zip(rows, sorted_utts):
                start = parse_timestamp(csv_row[0])
                end = parse_timestamp(csv_row[1])
                out_path = Path(args.output_dir) / utt["audio_path"]

                if out_path.exists():
                    skipped += 1
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    clip_with_ffmpeg(str(in_wav), str(out_path), start, end)
                    clipped += 1
                except subprocess.CalledProcessError as e:
                    print(f"  ERROR: {utt['audio_path']}: {e}")
                    errors += 1

            if (clipped + skipped) % 50 == 0 and clipped > 0:
                print(f"  Progress: {clipped} clipped, {skipped} skipped, {errors} errors")

        print(f"{locale}: done ({clipped} clipped so far)")

    print(f"\nTotal: {clipped} clipped, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
