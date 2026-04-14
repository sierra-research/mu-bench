#!/usr/bin/env python3
"""
Generate metadata.jsonl for HuggingFace from manifest.json.

Usage:
    python scripts/generate_hf_metadata.py [--manifest manifest.json] [--output metadata.jsonl]
"""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate HF metadata.jsonl from manifest")
    parser.add_argument("--manifest", default="manifest.json", type=Path)
    parser.add_argument("--output", default="metadata.jsonl", type=Path)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(args.output, "w", encoding="utf-8") as out:
        for u in manifest["utterances"]:
            conv_match = re.search(r"(\d+)", u["conversation_id"])
            conv_id = int(conv_match.group(1)) if conv_match else u["conversation_id"]

            row = {
                "file_name": u["audio_path"].removeprefix("audio/"),
                "locale": u["locale"],
                "conversation_id": conv_id,
                "turn_index": u["turn_index"],
                "duration_sec": u["duration_sec"],
                "transcript": u["transcript"],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Written {len(manifest['utterances'])} rows to {args.output}")


if __name__ == "__main__":
    main()
