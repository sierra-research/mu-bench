"""Audit scoring judge-config consistency across published providers.

Walks ``results/*/scores.json`` and reports any disagreement in the
``judge`` block — model snapshot, temperature, seed, or prompt SHAs.

This protects against partial re-scoring batches where only some
providers are re-scored under a new judge/prompt and the leaderboard
silently mixes apples and oranges. Exits non-zero when drift is found,
so CI can enforce consistency.

Usage:
    python scripts/check_judge_drift.py
    python scripts/check_judge_drift.py --results-dir results --strict
"""

import argparse
import json
import sys
from pathlib import Path

JUDGE_FIELDS = [
    "model",
    "modelSnapshot",
    "temperature",
    "seed",
    "normalizeGoldPromptSha",
    "normalizePredPromptSha",
    "significantErrorsPromptSha",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check judge-config consistency across results/*/scores.json")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any provider is missing a judge block or any fields disagree.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}")
        return 1

    judge_blocks: dict[str, dict] = {}
    missing: list[str] = []

    for scores_path in sorted(results_dir.glob("*/scores.json")):
        provider = scores_path.parent.name
        with open(scores_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
        judge = scores.get("judge")
        if not judge:
            missing.append(provider)
            continue
        judge_blocks[provider] = judge

    if not judge_blocks and not missing:
        print("No provider scores.json files found.")
        return 0

    if missing:
        print(f"Missing judge block in {len(missing)} provider(s): {', '.join(missing)}")

    # Collect the set of values seen per field across all providers.
    per_field: dict[str, dict[object, list[str]]] = {f: {} for f in JUDGE_FIELDS}
    for provider, judge in judge_blocks.items():
        for field in JUDGE_FIELDS:
            val = judge.get(field)
            per_field[field].setdefault(val, []).append(provider)

    drift_found = False
    print("\nJudge config fields:")
    for field in JUDGE_FIELDS:
        buckets = per_field[field]
        if len(buckets) == 1:
            (val,) = buckets.keys()
            print(f"  {field}: {val}  (all {sum(len(v) for v in buckets.values())} providers agree)")
        else:
            drift_found = True
            print(f"  {field}: DRIFT across providers")
            for val, providers in buckets.items():
                print(f"    {val!r}: {', '.join(sorted(providers))}")

    if not drift_found and not missing:
        print("\nAll providers agree on judge config.")
        return 0

    if args.strict:
        print("\nStrict mode: exiting non-zero due to drift or missing blocks.")
        return 1
    print("\nDrift or missing blocks detected (warning only; pass --strict to fail CI).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
