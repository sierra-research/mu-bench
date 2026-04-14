"""Compare transcripts between two submission directories.

Reports exact-match rates overall and per provider, useful for verifying
that a re-run produces consistent results.

Usage:
    python scripts/compare_transcripts.py \
        --old-base submissions/raw \
        --new-base latency_runs \
        --manifest manifest.json
"""

import argparse
import json
from pathlib import Path


def compare_provider(old_dir: Path, new_dir: Path, manifest_ids: dict[str, set[str]]) -> dict:
    """Compare transcripts for a single provider.

    Returns dict with total, matched, mismatched counts and mismatch details.
    """
    total = 0
    matched = 0
    mismatched = 0
    missing_old = 0
    missing_new = 0
    mismatches: list[dict] = []

    for locale, ids in sorted(manifest_ids.items()):
        for uid in sorted(ids):
            old_path = old_dir / locale / f"{uid}.txt"
            new_path = new_dir / locale / f"{uid}.txt"

            if not old_path.exists():
                missing_old += 1
                continue
            if not new_path.exists():
                missing_new += 1
                continue

            total += 1
            old_text = old_path.read_text(encoding="utf-8")
            new_text = new_path.read_text(encoding="utf-8")

            if old_text == new_text:
                matched += 1
            else:
                mismatched += 1
                mismatches.append(
                    {
                        "id": uid,
                        "locale": locale,
                        "old_len": len(old_text),
                        "new_len": len(new_text),
                    }
                )

    return {
        "total": total,
        "matched": matched,
        "mismatched": mismatched,
        "missing_old": missing_old,
        "missing_new": missing_new,
        "match_pct": round(matched / total * 100, 1) if total > 0 else 0,
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare transcripts between two submission bases")
    parser.add_argument(
        "--old-base",
        required=True,
        type=Path,
        help="Base dir of old submissions (e.g., submissions/raw)",
    )
    parser.add_argument(
        "--new-base",
        required=True,
        type=Path,
        help="Base dir of new submissions (e.g., latency_runs)",
    )
    parser.add_argument("--manifest", default="manifest.json", type=Path, help="Path to manifest.json")
    parser.add_argument("--json", type=Path, default=None, help="Write detailed results to JSON file")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_ids: dict[str, set[str]] = {}
    for utt in manifest["utterances"]:
        manifest_ids.setdefault(utt["locale"], set()).add(utt["id"])

    old_providers = {d.name for d in args.old_base.iterdir() if d.is_dir()} if args.old_base.exists() else set()
    new_providers = {d.name for d in args.new_base.iterdir() if d.is_dir()} if args.new_base.exists() else set()
    common = sorted(old_providers & new_providers)

    if not common:
        print("No common providers found between old and new bases.")
        return

    print(f"{'Provider':<30} {'Compared':>10} {'Match':>10} {'Mismatch':>10} {'Match %':>10}")
    print("-" * 72)

    results = {}
    grand_total = 0
    grand_matched = 0
    grand_mismatched = 0

    for provider in common:
        r = compare_provider(args.old_base / provider, args.new_base / provider, manifest_ids)
        results[provider] = r
        grand_total += r["total"]
        grand_matched += r["matched"]
        grand_mismatched += r["mismatched"]

        print(f"{provider:<30} {r['total']:>10} {r['matched']:>10} {r['mismatched']:>10} {r['match_pct']:>9.1f}%")
        if r["missing_old"] or r["missing_new"]:
            print(f"  (missing: {r['missing_old']} old, {r['missing_new']} new)")

    print("-" * 72)
    grand_pct = round(grand_matched / grand_total * 100, 1) if grand_total > 0 else 0
    print(f"{'TOTAL':<30} {grand_total:>10} {grand_matched:>10} {grand_mismatched:>10} {grand_pct:>9.1f}%")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results written to {args.json}")


if __name__ == "__main__":
    main()
