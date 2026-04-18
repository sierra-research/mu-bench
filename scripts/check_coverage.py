"""Hard-stop coverage gate for the fairness rollout.

For a transcribed output directory (raw or variance-run), checks that:
  1. The number of `<locale>/<utterance_id>.txt` files matches the
     manifest's expected count for every locale we care about.
  2. The `latency.json` file exists and its keyset (or
     `measurements` keyset, for new-schema files) matches the
     transcripts on disk exactly.

If anything is off, prints the missing utterance IDs (set diff) and
exits non-zero. NO retries, NO patching, NO filling in placeholders --
the operator decides whether to investigate the provider API / network /
keys before re-running the failing wave.

Usage:
    .venv/bin/python scripts/check_coverage.py <output_dir> --expected full
    .venv/bin/python scripts/check_coverage.py <output_dir> --expected variance
    .venv/bin/python scripts/check_coverage.py <output_dir> --locales en-US zh-CN

`--expected full` checks all five locales (4,270 utts total).
`--expected variance` checks en-US + zh-CN only (1,657 utts total).
`--locales` overrides both with an explicit list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "manifest.json"

VARIANCE_LOCALES = ["en-US", "zh-CN"]


def load_expected(manifest_path: Path) -> dict[str, set[str]]:
    """Return {locale: set(utterance_ids)} from the manifest."""
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    expected: dict[str, set[str]] = {}
    for utt in manifest["utterances"]:
        expected.setdefault(utt["locale"], set()).add(utt["id"])
    return expected


def collect_transcripts(output_dir: Path, locales: list[str]) -> dict[str, set[str]]:
    """Return {locale: set(utterance_ids_with_a_txt_file)}."""
    found: dict[str, set[str]] = {}
    for locale in locales:
        loc_dir = output_dir / locale
        if not loc_dir.is_dir():
            found[locale] = set()
            continue
        found[locale] = {p.stem for p in loc_dir.glob("*.txt")}
    return found


def collect_latency_keys(latency_path: Path) -> dict[str, set[str]] | None:
    """Return {locale: set(utterance_ids)} from latency.json.

    Accepts both the legacy flat schema and the new schema with
    `measurements`. Returns None if the file is missing.
    """
    if not latency_path.is_file():
        return None
    with latency_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "measurements" in data:
        keys = data["measurements"].keys()
    else:
        keys = data.keys()
    by_locale: dict[str, set[str]] = {}
    for key in keys:
        if "/" not in key:
            continue
        locale, utt_id = key.split("/", 1)
        by_locale.setdefault(locale, set()).add(utt_id)
    return by_locale


def check(
    output_dir: Path,
    locales: list[str],
    expected: dict[str, set[str]],
) -> int:
    """Run the coverage check. Returns 0 on success, 1 on failure."""
    print(f"Coverage gate: {output_dir}")
    print(f"  Locales checked: {', '.join(locales)}")

    found_transcripts = collect_transcripts(output_dir, locales)
    latency_path = output_dir / "latency.json"
    found_latency = collect_latency_keys(latency_path)

    failures: list[str] = []
    total_expected = 0
    total_transcripts = 0
    total_latency = 0

    for locale in locales:
        if locale not in expected:
            failures.append(f"  [{locale}] NOT IN MANIFEST")
            continue
        exp_ids = expected[locale]
        total_expected += len(exp_ids)

        tx_ids = found_transcripts.get(locale, set())
        total_transcripts += len(tx_ids)
        missing_tx = exp_ids - tx_ids
        extra_tx = tx_ids - exp_ids

        print(
            f"  [{locale}] transcripts: {len(tx_ids)}/{len(exp_ids)}"
            + (f" MISSING={len(missing_tx)}" if missing_tx else "")
            + (f" EXTRA={len(extra_tx)}" if extra_tx else "")
        )
        if missing_tx:
            sample = sorted(missing_tx)[:10]
            failures.append(f"  [{locale}] missing {len(missing_tx)} transcripts; first 10: {sample}")
        if extra_tx:
            sample = sorted(extra_tx)[:10]
            failures.append(f"  [{locale}] extra {len(extra_tx)} transcripts not in manifest; first 10: {sample}")

        if found_latency is None:
            failures.append(f"  [{locale}] latency.json is MISSING (expected at {latency_path})")
            continue
        lat_ids = found_latency.get(locale, set())
        total_latency += len(lat_ids)
        missing_lat = exp_ids - lat_ids
        extra_lat = lat_ids - exp_ids
        tx_no_lat = tx_ids - lat_ids
        lat_no_tx = lat_ids - tx_ids

        print(
            f"  [{locale}] latency:     {len(lat_ids)}/{len(exp_ids)}"
            + (f" MISSING={len(missing_lat)}" if missing_lat else "")
            + (f" EXTRA={len(extra_lat)}" if extra_lat else "")
        )
        if missing_lat:
            sample = sorted(missing_lat)[:10]
            failures.append(f"  [{locale}] missing {len(missing_lat)} latency entries; first 10: {sample}")
        if tx_no_lat:
            sample = sorted(tx_no_lat)[:10]
            failures.append(f"  [{locale}] {len(tx_no_lat)} transcripts have no latency entry; first 10: {sample}")
        if lat_no_tx:
            sample = sorted(lat_no_tx)[:10]
            failures.append(f"  [{locale}] {len(lat_no_tx)} latency entries have no transcript; first 10: {sample}")

    print(
        f"  TOTAL transcripts: {total_transcripts}/{total_expected}   TOTAL latency: {total_latency}/{total_expected}"
    )

    if failures:
        print()
        print("COVERAGE GATE FAILED. Do not proceed to scoring.")
        for f in failures:
            print(f)
        print()
        print(
            "Per the rollout plan, we do NOT patch failures. Investigate the "
            "provider API / network / keys, then re-run the failing wave from "
            "scratch."
        )
        return 1

    print("  OK -- coverage gate passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="Provider output directory (raw or variance run)")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to manifest.json (default: {DEFAULT_MANIFEST.relative_to(REPO_ROOT)})",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--expected",
        choices=("full", "variance"),
        help="Preset locale set: 'full' = all 5 locales (4,270 utts); 'variance' = en-US + zh-CN (1,657 utts)",
    )
    group.add_argument(
        "--locales",
        nargs="+",
        help="Explicit locale list (overrides --expected)",
    )
    args = parser.parse_args(argv)

    if not args.output_dir.is_dir():
        print(f"ERROR: output_dir does not exist: {args.output_dir}")
        return 2
    if not args.manifest.is_file():
        print(f"ERROR: manifest not found: {args.manifest}")
        return 2

    expected = load_expected(args.manifest)
    if args.locales:
        locales = args.locales
    elif args.expected == "variance":
        locales = VARIANCE_LOCALES
    else:
        locales = sorted(expected.keys())

    return check(args.output_dir, locales, expected)


if __name__ == "__main__":
    sys.exit(main())
