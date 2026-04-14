"""Compute p50 and p95 latency statistics from latency.json files.

Reads latency.json from each provider directory, computes per-locale and
overall percentile statistics, and optionally merges them into scores.json.

Usage:
    # Single provider
    python scripts/latency_stats.py \
        --submission-dir latency_runs/deepgram-nova3 \
        --manifest manifest.json \
        --output-dir results/deepgram-nova3

    # All providers under a base directory
    python scripts/latency_stats.py \
        --base-dir latency_runs \
        --manifest manifest.json \
        --results-dir results
"""

import argparse
import json
import math
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def compute_latency_stats(
    latency: dict[str, float],
    manifest: dict,
    *,
    drop_first: int = 0,
    min_latency_ms: float = 0,
) -> dict:
    """Compute per-locale and overall p50/p95 from latency data.

    Args:
        drop_first: Skip the first N entries (cold-start warmup artifacts).
        min_latency_ms: Drop entries below this threshold (likely cache hits).

    Returns dict with structure:
        {
            "locales": {
                "en-US": {"latencyP50Ms": ..., "latencyP95Ms": ..., "count": ...},
                ...
            },
            "overall": {"latencyP50Ms": ..., "latencyP95Ms": ..., "count": ...},
            "filtered": {"dropped_warmup": ..., "dropped_min": ..., "total_kept": ...}
        }
    """
    locale_map: dict[str, str] = {}
    for utt in manifest["utterances"]:
        locale_map[utt["id"]] = utt["locale"]

    items = list(latency.items())
    dropped_warmup = min(drop_first, len(items))
    items = items[dropped_warmup:]

    dropped_min = 0
    by_locale: dict[str, list[float]] = {}
    all_values: list[float] = []

    for key, ms in items:
        if ms < min_latency_ms:
            dropped_min += 1
            continue
        if "/" in key:
            locale = key.split("/", 1)[0]
        else:
            locale = locale_map.get(key)
            if locale is None:
                continue
        by_locale.setdefault(locale, []).append(ms)
        all_values.append(ms)

    result: dict = {"locales": {}}
    for locale in sorted(by_locale.keys()):
        vals = by_locale[locale]
        result["locales"][locale] = {
            "latencyP50Ms": round(percentile(vals, 50), 1),
            "latencyP95Ms": round(percentile(vals, 95), 1),
            "count": len(vals),
        }

    if all_values:
        result["overall"] = {
            "latencyP50Ms": round(percentile(all_values, 50), 1),
            "latencyP95Ms": round(percentile(all_values, 95), 1),
            "count": len(all_values),
        }

    result["filtered"] = {
        "dropped_warmup": dropped_warmup,
        "dropped_min": dropped_min,
        "total_kept": len(all_values),
    }

    return result


def process_provider(
    submission_dir: Path,
    manifest: dict,
    output_dir: Path | None,
    *,
    drop_first: int = 0,
    min_latency_ms: float = 0,
) -> dict | None:
    """Process a single provider's latency data."""
    latency_path = submission_dir / "latency.json"
    if not latency_path.exists():
        print(f"  No latency.json found in {submission_dir}, skipping")
        return None

    with open(latency_path, "r", encoding="utf-8") as f:
        latency = json.load(f)

    stats = compute_latency_stats(
        latency,
        manifest,
        drop_first=drop_first,
        min_latency_ms=min_latency_ms,
    )
    provider_name = submission_dir.name
    filt = stats.get("filtered", {})
    total_kept = filt.get("total_kept", 0)
    d_warmup = filt.get("dropped_warmup", 0)
    d_min = filt.get("dropped_min", 0)
    print(f"\n  {provider_name}: {total_kept} utterances (dropped {d_warmup} warmup, {d_min} below min)")

    for locale, data in sorted(stats.get("locales", {}).items()):
        print(f"    {locale}: p50={data['latencyP50Ms']:.0f}ms  p95={data['latencyP95Ms']:.0f}ms  (n={data['count']})")

    if stats.get("overall"):
        o = stats["overall"]
        print(f"    overall: p50={o['latencyP50Ms']:.0f}ms  p95={o['latencyP95Ms']:.0f}ms  (n={o['count']})")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        scores_path = output_dir / "scores.json"
        if scores_path.exists():
            with open(scores_path, "r", encoding="utf-8") as f:
                scores = json.load(f)
            for locale, data in stats.get("locales", {}).items():
                if locale in scores.get("locales", {}):
                    scores["locales"][locale]["latencyP50Ms"] = data["latencyP50Ms"]
                    scores["locales"][locale]["latencyP95Ms"] = data["latencyP95Ms"]
            if stats.get("overall") and "overall" in scores:
                scores["overall"]["latencyP50Ms"] = stats["overall"]["latencyP50Ms"]
                scores["overall"]["latencyP95Ms"] = stats["overall"]["latencyP95Ms"]
            with open(scores_path, "w", encoding="utf-8") as f:
                json.dump(scores, f, indent=2, ensure_ascii=False)
            print(f"    Updated {scores_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute p50/p95 latency statistics")
    parser.add_argument(
        "--submission-dir",
        type=Path,
        help="Single provider submission directory with latency.json",
    )
    parser.add_argument("--base-dir", type=Path, help="Base directory containing multiple provider dirs")
    parser.add_argument("--manifest", default="manifest.json", type=Path, help="Path to manifest.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output dir for single provider (writes into scores.json)",
    )
    parser.add_argument("--results-dir", type=Path, help="Results base dir for --base-dir mode")
    parser.add_argument(
        "--drop-first",
        type=int,
        default=0,
        help="Drop first N entries per provider (cold-start warmup artifacts)",
    )
    parser.add_argument(
        "--min-latency-ms",
        type=float,
        default=0,
        help="Drop entries below this threshold in ms (likely cache hits)",
    )
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    filter_kwargs = dict(drop_first=args.drop_first, min_latency_ms=args.min_latency_ms)

    if args.submission_dir:
        process_provider(args.submission_dir, manifest, args.output_dir, **filter_kwargs)
    elif args.base_dir:
        if not args.base_dir.exists():
            print(f"Base directory not found: {args.base_dir}")
            return
        for provider_dir in sorted(args.base_dir.iterdir()):
            if not provider_dir.is_dir():
                continue
            out = args.results_dir / provider_dir.name if args.results_dir else None
            process_provider(provider_dir, manifest, out, **filter_kwargs)
    else:
        parser.error("Provide either --submission-dir or --base-dir")


if __name__ == "__main__":
    main()
