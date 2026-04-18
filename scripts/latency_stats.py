"""Compute p50 and p95 latency statistics from latency.json files.

Requires the new schema (item 4 of the fairness-fixes plan):

    {
      "meta": {"protocol": "batch"|"streaming", "region": "us-east-1", ...},
      "measurements": {
        "en-US/conv-0-turn-0": {"roundTripMs": 269.1},
        "en-US/conv-0-turn-1": {"ttftMs": 190.0, "completeMs": 780.0}
      }
    }

The legacy flat ``{"<locale>/<uid>": ms}`` schema was dropped with the
rollout PR.

Output fields merged into ``scores.json`` under each locale and
``overall``:

  * ``completeP50Ms`` / ``completeP95Ms`` — **unified cross-protocol
    sortable metric** = time to complete transcript. For batch this
    aliases ``roundTripP50/95Ms``; for streaming it is the percentile of
    ``completeMs``. The leaderboard UI sorts on this.
  * ``roundTripP50Ms`` / ``roundTripP95Ms`` — batch-only.
  * ``ttftP50Ms`` / ``ttftP95Ms`` — streaming-only, rendered as a
    ``+TTFT`` annotation.

``scores.json`` also carries a ``latencyMeta`` block with protocol and
region (preserved from the input) so drift checks and the UI can pick
the right badge.

Usage:
    # Single provider
    python scripts/latency_stats.py \
        --submission-dir submissions/raw/deepgram-nova3 \
        --manifest manifest.json \
        --output-dir results/deepgram-nova3

    # All providers under a base directory
    python scripts/latency_stats.py \
        --base-dir submissions/raw \
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


def _percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p50": round(percentile(values, 50), 1),
        "p95": round(percentile(values, 95), 1),
    }


def _parse_latency_input(latency: dict, manifest: dict) -> tuple[str, str, dict[str, dict[str, float]]]:
    """Parse the new-schema latency.json into a common per-entry form.

    Returns ``(protocol, region, per_entry)`` where ``per_entry`` is
    ``{"<locale>/<uid>": {"roundTripMs": ..., "ttftMs": ..., "completeMs": ...}}``.
    For batch, ``completeMs`` aliases ``roundTripMs``.
    """
    locale_map = {utt["id"]: utt["locale"] for utt in manifest["utterances"]}

    if not (isinstance(latency, dict) and "meta" in latency and "measurements" in latency):
        raise ValueError(
            "latency.json must use the new schema {'meta': ..., 'measurements': ...}; "
            "the legacy flat '<locale>/<uid>' -> ms schema was dropped with the rollout PR."
        )

    meta = latency.get("meta", {}) or {}
    protocol = meta.get("protocol", "batch")
    region = meta.get("region", "unknown")
    raw = latency.get("measurements", {}) or {}
    per_entry: dict[str, dict[str, float]] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        if "/" not in key:
            locale = locale_map.get(key)
            if locale is None:
                continue
            key = f"{locale}/{key}"
        rt = entry.get("roundTripMs")
        ttft = entry.get("ttftMs")
        complete = entry.get("completeMs")
        row: dict[str, float] = {}
        if isinstance(rt, (int, float)):
            row["roundTripMs"] = float(rt)
            # For batch the round-trip IS the complete-transcript latency.
            if protocol == "batch":
                row.setdefault("completeMs", float(rt))
        if isinstance(ttft, (int, float)):
            row["ttftMs"] = float(ttft)
        if isinstance(complete, (int, float)):
            row["completeMs"] = float(complete)
        if row:
            per_entry[key] = row
    return protocol, region, per_entry


def compute_latency_stats(
    latency: dict,
    manifest: dict,
    *,
    drop_first: int = 0,
    min_latency_ms: float = 0,
) -> dict:
    """Compute per-locale and overall latency percentiles.

    Args:
        drop_first: Skip the first N entries (cold-start warmup artifacts).
        min_latency_ms: Drop entries with round-trip below this threshold.
    """
    protocol, region, per_entry = _parse_latency_input(latency, manifest)

    items = list(per_entry.items())
    dropped_warmup = min(drop_first, len(items))
    items = items[dropped_warmup:]

    # Filter by min_latency_ms applied to the primary (completeMs) metric.
    dropped_min = 0
    by_locale_rt: dict[str, list[float]] = {}
    by_locale_ttft: dict[str, list[float]] = {}
    by_locale_complete: dict[str, list[float]] = {}
    all_rt: list[float] = []
    all_ttft: list[float] = []
    all_complete: list[float] = []

    for key, row in items:
        locale = key.split("/", 1)[0]
        rt = row.get("roundTripMs")
        ttft = row.get("ttftMs")
        complete = row.get("completeMs")
        if complete is not None and complete < min_latency_ms:
            dropped_min += 1
            continue
        if rt is not None:
            by_locale_rt.setdefault(locale, []).append(rt)
            all_rt.append(rt)
        if ttft is not None:
            by_locale_ttft.setdefault(locale, []).append(ttft)
            all_ttft.append(ttft)
        if complete is not None:
            by_locale_complete.setdefault(locale, []).append(complete)
            all_complete.append(complete)

    def _bucket_for(locale: str, proto: str) -> dict:
        stats: dict = {}
        vals = by_locale_complete.get(locale, [])
        if vals:
            pct = _percentiles(vals)
            stats["completeP50Ms"] = pct["p50"]
            stats["completeP95Ms"] = pct["p95"]
        if proto == "batch" and by_locale_rt.get(locale):
            pct = _percentiles(by_locale_rt[locale])
            stats["roundTripP50Ms"] = pct["p50"]
            stats["roundTripP95Ms"] = pct["p95"]
        if proto == "streaming" and by_locale_ttft.get(locale):
            pct = _percentiles(by_locale_ttft[locale])
            stats["ttftP50Ms"] = pct["p50"]
            stats["ttftP95Ms"] = pct["p95"]
        stats["count"] = len(vals) if vals else len(by_locale_rt.get(locale, []))
        return stats

    locale_keys = sorted(by_locale_complete.keys() | by_locale_rt.keys())
    result: dict = {
        "locales": {locale: _bucket_for(locale, protocol) for locale in locale_keys},
        "protocol": protocol,
        "region": region,
    }

    overall_vals = all_complete or all_rt
    if overall_vals:
        overall: dict = {}
        if all_complete:
            pct = _percentiles(all_complete)
            overall["completeP50Ms"] = pct["p50"]
            overall["completeP95Ms"] = pct["p95"]
        if protocol == "batch" and all_rt:
            pct = _percentiles(all_rt)
            overall["roundTripP50Ms"] = pct["p50"]
            overall["roundTripP95Ms"] = pct["p95"]
        if protocol == "streaming" and all_ttft:
            pct = _percentiles(all_ttft)
            overall["ttftP50Ms"] = pct["p50"]
            overall["ttftP95Ms"] = pct["p95"]
        overall["count"] = len(overall_vals)
        result["overall"] = overall

    result["filtered"] = {
        "dropped_warmup": dropped_warmup,
        "dropped_min": dropped_min,
        "total_kept": len(overall_vals),
    }

    return result


def _merge_into_scores(scores: dict, stats: dict) -> None:
    """Merge latency stats into an existing scores.json dict in place."""
    per_locale_stats = stats.get("locales", {})
    for locale, data in per_locale_stats.items():
        if locale in scores.get("locales", {}):
            for k, v in data.items():
                if k == "count":
                    continue
                scores["locales"][locale][k] = v
    if stats.get("overall") and scores.get("overall"):
        for k, v in stats["overall"].items():
            if k == "count":
                continue
            scores["overall"][k] = v
    scores["latencyMeta"] = {
        "protocol": stats.get("protocol", "batch"),
        "region": stats.get("region", "unknown"),
    }


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
    print(
        f"\n  {provider_name}: protocol={stats.get('protocol')}, region={stats.get('region')}, "
        f"{total_kept} utterances (dropped {d_warmup} warmup, {d_min} below min)"
    )

    for locale, data in sorted(stats.get("locales", {}).items()):
        complete_p50 = data.get("completeP50Ms")
        complete_p95 = data.get("completeP95Ms")
        extra = ""
        if "ttftP95Ms" in data:
            extra = f"  +TTFT p95={data['ttftP95Ms']:.0f}ms"
        if complete_p50 is not None:
            print(
                f"    {locale}: complete p50={complete_p50:.0f}ms  p95={complete_p95:.0f}ms{extra}  (n={data['count']})"
            )

    if stats.get("overall"):
        o = stats["overall"]
        extra = f"  +TTFT p95={o['ttftP95Ms']:.0f}ms" if "ttftP95Ms" in o else ""
        p50 = o.get("completeP50Ms", 0)
        p95 = o.get("completeP95Ms", 0)
        print(f"    overall: complete p50={p50:.0f}ms  p95={p95:.0f}ms{extra}  (n={o['count']})")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        scores_path = output_dir / "scores.json"
        if scores_path.exists():
            with open(scores_path, "r", encoding="utf-8") as f:
                scores = json.load(f)
            _merge_into_scores(scores, stats)
            with open(scores_path, "w", encoding="utf-8") as f:
                json.dump(scores, f, indent=2, ensure_ascii=False)
            print(f"    Updated {scores_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute latency percentiles")
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
