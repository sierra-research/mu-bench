"""Per-file cache invalidation for `results/<name>/details/`.

Companion to `invalidate_normalized_cache.py`. `scoring.score` already
skips per-utterance metrics whose cached `details/<locale>/<uid>.json`
entry has `wer` / `significantWer` set, so we just have to prune only
the detail files whose normalized prediction (or gold) changed.

Subcommands:
  invalidate  If judge config or manifest gold hash changed, wipe. Else
              delete `details/<locale>/<uid>.json` for any utterance
              whose normalized prediction file hash changed since the
              last cache write.
  update      After `scoring.score` succeeds, record the current
              normalized prediction hashes and invariants so the next
              run can diff against them.

Cache metadata lives at `<results-dir>/.cache_meta.json` and is cached
alongside `details/` and `scores.json`.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

META_FILENAME = ".cache_meta.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _enumerate_normalized(normalized_dir: Path):
    for locale_dir in sorted(normalized_dir.iterdir()):
        if not locale_dir.is_dir() or locale_dir.name.startswith("_"):
            continue
        for txt in sorted(locale_dir.glob("*.txt")):
            yield locale_dir.name, txt.stem, _sha256_file(txt)


def _current_files(normalized_dir: Path) -> dict:
    return {f"{locale}/{uid}": h for locale, uid, h in _enumerate_normalized(normalized_dir)}


def _wipe(results_dir: Path) -> None:
    if not results_dir.exists():
        return
    details_dir = results_dir / "details"
    if details_dir.exists():
        for p in details_dir.rglob("*.json"):
            p.unlink()
    for name in ("scores.json", META_FILENAME):
        p = results_dir / name
        if p.exists():
            p.unlink()


def cmd_invalidate(args) -> None:
    meta_path = args.results_dir / META_FILENAME
    if not args.results_dir.exists() or not meta_path.exists():
        print("No prior results cache; full scoring will run.")
        return

    try:
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Results cache metadata invalid ({e}); wiping.")
        _wipe(args.results_dir)
        return

    current_judge = {
        "model": args.judge_model,
        "temperature": args.judge_temperature,
        "seed": args.judge_seed,
    }
    if cached.get("judge") != current_judge:
        print("Judge config changed; wiping results cache.")
        _wipe(args.results_dir)
        return
    if cached.get("manifestGoldHash") != args.manifest_gold_hash:
        print("Manifest gold hash changed; wiping results cache.")
        _wipe(args.results_dir)
        return

    current = _current_files(args.normalized_dir)
    cached_files = cached.get("normalizedFiles", {})
    details_dir = args.results_dir / "details"

    changed = 0
    removed = 0
    for key, current_hash in current.items():
        if cached_files.get(key) != current_hash:
            detail_path = details_dir / f"{key}.json"
            if detail_path.exists():
                detail_path.unlink()
                changed += 1
    for key in cached_files:
        if key not in current:
            detail_path = details_dir / f"{key}.json"
            if detail_path.exists():
                detail_path.unlink()
                removed += 1

    # scores.json is rebuilt from details each run, but if ANY detail was
    # invalidated we blow it away to avoid a stale aggregate hanging around
    # if score.py short-circuits on error.
    if changed or removed:
        scores_path = args.results_dir / "scores.json"
        if scores_path.exists():
            scores_path.unlink()

    retained = sum(1 for key in current if (details_dir / f"{key}.json").exists())
    print(
        f"Results cache hit. {changed} detail(s) invalidated (normalized changed), "
        f"{removed} stale entries removed, {retained} retained."
    )


def cmd_update(args) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "judge": {
            "model": args.judge_model,
            "temperature": args.judge_temperature,
            "seed": args.judge_seed,
        },
        "manifestGoldHash": args.manifest_gold_hash,
        "normalizedFiles": _current_files(args.normalized_dir),
    }
    meta_path = args.results_dir / META_FILENAME
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {meta_path} ({len(meta['normalizedFiles'])} normalized files hashed).")


def _add_common(p):
    p.add_argument("--normalized-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-temperature", required=True)
    p.add_argument("--judge-seed", required=True)
    p.add_argument("--manifest-gold-hash-file", type=Path, required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_common(sub.add_parser("invalidate"))
    _add_common(sub.add_parser("update"))
    args = parser.parse_args()

    if not args.manifest_gold_hash_file.exists():
        print(f"ERROR: manifest gold hash file missing: {args.manifest_gold_hash_file}")
        sys.exit(1)
    args.manifest_gold_hash = args.manifest_gold_hash_file.read_text(encoding="utf-8").strip()
    args.normalized_dir = args.normalized_dir.resolve()
    args.results_dir = args.results_dir.resolve()

    if args.cmd == "invalidate":
        cmd_invalidate(args)
    elif args.cmd == "update":
        cmd_update(args)


if __name__ == "__main__":
    main()
