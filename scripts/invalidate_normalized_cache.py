"""Per-file cache invalidation for `submissions/normalized/<name>/`.

Runs between restoring the GitHub Actions cache and invoking
`scoring.normalize`. `normalize.py` already skips any (locale, uid) whose
output file already exists, so if we delete only the stale files here,
normalize only regenerates what actually changed.

Subcommands:
  invalidate  Delete normalized .txt files whose raw transcript content
              changed since the last cache write. If the judge config or
              the manifest gold hash changed, wipe the whole cache.
  update      After normalize succeeds, record the current content hashes
              and invariants so the next run can diff against them.

The metadata file lives at `<normalized-dir>/.cache_meta.json` and is
cached alongside the normalized output.
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


def _enumerate_raw(submission_dir: Path):
    for locale_dir in sorted(submission_dir.iterdir()):
        if not locale_dir.is_dir():
            continue
        for txt in sorted(locale_dir.glob("*.txt")):
            yield locale_dir.name, txt.stem, _sha256_file(txt)


def _current_files(submission_dir: Path) -> dict:
    return {f"{locale}/{uid}": h for locale, uid, h in _enumerate_raw(submission_dir)}


def _wipe(normalized_dir: Path) -> None:
    if not normalized_dir.exists():
        return
    for p in normalized_dir.rglob("*.txt"):
        p.unlink()
    meta_path = normalized_dir / META_FILENAME
    if meta_path.exists():
        meta_path.unlink()


def cmd_invalidate(args) -> None:
    meta_path = args.normalized_dir / META_FILENAME
    if not args.normalized_dir.exists() or not meta_path.exists():
        print("No prior normalized cache; full normalize will run.")
        return

    try:
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Cache metadata invalid ({e}); wiping.")
        _wipe(args.normalized_dir)
        return

    current_judge = {
        "model": args.judge_model,
        "temperature": args.judge_temperature,
        "seed": args.judge_seed,
    }
    if cached.get("judge") != current_judge:
        print("Judge config changed; wiping normalized cache.")
        _wipe(args.normalized_dir)
        return
    if cached.get("manifestGoldHash") != args.manifest_gold_hash:
        print("Manifest gold hash changed; wiping normalized cache.")
        _wipe(args.normalized_dir)
        return

    current = _current_files(args.submission_dir)
    cached_files = cached.get("files", {})

    changed = 0
    removed = 0
    for key, current_hash in current.items():
        if cached_files.get(key) != current_hash:
            norm_path = args.normalized_dir / f"{key}.txt"
            if norm_path.exists():
                norm_path.unlink()
                changed += 1
    for key in cached_files:
        if key not in current:
            norm_path = args.normalized_dir / f"{key}.txt"
            if norm_path.exists():
                norm_path.unlink()
                removed += 1

    retained = sum(1 for key in current if (args.normalized_dir / f"{key}.txt").exists())
    print(
        f"Cache hit. {changed} file(s) invalidated (content changed), "
        f"{removed} stale entries removed, {retained} retained."
    )


def cmd_update(args) -> None:
    args.normalized_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "judge": {
            "model": args.judge_model,
            "temperature": args.judge_temperature,
            "seed": args.judge_seed,
        },
        "manifestGoldHash": args.manifest_gold_hash,
        "files": _current_files(args.submission_dir),
    }
    meta_path = args.normalized_dir / META_FILENAME
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {meta_path} ({len(meta['files'])} files hashed).")


def _add_common(p):
    p.add_argument("--submission-dir", type=Path, required=True)
    p.add_argument("--normalized-dir", type=Path, required=True)
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
    args.submission_dir = args.submission_dir.resolve()
    args.normalized_dir = args.normalized_dir.resolve()

    if args.cmd == "invalidate":
        cmd_invalidate(args)
    elif args.cmd == "update":
        cmd_update(args)


if __name__ == "__main__":
    main()
