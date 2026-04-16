"""Validate submission format against the manifest.

Checks directory structure, metadata.yaml, latency.json, and transcript coverage
against manifest.json. Run this locally before opening a PR.

Usage:
    python scoring/validate.py submissions/raw/your-model-name --manifest manifest.json
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

REQUIRED_METADATA_FIELDS = ["model", "organization", "version", "date"]

MAX_FILE_SIZE_BYTES = 10_000
MAX_TOTAL_SIZE_MB = 50
ALLOWED_FILES = {".txt"}
ALLOWED_ROOT_FILES = {
    "metadata.yaml",
    "metadata.json",
    "latency.json",
}

DOCS_POINTER = "See submissions/SUBMITTING.md for the full submission contract."


def load_manifest(manifest_path):
    """Load manifest.json and build a locale -> set(utterance_ids) map."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    locale_ids = {}
    for utt in manifest["utterances"]:
        locale = utt["locale"]
        if locale not in locale_ids:
            locale_ids[locale] = set()
        locale_ids[locale].add(utt["id"])

    return manifest["locales"], locale_ids


def validate_metadata(submission_dir):
    """Validate metadata.yaml (or metadata.json) exists and has required fields."""
    issues = []
    metadata_path = submission_dir / "metadata.yaml"
    metadata_json_path = submission_dir / "metadata.json"

    if not metadata_path.exists() and not metadata_json_path.exists():
        issues.append("metadata.yaml (or metadata.json) is missing (required)")
        return issues

    metadata = None

    if metadata_path.exists():
        if yaml is None:
            content = metadata_path.read_text(encoding="utf-8")
            for field in REQUIRED_METADATA_FIELDS:
                if f"{field}:" not in content and f"{field} :" not in content:
                    issues.append(f"metadata.yaml may be missing required field: {field}")
            return issues

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f)

        if not isinstance(metadata, dict):
            issues.append("metadata.yaml is not a valid YAML mapping")
            return issues
    elif metadata_json_path.exists():
        import json as _json

        try:
            with open(metadata_json_path, "r", encoding="utf-8") as f:
                metadata = _json.load(f)
        except Exception as e:
            issues.append(f"metadata.json is not valid JSON: {e}")
            return issues

        if not isinstance(metadata, dict):
            issues.append("metadata.json is not a valid JSON object")
            return issues

    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata or metadata[field] is None or str(metadata[field]).strip() == "":
            issues.append(f"metadata missing required field: {field}")

    return issues


def validate_submission_safety(submission_dir):
    """Check for disallowed files, oversized files, symlinks, and non-UTF-8 content."""
    issues = []
    total_size = 0

    for path in submission_dir.rglob("*"):
        rel = path.relative_to(submission_dir)

        if path.is_symlink():
            issues.append(f"Symlinks are not allowed: {rel}")
            continue

        if not path.is_file():
            continue

        if path.parent == submission_dir:
            if path.name not in ALLOWED_ROOT_FILES:
                issues.append(f"Unexpected file in submission root: {rel}")
            continue

        if path.suffix not in ALLOWED_FILES:
            issues.append(f"Disallowed file type: {rel} (only .txt allowed)")
            continue

        size = path.stat().st_size
        total_size += size

        if size > MAX_FILE_SIZE_BYTES:
            issues.append(f"File too large: {rel} ({size:,} bytes, max {MAX_FILE_SIZE_BYTES:,})")

        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(f"File is not valid UTF-8 text: {rel}")

    total_size_mb = total_size / (1024 * 1024)
    if total_size_mb > MAX_TOTAL_SIZE_MB:
        issues.append(f"Total submission size too large: {total_size_mb:.1f} MB (max {MAX_TOTAL_SIZE_MB} MB)")

    return issues


def validate_latency(submission_dir, submitted_by_locale):
    """Validate latency.json is present, valid, and covers every submitted transcript.

    `submitted_by_locale` is {locale: set(utterance_ids)} — only locales/IDs the
    submitter actually shipped. Partial-locale submissions are allowed, but within
    each submitted locale every transcript must have a matching latency entry.
    """
    issues = []
    warnings = []
    latency_path = submission_dir / "latency.json"

    if not latency_path.exists():
        issues.append("latency.json is missing (required). Include per-utterance API latency in ms.")
        return issues, warnings

    try:
        with open(latency_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        issues.append(f"latency.json is not valid JSON: {e}")
        return issues, warnings

    if not isinstance(data, dict):
        issues.append("latency.json must be a JSON object mapping '<locale>/<utterance_id>' to latency in ms")
        return issues, warnings

    bad_values = []
    bare_keys = []
    for key, val in data.items():
        if not isinstance(val, (int, float)):
            bad_values.append(key)
        if "/" not in key:
            bare_keys.append(key)

    if bad_values:
        issues.append(
            f"latency.json has non-numeric values for {len(bad_values)} key(s); first few: {', '.join(bad_values[:5])}"
        )

    if bare_keys:
        issues.append(
            f"latency.json has {len(bare_keys)} key(s) without '<locale>/' prefix; keys must be "
            f"'<locale>/<utterance_id>' (e.g. 'en-US/conv-0-turn-0'). First few: "
            f"{', '.join(bare_keys[:5])}"
        )

    required_keys = set()
    for locale, ids in submitted_by_locale.items():
        for uid in ids:
            required_keys.add(f"{locale}/{uid}")

    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        shown = sorted(missing_keys)[:10]
        more = f" (+{len(missing_keys) - len(shown)} more)" if len(missing_keys) > len(shown) else ""
        issues.append(
            f"latency.json is missing {len(missing_keys)} entries that have transcript files:\n"
            + "\n".join(f"    - {k}" for k in shown)
            + more
        )

    extra = set(data.keys()) - required_keys
    if extra:
        warnings.append(f"latency.json has {len(extra)} key(s) with no matching transcript (will be ignored)")

    print(f"  latency.json: {len(data)} entries ({len(required_keys - missing_keys)} match shipped transcripts)")
    return issues, warnings


def validate_files(submission_dir, locale_ids):
    """Validate transcript files against the manifest.

    Returns (issues, warnings, submitted_by_locale) where submitted_by_locale
    is a {locale: set(utterance_ids)} map of what was actually shipped — partial
    submissions (subset of locales) are allowed.
    """
    issues = []
    warnings = []
    submitted_by_locale = {}
    total_matched = 0
    total_expected = 0

    locale_dirs = [d for d in submission_dir.iterdir() if d.is_dir() and d.name in locale_ids]

    if not locale_dirs:
        issues.append("No locale directories found matching manifest locales (en-US, es-MX, tr-TR, vi-VN, zh-CN)")
        return issues, warnings, submitted_by_locale

    for locale_dir in sorted(locale_dirs, key=lambda d: d.name):
        locale = locale_dir.name
        expected_ids = locale_ids[locale]

        submission_ids = {f.stem for f in locale_dir.glob("*.txt")}
        matched = expected_ids & submission_ids
        missing = expected_ids - submission_ids
        extra = submission_ids - expected_ids
        submitted_by_locale[locale] = matched

        empty = [f.stem for f in locale_dir.glob("*.txt") if f.stat().st_size == 0]

        total_expected += len(expected_ids)
        total_matched += len(matched)

        status = f"  {locale}: {len(matched)}/{len(expected_ids)} utterances"
        if missing:
            status += f" ({len(missing)} missing)"
        if extra:
            status += f" ({len(extra)} extra)"
        if empty:
            status += f" ({len(empty)} empty)"
        print(status)

        if extra:
            warnings.append(
                f"{locale}: {len(extra)} extra files not in manifest (will be ignored): "
                f"{', '.join(sorted(list(extra)[:5]))}"
                f"{'...' if len(extra) > 5 else ''}"
            )

        if missing:
            missing_list = sorted(missing)
            issues.append(
                f"{locale}: {len(missing)}/{len(expected_ids)} utterances missing. Within each included locale "
                f"you must ship a .txt file for every utterance in the manifest:\n"
                + "\n".join(f"    - {m}" for m in missing_list[:10])
                + (f" (+{len(missing) - 10} more)" if len(missing) > 10 else "")
            )

        if empty:
            warnings.append(
                f"{locale}: {len(empty)} empty transcript file(s) — fine if the utterance is silent/inaudible"
            )

    submitted_locales = set(submitted_by_locale.keys())
    skipped = set(locale_ids) - submitted_locales
    if skipped:
        warnings.append(
            f"Partial submission: {len(skipped)} locale(s) skipped ({', '.join(sorted(skipped))}). "
            "Only submitted locales will be scored."
        )

    unknown_dirs = [
        d.name
        for d in submission_dir.iterdir()
        if d.is_dir() and d.name not in locale_ids and d.name not in (".", "..")
    ]
    if unknown_dirs:
        warnings.append(f"Unknown locale directories (ignored): {', '.join(unknown_dirs)}")

    print(f"\n  Total: {total_matched}/{total_expected} utterances matched across submitted locales")

    return issues, warnings, submitted_by_locale


def main():
    parser = argparse.ArgumentParser(description="Validate a submission directory")
    parser.add_argument("submission_dir", type=Path, help="Path to submission directory")
    parser.add_argument("--manifest", default="manifest.json", type=Path, help="Path to manifest.json")
    args = parser.parse_args()

    submission_dir = args.submission_dir.resolve()
    manifest_path = args.manifest.resolve()

    if not submission_dir.exists():
        print(f"ERROR: submission directory not found: {submission_dir}")
        sys.exit(1)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"Validating: {submission_dir}\n")

    print("Running safety checks...")
    safety_issues = validate_submission_safety(submission_dir)
    if safety_issues:
        print(f"\nSAFETY ISSUES ({len(safety_issues)}):")
        for issue in safety_issues:
            print(f"  - {issue}")
        print(f"\n{DOCS_POINTER}")
        print("Validation FAILED (safety).")
        sys.exit(1)
    print("Safety checks passed.\n")

    _, locale_ids = load_manifest(manifest_path)

    metadata_issues = validate_metadata(submission_dir)

    file_issues, file_warnings, submitted_by_locale = validate_files(submission_dir, locale_ids)

    latency_issues, latency_warnings = validate_latency(submission_dir, submitted_by_locale)
    file_warnings.extend(latency_warnings)

    all_issues = metadata_issues + file_issues + latency_issues

    if file_warnings:
        print(f"\nWARNINGS ({len(file_warnings)}):")
        for w in file_warnings:
            print(f"  - {w}")

    if all_issues:
        print(f"\nISSUES ({len(all_issues)}):")
        for issue in all_issues:
            print(f"  - {issue}")
        print(f"\n{DOCS_POINTER}")
        print("Validation FAILED.")
        sys.exit(1)
    else:
        print("\nValidation passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
