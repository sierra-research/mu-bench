"""Validate submission format against the manifest.

Checks directory structure, metadata.yaml, and file coverage against manifest.json.

Usage:
    python scoring/validate.py submissions/raw/deepgram-nova3 --manifest manifest.json
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

MAX_FILE_SIZE_BYTES = 10_000  # 10 KB — no transcript should be longer than this
MAX_TOTAL_SIZE_MB = 50  # 50 MB aggregate cap for the entire submission
ALLOWED_FILES = {".txt"}  # Only .txt files allowed in locale dirs
ALLOWED_ROOT_FILES = {
    "metadata.yaml",
    "metadata.json",
    "latency.json",
}  # Allowed in submission root


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
    """Validate metadata.yaml exists and has required fields."""
    issues = []
    metadata_path = submission_dir / "metadata.yaml"

    if not metadata_path.exists():
        issues.append("metadata.yaml is missing")
        return issues

    if yaml is None:
        # Fall back to basic check if pyyaml not installed
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

    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata or metadata[field] is None or str(metadata[field]).strip() == "":
            issues.append(f"metadata.yaml missing required field: {field}")

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

        # Root-level files: only metadata.yaml/json allowed
        if path.parent == submission_dir:
            if path.name not in ALLOWED_ROOT_FILES:
                issues.append(f"Unexpected file in submission root: {rel}")
            continue

        # Inside locale dirs: only .txt files allowed
        if path.suffix not in ALLOWED_FILES:
            issues.append(f"Disallowed file type: {rel} (only .txt allowed)")
            continue

        size = path.stat().st_size
        total_size += size

        if size > MAX_FILE_SIZE_BYTES:
            issues.append(f"File too large: {rel} ({size:,} bytes, max {MAX_FILE_SIZE_BYTES:,})")

        # Verify file is valid UTF-8 text
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(f"File is not valid UTF-8 text: {rel}")

    total_size_mb = total_size / (1024 * 1024)
    if total_size_mb > MAX_TOTAL_SIZE_MB:
        issues.append(f"Total submission size too large: {total_size_mb:.1f} MB (max {MAX_TOTAL_SIZE_MB} MB)")

    return issues


def validate_latency(submission_dir, locale_ids):
    """Validate latency.json if present (optional file)."""
    warnings = []
    latency_path = submission_dir / "latency.json"
    if not latency_path.exists():
        return [], warnings

    try:
        with open(latency_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return [f"latency.json is not valid JSON: {e}"], warnings

    if not isinstance(data, dict):
        return ["latency.json must be a JSON object mapping utterance IDs to latency values"], warnings

    all_ids = set()
    for ids in locale_ids.values():
        all_ids |= ids

    bad_values = []
    for uid, val in data.items():
        if not isinstance(val, (int, float)):
            bad_values.append(uid)

    if bad_values:
        return [f"latency.json has non-numeric values for: {', '.join(bad_values[:5])}"], warnings

    # Keys can be "locale/utterance_id" or bare "utterance_id"
    all_keyed_ids = set()
    for locale, ids in locale_ids.items():
        for uid in ids:
            all_keyed_ids.add(f"{locale}/{uid}")
            all_keyed_ids.add(uid)

    matched = sum(1 for k in data if k in all_keyed_ids)
    extra_count = len(data) - matched
    if extra_count:
        warnings.append(f"latency.json has {extra_count} keys not matching manifest (will be ignored)")

    print(f"  latency.json: {len(data)} entries ({matched} match manifest)")
    return [], warnings


def validate_files(submission_dir, manifest_locales, locale_ids):
    """Validate transcript files against the manifest."""
    issues = []
    warnings = []
    total_matched = 0
    total_expected = 0

    # Find locale dirs in submission
    locale_dirs = [d for d in submission_dir.iterdir() if d.is_dir() and d.name in locale_ids]

    if not locale_dirs:
        issues.append("No locale directories found matching manifest locales")
        return issues, warnings

    for locale_dir in sorted(locale_dirs, key=lambda d: d.name):
        locale = locale_dir.name
        expected_ids = locale_ids[locale]
        total_expected += len(expected_ids)

        submission_ids = {f.stem for f in locale_dir.glob("*.txt")}
        matched = expected_ids & submission_ids
        missing = expected_ids - submission_ids
        extra = submission_ids - expected_ids
        total_matched += len(matched)

        # Check for empty files
        empty = [f.stem for f in locale_dir.glob("*.txt") if f.stat().st_size == 0]

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
            warnings.append(
                f"{locale}: {len(missing)}/{len(expected_ids)} utterances missing "
                f"(these will be scored as full errors):\n" + "\n".join(f"    - {m}" for m in missing_list)
            )

        if empty:
            warnings.append(f"{locale}: {len(empty)} empty transcript files")

    # Check for unexpected locale dirs
    unknown_dirs = [
        d.name
        for d in submission_dir.iterdir()
        if d.is_dir() and d.name not in locale_ids and d.name not in (".", "..")
    ]
    if unknown_dirs:
        warnings.append(f"Unknown locale directories: {', '.join(unknown_dirs)}")

    print(f"\n  Total: {total_matched}/{total_expected} utterances matched")

    return issues, warnings


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

    # Safety checks first (file types, sizes, symlinks, encoding)
    print("Running safety checks...")
    safety_issues = validate_submission_safety(submission_dir)
    if safety_issues:
        print(f"\nSAFETY ISSUES ({len(safety_issues)}):")
        for issue in safety_issues:
            print(f"  - {issue}")
        print("\nValidation FAILED (safety).")
        sys.exit(1)
    print("Safety checks passed.\n")

    manifest_locales, locale_ids = load_manifest(manifest_path)

    # Validate metadata
    metadata_issues = validate_metadata(submission_dir)

    # Validate files
    file_issues, file_warnings = validate_files(submission_dir, manifest_locales, locale_ids)

    # Validate latency (optional)
    latency_issues, latency_warnings = validate_latency(submission_dir, locale_ids)
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
        print("\nValidation FAILED.")
        sys.exit(1)
    else:
        print("\nValidation passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
