"""Validate submission format against the manifest.

Checks directory structure, metadata.yaml, latency.json, and transcript coverage
against manifest.json. Run this locally before opening a PR.

Usage:
    python scoring/validate.py submissions/raw/your-model-name --manifest manifest.json

New in the fairness-fixes release:
  * ``metadata.yaml`` must include a ``config:`` block with the six
    required keys (item 5). Values must be ``default`` or an explicit
    disclosure string; non-default values require an ``override: <key>``
    line in ``notes``.
  * ``latency.json`` may be either the legacy flat map (warns during the
    back-compat window) or the new schema with ``meta.protocol``,
    ``meta.region``, and protocol-specific per-entry fields (item 4).
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

# Required inference-config disclosure (item 5). Keys are fixed; values
# must be ``default`` unless explicitly disclosed with an ``override:`` line
# in ``notes``.
REQUIRED_CONFIG_KEYS = (
    "beamSize",
    "languageHint",
    "customVocabulary",
    "noiseSuppression",
    "domainAdaptation",
    "keywordBoosting",
)

# Allowed regions for ``latency.json meta.region``. Small pinned set so
# cross-provider latency comparisons stay apples-to-apples. Expand by PR
# when a new region is needed; adding a region requires a maintainer
# decision because it changes what counts as a valid submission.
ALLOWED_LATENCY_REGIONS = {
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "eu-west-1",
    "eu-central-1",
    "ap-southeast-1",
    "ap-northeast-1",
}

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


def _validate_config_block(metadata):
    """Check the ``config:`` block. Returns a list of issue strings.

    Rules:
      * ``config`` must be a mapping.
      * Every required key must be present; no extras allowed.
      * Each value must be a non-empty string (either ``default`` or an
        explicit disclosure like ``"10"`` or ``"enabled: 42 terms"``).
      * If any value is not ``default``, ``notes`` must contain a line
        ``override: <key>`` (possibly along with free text) so reviewers
        can see the justification at a glance.
    """
    issues = []
    if "config" not in metadata:
        issues.append(
            "metadata is missing required 'config' block. It must declare the six inference-config "
            f"keys: {', '.join(REQUIRED_CONFIG_KEYS)}. Each value must be 'default' or an explicit "
            "disclosure string; see submissions/SUBMITTING.md."
        )
        return issues
    cfg = metadata.get("config")
    if not isinstance(cfg, dict):
        issues.append("metadata 'config' block must be a YAML mapping")
        return issues

    present_keys = set(cfg.keys())
    required = set(REQUIRED_CONFIG_KEYS)
    missing = required - present_keys
    extra = present_keys - required
    if missing:
        issues.append(f"metadata.config is missing required key(s): {', '.join(sorted(missing))}")
    if extra:
        issues.append(f"metadata.config has unknown key(s): {', '.join(sorted(extra))} (not in the v1 schema)")

    non_default_keys = []
    for key in REQUIRED_CONFIG_KEYS:
        if key not in cfg:
            continue
        val = cfg[key]
        if val is None or (isinstance(val, str) and val.strip() == ""):
            issues.append(f"metadata.config.{key} must be a non-empty string ('default' or an explicit value)")
            continue
        if not isinstance(val, str):
            # YAML might load "10" as int; accept after coercion but require string in file.
            issues.append(
                f"metadata.config.{key} must be a string (got {type(val).__name__}); "
                f'quote the value in metadata.yaml, e.g. "{val}"'
            )
            continue
        if val.strip() != "default":
            non_default_keys.append(key)

    if non_default_keys:
        notes = metadata.get("notes") or ""
        notes_str = notes if isinstance(notes, str) else ""
        notes_lines = [ln.strip() for ln in notes_str.splitlines()]
        missing_overrides = []
        for key in non_default_keys:
            marker = f"override: {key}"
            if not any(ln.startswith(marker) for ln in notes_lines):
                missing_overrides.append(key)
        if missing_overrides:
            issues.append(
                "metadata.config declares non-default value(s) without a matching "
                f"'override: <key>' line in notes: {', '.join(missing_overrides)}. "
                "Add a line to `notes:` like `override: beamSize because ...` for each."
            )

    return issues


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
            if "config:" not in content:
                issues.append(
                    "metadata.yaml may be missing required 'config:' block (install PyYAML for a full schema check)"
                )
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

    issues.extend(_validate_config_block(metadata))
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


def _validate_latency_legacy(data, submitted_by_locale, warnings):
    """Validate the legacy flat latency.json schema.

    Back-compat fence during rollout: accepted with a warning, interpreted
    as ``protocol=batch``, ``region=unknown``. Drops when the rollout PR
    migrates every provider to the new schema.
    """
    issues = []
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

    warnings.append(
        "latency.json is using the legacy flat schema (string keys -> latency ms). "
        "This is accepted for now as protocol=batch / region=unknown. Migrate to the "
        "new schema with a 'meta' block and per-entry 'roundTripMs' / 'ttftMs' / "
        "'completeMs' before the rollout PR lands — see submissions/SUBMITTING.md."
    )
    print(f"  latency.json (legacy): {len(data)} entries ({len(required_keys - missing_keys)} match transcripts)")
    return issues


def _validate_latency_new_schema(data, submitted_by_locale, warnings):
    """Validate the new latency.json schema.

    Top-level shape:
        {"meta": {"protocol": "batch"|"streaming", "region": "us-east-1", ...},
         "measurements": {"<locale>/<uid>": {"roundTripMs": 123.4} | ...}}
    """
    issues = []

    meta = data.get("meta")
    measurements = data.get("measurements")

    if not isinstance(meta, dict):
        issues.append("latency.json 'meta' block is missing or not a mapping")
    if not isinstance(measurements, dict):
        issues.append("latency.json 'measurements' block is missing or not a mapping")
    if issues:
        return issues

    protocol = meta.get("protocol")
    if protocol not in {"batch", "streaming"}:
        issues.append(f"latency.json meta.protocol must be 'batch' or 'streaming'; got {protocol!r}")
    region = meta.get("region")
    if not isinstance(region, str) or not region.strip():
        issues.append("latency.json meta.region is required (non-empty string)")
    elif region not in ALLOWED_LATENCY_REGIONS:
        issues.append(
            f"latency.json meta.region={region!r} is not in the allowed set "
            f"{sorted(ALLOWED_LATENCY_REGIONS)}. Ask a maintainer to expand the list "
            "if you need a new region."
        )

    # Soft meta-fields that we record but don't gate on.
    if "clientLocation" not in meta:
        warnings.append("latency.json meta.clientLocation is recommended (e.g. 'aws:us-east-1')")
    if "concurrency" in meta:
        conc = meta["concurrency"]
        if not isinstance(conc, int) or conc < 1:
            issues.append("latency.json meta.concurrency must be a positive integer")
        elif conc != 1:
            warnings.append(
                f"latency.json meta.concurrency={conc}; cross-provider latency is only comparable at concurrency=1."
            )

    # Validate per-entry fields by protocol.
    bad_keys = []
    missing_fields = []
    bad_values = []
    for key, entry in measurements.items():
        if "/" not in key:
            bad_keys.append(key)
            continue
        if not isinstance(entry, dict):
            bad_values.append(key)
            continue
        if protocol == "batch":
            rt = entry.get("roundTripMs")
            if not isinstance(rt, (int, float)):
                missing_fields.append(f"{key} (missing/invalid roundTripMs)")
        elif protocol == "streaming":
            ttft = entry.get("ttftMs")
            complete = entry.get("completeMs")
            if not isinstance(ttft, (int, float)):
                missing_fields.append(f"{key} (missing/invalid ttftMs)")
            if not isinstance(complete, (int, float)):
                missing_fields.append(f"{key} (missing/invalid completeMs)")

    if bad_keys:
        issues.append(
            f"latency.json measurements has {len(bad_keys)} key(s) without '<locale>/' prefix; "
            f"first few: {', '.join(bad_keys[:5])}"
        )
    if bad_values:
        issues.append(
            f"latency.json measurements has {len(bad_values)} non-object entry value(s); "
            f"first few: {', '.join(bad_values[:5])}"
        )
    if missing_fields:
        shown = missing_fields[:10]
        more = f" (+{len(missing_fields) - len(shown)} more)" if len(missing_fields) > len(shown) else ""
        issues.append(
            f"latency.json measurements has {len(missing_fields)} entry(ies) missing required "
            f"fields for protocol={protocol}:\n" + "\n".join(f"    - {k}" for k in shown) + more
        )

    # Coverage against shipped transcripts.
    required_keys = set()
    for locale, ids in submitted_by_locale.items():
        for uid in ids:
            required_keys.add(f"{locale}/{uid}")
    missing_keys = required_keys - set(measurements.keys())
    if missing_keys:
        shown = sorted(missing_keys)[:10]
        more = f" (+{len(missing_keys) - len(shown)} more)" if len(missing_keys) > len(shown) else ""
        issues.append(
            f"latency.json measurements is missing {len(missing_keys)} entries that have "
            f"transcript files:\n" + "\n".join(f"    - {k}" for k in shown) + more
        )

    extra = set(measurements.keys()) - required_keys
    if extra:
        warnings.append(
            f"latency.json measurements has {len(extra)} key(s) with no matching transcript (will be ignored)"
        )

    print(
        f"  latency.json (new schema): protocol={protocol}, region={region}, "
        f"{len(measurements)} entries ({len(required_keys - missing_keys)} match transcripts)"
    )
    return issues


def _looks_like_new_schema(data) -> bool:
    return isinstance(data, dict) and "meta" in data and "measurements" in data


def validate_latency(submission_dir, submitted_by_locale):
    """Validate latency.json is present, valid, and covers every submitted transcript.

    Accepts either the legacy flat schema (back-compat during rollout; warns)
    or the new schema with ``meta`` + ``measurements`` (item 4).
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
        issues.append(
            "latency.json must be a JSON object: either {'meta': ..., 'measurements': ...} (new) "
            "or a flat map '<locale>/<uid>' -> ms (legacy)"
        )
        return issues, warnings

    if _looks_like_new_schema(data):
        issues.extend(_validate_latency_new_schema(data, submitted_by_locale, warnings))
    else:
        issues.extend(_validate_latency_legacy(data, submitted_by_locale, warnings))

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
