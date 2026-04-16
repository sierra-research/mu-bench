#!/usr/bin/env python3
"""
Three-way data integrity verification: manifest.json vs HuggingFace vs local audio.

Supports both online mode (downloads from HF) and offline mode (uses a pre-downloaded
HF dataset directory via --hf-dir).

Usage:
    # Online (downloads from HF):
    HF_TOKEN=... python scripts/verify_data.py --manifest manifest.json --audio-dir audio

    # Offline (pre-downloaded HF dataset):
    python scripts/verify_data.py --manifest manifest.json --audio-dir audio --hf-dir /path/to/hf-dataset
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

REPO_ID = "sierra-research/mu-bench"
DURATION_TOLERANCE = 0.001  # seconds
EXPECTED_SAMPLERATE = 8000
EXPECTED_CHANNELS = 1

METADATA_REQUIRED_FIELDS = ("file_name", "locale", "conversation_id", "turn_index", "duration_sec", "transcript")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def audio_info(path: Path) -> sf.SoundFile:
    return sf.info(str(path))


def audio_duration(path: Path) -> float:
    info = audio_info(path)
    return info.frames / info.samplerate


class Checker:
    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = ""):
        self.results.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}", flush=True)
        if not passed and detail:
            for line in detail.strip().split("\n"):
                print(f"         {line}")
            sys.stdout.flush()

    @property
    def all_passed(self) -> bool:
        return all(ok for _, ok, _ in self.results)

    def summary(self):
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        failed = total - passed
        print(f"\n{'=' * 60}")
        print(f"Summary: {passed}/{total} checks passed, {failed} failed")
        if failed:
            print("\nFailed checks:")
            for name, ok, detail in self.results:
                if not ok:
                    print(f"  - {name}")
                    if detail:
                        for line in detail.strip().split("\n")[:10]:
                            print(f"    {line}")
        print(f"{'=' * 60}")


def load_manifest(manifest_path: Path) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def phase0_manifest_consistency(manifest: dict, checker: Checker):
    print("\n--- Phase 0: Manifest internal consistency ---")

    utts = manifest["utterances"]
    locales_declared = set(manifest["locales"])

    locale_id_pairs = [(u["locale"], u["id"]) for u in utts]
    pair_set = set(locale_id_pairs)
    checker.check(
        "No duplicate utterance IDs within each locale",
        len(locale_id_pairs) == len(pair_set),
        f"{len(locale_id_pairs) - len(pair_set)} duplicates" if len(locale_id_pairs) != len(pair_set) else "",
    )

    paths = [u["audio_path"] for u in utts]
    path_set = set(paths)
    checker.check(
        "No duplicate audio paths",
        len(paths) == len(path_set),
        f"{len(paths) - len(path_set)} duplicates" if len(paths) != len(path_set) else "",
    )

    bad_paths = []
    for u in utts:
        expected = f"audio/{u['locale']}/{u['id']}.wav"
        if u["audio_path"] != expected:
            bad_paths.append(f"{u['id']}: got {u['audio_path']}, expected {expected}")
    checker.check(
        "All audio_path values follow pattern audio/<locale>/<id>.wav",
        len(bad_paths) == 0,
        "\n".join(bad_paths[:10]) + (f"\n... and {len(bad_paths) - 10} more" if len(bad_paths) > 10 else ""),
    )

    bad_durations = [
        u["id"] for u in utts if not isinstance(u.get("duration_sec"), (int, float)) or u["duration_sec"] <= 0
    ]
    checker.check(
        "All duration_sec values are positive numbers",
        len(bad_durations) == 0,
        f"Bad durations: {bad_durations[:10]}" if bad_durations else "",
    )

    locales_used = {u["locale"] for u in utts}
    missing_locales = locales_used - locales_declared
    checker.check(
        "All utterance locales are declared in manifest locales array",
        len(missing_locales) == 0,
        f"Undeclared locales: {missing_locales}" if missing_locales else "",
    )

    convs: dict[str, list[int]] = defaultdict(list)
    for u in utts:
        convs[f"{u['locale']}/{u['conversation_id']}"].append(u["turn_index"])
    bad_seqs = []
    for conv_key, turns in convs.items():
        turns_sorted = sorted(turns)
        expected = list(range(len(turns_sorted)))
        if turns_sorted != expected:
            bad_seqs.append(f"{conv_key}: got {turns_sorted}, expected {expected}")
    checker.check(
        "Turn indices are sequential (0, 1, 2, ...) per conversation",
        len(bad_seqs) == 0,
        "\n".join(bad_seqs[:10]) + (f"\n... and {len(bad_seqs) - 10} more" if len(bad_seqs) > 10 else ""),
    )


def phase1_list_hf_online(token: str) -> tuple[set[str], list[str]]:
    """List HF files via API, return (wav_paths, all_files)."""
    from huggingface_hub import HfApi

    api = HfApi()
    all_files = list(api.list_repo_files(REPO_ID, repo_type="dataset", token=token))
    wav_files = {f for f in all_files if f.endswith(".wav")}
    non_wav = sorted(set(all_files) - wav_files)

    print(f"  HF total files: {len(all_files)} ({len(wav_files)} wav, {len(non_wav)} non-wav)")
    if non_wav:
        print(f"  Non-wav files on HF: {non_wav}")

    return wav_files, all_files


def phase1_list_hf_offline(hf_dir: Path) -> tuple[set[str], list[str]]:
    """Walk a pre-downloaded HF dataset directory, return (wav_paths, all_files)."""
    all_files = []
    wav_files = set()

    for path in sorted(hf_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(hf_dir)
        parts = rel.parts
        if parts[0].startswith("."):
            continue
        rel_str = str(rel)
        all_files.append(rel_str)
        if rel_str.endswith(".wav"):
            wav_files.add(rel_str)

    non_wav = sorted(set(all_files) - wav_files)
    print(f"  HF dir total files: {len(all_files)} ({len(wav_files)} wav, {len(non_wav)} non-wav)")
    if non_wav:
        print(f"  Non-wav files: {non_wav}")

    return wav_files, all_files


def phase1_list_hf(token: str | None, hf_dir: Path | None, checker: Checker) -> tuple[set[str], list[str]]:
    print("\n--- Phase 1: List HuggingFace files ---")
    if hf_dir:
        return phase1_list_hf_offline(hf_dir)
    else:
        return phase1_list_hf_online(token)


def _read_metadata_jsonl(meta_path: Path) -> tuple[bytes, list[str], list[dict]]:
    """Read metadata.jsonl, return (raw_bytes, raw_lines, parsed_rows)."""
    raw_bytes = meta_path.read_bytes()
    raw_text = raw_bytes.decode("utf-8")
    raw_lines = [line for line in raw_text.split("\n") if line.strip()]
    rows = [json.loads(line) for line in raw_lines]
    return raw_bytes, raw_lines, rows


def phase1b_compare_hf_metadata(
    manifest: dict,
    hf_all_files: list[str],
    hf_dir: Path | None,
    token: str | None,
    temp_dir: Path,
    checker: Checker,
) -> dict[str, dict]:
    """Compare HF metadata against local manifest. Returns dict of HF utterances keyed by locale/id."""
    print("\n--- Phase 1b: HF metadata comparison ---")

    hf_utts: dict[str, dict] = {}
    meta_path: Path | None = None

    if "manifest.json" in hf_all_files:
        print("  Found manifest.json on HF — comparing")
        if hf_dir:
            hf_manifest_path = hf_dir / "manifest.json"
        else:
            from huggingface_hub import hf_hub_download

            hf_manifest_path = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename="manifest.json",
                    repo_type="dataset",
                    local_dir=str(temp_dir),
                    token=token,
                )
            )
        with open(hf_manifest_path, "r", encoding="utf-8") as f:
            hf_manifest = json.load(f)

        checker.check(
            "HF manifest locales match local manifest",
            set(hf_manifest.get("locales", [])) == set(manifest.get("locales", [])),
            f"HF: {hf_manifest.get('locales')}, Local: {manifest.get('locales')}",
        )
        for u in hf_manifest.get("utterances", []):
            key = f"{u.get('locale', '')}/{u.get('id', '')}"
            hf_utts[key] = u
        return hf_utts

    if "metadata.jsonl" in hf_all_files:
        print("  Found metadata.jsonl on HF — comparing")
        if hf_dir:
            meta_path = hf_dir / "metadata.jsonl"
        else:
            from huggingface_hub import hf_hub_download

            meta_path = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename="metadata.jsonl",
                    repo_type="dataset",
                    local_dir=str(temp_dir),
                    token=token,
                )
            )
    else:
        print("  No manifest.json or metadata.jsonl found on HF — skipping metadata comparison")
        return hf_utts

    _, raw_lines, rows = _read_metadata_jsonl(meta_path)

    # --- Check: line count matches manifest utterance count ---
    expected_count = len(manifest["utterances"])
    checker.check(
        f"metadata.jsonl line count ({len(raw_lines)}) == manifest utterance count ({expected_count})",
        len(raw_lines) == expected_count,
        f"metadata.jsonl has {len(raw_lines)} non-blank lines, manifest has {expected_count} utterances",
    )

    # --- Check: no duplicate file_name entries ---
    file_names = [row.get("file_name", "") for row in rows]
    file_name_counts = Counter(file_names)
    duplicates = {fn: cnt for fn, cnt in file_name_counts.items() if cnt > 1}
    checker.check(
        "No duplicate file_name entries in metadata.jsonl",
        len(duplicates) == 0,
        "\n".join(f"{fn}: appears {cnt} times" for fn, cnt in sorted(duplicates.items())[:10]) if duplicates else "",
    )

    # --- Check: all required fields present on every row ---
    rows_missing_fields = []
    for i, row in enumerate(rows):
        missing = [f for f in METADATA_REQUIRED_FIELDS if f not in row]
        if missing:
            rows_missing_fields.append(f"line {i + 1}: missing {missing}")
    checker.check(
        f"All {len(METADATA_REQUIRED_FIELDS)} required fields present on every metadata.jsonl row",
        len(rows_missing_fields) == 0,
        "\n".join(rows_missing_fields[:10])
        + (f"\n... and {len(rows_missing_fields) - 10} more" if len(rows_missing_fields) > 10 else "")
        if rows_missing_fields
        else "",
    )

    # --- Check: file_name follows <locale>/<id>.wav pattern ---
    bad_file_names = []
    for i, row in enumerate(rows):
        fn = row.get("file_name", "")
        locale = row.get("locale", "")
        if not re.match(r"^[a-z]{2}-[A-Z]{2}/conv-\d+-turn-\d+\.wav$", fn):
            bad_file_names.append(f"line {i + 1}: {fn!r} doesn't match expected pattern")
        elif not fn.startswith(f"{locale}/"):
            bad_file_names.append(f"line {i + 1}: {fn!r} doesn't start with locale {locale!r}")
    checker.check(
        "All file_name values match <locale>/<id>.wav pattern",
        len(bad_file_names) == 0,
        "\n".join(bad_file_names[:10]) if bad_file_names else "",
    )

    # --- Build hf_utts dict, deriving id from file_name ---
    for row in rows:
        file_name = row.get("file_name", "")
        stem = file_name.removesuffix(".wav")
        parts = stem.split("/", 1)
        if len(parts) == 2:
            _, utt_id = parts
            row["_derived_id"] = utt_id
        else:
            row["_derived_id"] = file_name
        key = f"{row.get('locale', '')}/{row.get('_derived_id', '')}"
        hf_utts[key] = row

    # --- Check: utterance ID sets match ---
    local_utts = {}
    for u in manifest.get("utterances", []):
        key = f"{u['locale']}/{u['id']}"
        local_utts[key] = u

    checker.check(
        "HF metadata utterance count matches local manifest",
        len(hf_utts) == len(local_utts),
        f"HF: {len(hf_utts)}, Local: {len(local_utts)}",
    )

    hf_keys = set(hf_utts.keys())
    local_keys = set(local_utts.keys())
    only_hf = hf_keys - local_keys
    only_local = local_keys - hf_keys
    checker.check(
        "HF metadata utterance IDs match local manifest exactly",
        len(only_hf) == 0 and len(only_local) == 0,
        (f"Only in HF ({len(only_hf)}): {sorted(only_hf)[:10]}\n" if only_hf else "")
        + (f"Only in local ({len(only_local)}): {sorted(only_local)[:10]}" if only_local else ""),
    )

    # --- Field-by-field comparison (no synthetic fields, no silent skips) ---
    common = hf_keys & local_keys
    fields_to_check = ("locale", "conversation_id", "turn_index", "transcript", "duration_sec")
    field_mismatches: dict[str, list[str]] = defaultdict(list)
    for key in sorted(common):
        hf_u = hf_utts[key]
        local_u = local_utts[key]
        for field in fields_to_check:
            hf_val = hf_u.get(field)
            local_val = local_u.get(field)
            if hf_val is None:
                field_mismatches[field].append(f"{key}: field missing from HF metadata")
                continue
            if field == "conversation_id":
                hf_norm = int(re.search(r"\d+", str(hf_val)).group()) if re.search(r"\d+", str(hf_val)) else hf_val
                local_norm = (
                    int(re.search(r"\d+", str(local_val)).group()) if re.search(r"\d+", str(local_val)) else local_val
                )
                if hf_norm != local_norm:
                    field_mismatches[field].append(f"{key}: HF={hf_val!r}, local={local_val!r}")
            elif field == "duration_sec":
                if isinstance(hf_val, (int, float)) and isinstance(local_val, (int, float)):
                    if abs(hf_val - local_val) > DURATION_TOLERANCE:
                        field_mismatches[field].append(f"{key}: HF={hf_val}, local={local_val}")
                elif hf_val != local_val:
                    field_mismatches[field].append(f"{key}: HF={hf_val!r}, local={local_val!r}")
            elif hf_val != local_val:
                field_mismatches[field].append(f"{key}: HF={hf_val!r}, local={local_val!r}")

    for field in fields_to_check:
        mismatches = field_mismatches.get(field, [])
        checker.check(
            f"HF metadata '{field}' matches local for all utterances",
            len(mismatches) == 0,
            "\n".join(mismatches[:5]) + (f"\n... and {len(mismatches) - 5} more" if len(mismatches) > 5 else "")
            if mismatches
            else "",
        )

    return hf_utts


def phase1c_regeneration_check(
    manifest: dict, hf_all_files: list[str], hf_dir: Path | None, token: str | None, temp_dir: Path, checker: Checker
):
    """Regenerate metadata.jsonl from manifest and byte-compare with HF copy."""
    print("\n--- Phase 1c: metadata.jsonl regeneration check ---")

    if "metadata.jsonl" not in hf_all_files:
        print("  No metadata.jsonl on HF — skipping regeneration check")
        return

    if hf_dir:
        meta_path = hf_dir / "metadata.jsonl"
    else:
        meta_path = temp_dir / "metadata.jsonl"

    if not meta_path.exists():
        checker.check("metadata.jsonl exists for regeneration check", False, f"Not found at {meta_path}")
        return

    hf_raw = meta_path.read_bytes()

    buf = io.StringIO()
    for u in manifest["utterances"]:
        conv_match = re.search(r"(\d+)", u["conversation_id"])
        conv_id = int(conv_match.group(1)) if conv_match else u["conversation_id"]
        row = {
            "file_name": u["audio_path"].removeprefix("audio/"),
            "locale": u["locale"],
            "conversation_id": conv_id,
            "turn_index": u["turn_index"],
            "duration_sec": u["duration_sec"],
            "transcript": u["transcript"],
        }
        buf.write(json.dumps(row, ensure_ascii=False) + "\n")

    regenerated = buf.getvalue().encode("utf-8")

    if hf_raw == regenerated:
        checker.check("Regenerated metadata.jsonl is byte-identical to HF copy", True)
        return

    hf_lines = hf_raw.decode("utf-8").split("\n")
    regen_lines = buf.getvalue().split("\n")

    first_diff_line = None
    for i, (hf_line, regen_line) in enumerate(zip(hf_lines, regen_lines)):
        if hf_line != regen_line:
            first_diff_line = i + 1
            break

    if first_diff_line is None and len(hf_lines) != len(regen_lines):
        detail = f"Line counts differ: HF has {len(hf_lines)} lines, regenerated has {len(regen_lines)}"
    elif first_diff_line:
        detail = (
            f"First difference at line {first_diff_line}:\n"
            f"  HF:    {hf_lines[first_diff_line - 1][:200]}\n"
            f"  Regen: {regen_lines[first_diff_line - 1][:200]}"
        )
    else:
        detail = f"Byte lengths differ: HF={len(hf_raw)}, regen={len(regenerated)}"

    checker.check("Regenerated metadata.jsonl is byte-identical to HF copy", False, detail)


def phase2_download_hf(hf_wav_files: set[str], token: str, temp_dir: Path, checker: Checker) -> list[str]:
    """Download all wav files from HF to temp_dir. Returns list of files that failed to download."""
    from huggingface_hub import hf_hub_download

    print(f"\n--- Phase 2: Download HF audio to {temp_dir} ---")

    failed = []
    total = len(hf_wav_files)
    for i, filepath in enumerate(sorted(hf_wav_files)):
        out_path = temp_dir / filepath
        if out_path.exists():
            if (i + 1) % 200 == 0 or (i + 1) == total:
                print(f"  {i + 1}/{total} (cached/downloaded)")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filepath,
                repo_type="dataset",
                local_dir=str(temp_dir),
                token=token,
            )
        except Exception as e:
            failed.append(f"{filepath}: {e.__class__.__name__}")
            print(f"  FAILED to download {filepath}: {e.__class__.__name__}")
        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total} downloaded")

    checker.check(
        f"All {total} HF files downloaded successfully",
        len(failed) == 0,
        "\n".join(failed[:20]) + (f"\n... and {len(failed) - 20} more" if len(failed) > 20 else "") if failed else "",
    )
    print(f"  Download complete: {total - len(failed)}/{total} files")
    return failed


def _check_audio_format(wav_path: Path) -> list[str]:
    """Check a wav file for expected format (8kHz mono). Returns list of issues."""
    issues = []
    try:
        info = sf.info(str(wav_path))
    except Exception as e:
        return [f"unreadable WAV: {e}"]
    if info.samplerate != EXPECTED_SAMPLERATE:
        issues.append(f"samplerate={info.samplerate} (expected {EXPECTED_SAMPLERATE})")
    if info.channels != EXPECTED_CHANNELS:
        issues.append(f"channels={info.channels} (expected {EXPECTED_CHANNELS})")
    return issues


def phase3_manifest_vs_hf(manifest: dict, hf_wav_files: set[str], temp_dir: Path, checker: Checker):
    print("\n--- Phase 3: Manifest vs HuggingFace ---")

    manifest_hf_paths = set()
    path_to_utt = {}
    for u in manifest["utterances"]:
        hf_path = u["audio_path"].removeprefix("audio/")
        manifest_hf_paths.add(hf_path)
        path_to_utt[hf_path] = u

    in_manifest_not_hf = manifest_hf_paths - hf_wav_files
    checker.check(
        "Every manifest audio path exists in HF",
        len(in_manifest_not_hf) == 0,
        f"{len(in_manifest_not_hf)} missing from HF:\n" + "\n".join(sorted(in_manifest_not_hf)[:10])
        if in_manifest_not_hf
        else "",
    )

    in_hf_not_manifest = hf_wav_files - manifest_hf_paths
    checker.check(
        "Every HF wav has a manifest entry (no extras)",
        len(in_hf_not_manifest) == 0,
        f"{len(in_hf_not_manifest)} extra in HF:\n" + "\n".join(sorted(in_hf_not_manifest)[:10])
        if in_hf_not_manifest
        else "",
    )

    common = manifest_hf_paths & hf_wav_files
    duration_mismatches = []
    format_issues = []
    for i, hf_path in enumerate(sorted(common)):
        utt = path_to_utt[hf_path]
        wav_path = temp_dir / hf_path
        if not wav_path.exists():
            duration_mismatches.append(f"{hf_path}: file not found in HF dir")
            continue
        try:
            info = sf.info(str(wav_path))
        except Exception as e:
            format_issues.append(f"{hf_path}: unreadable WAV: {e}")
            continue

        actual_duration = info.frames / info.samplerate
        expected = utt["duration_sec"]
        if abs(actual_duration - expected) > DURATION_TOLERANCE:
            duration_mismatches.append(
                f"{utt['id']}: manifest={expected:.4f}s, actual={actual_duration:.4f}s, "
                f"diff={abs(actual_duration - expected):.4f}s"
            )

        if info.samplerate != EXPECTED_SAMPLERATE:
            format_issues.append(f"{hf_path}: samplerate={info.samplerate} (expected {EXPECTED_SAMPLERATE})")
        if info.channels != EXPECTED_CHANNELS:
            format_issues.append(f"{hf_path}: channels={info.channels} (expected {EXPECTED_CHANNELS})")

        if (i + 1) % 500 == 0:
            print(f"  Checked {i + 1}/{len(common)} HF audio files...")

    checker.check(
        f"duration_sec matches HF audio for all {len(common)} files (tol={DURATION_TOLERANCE}s)",
        len(duration_mismatches) == 0,
        "\n".join(duration_mismatches[:10])
        + (f"\n... and {len(duration_mismatches) - 10} more" if len(duration_mismatches) > 10 else "")
        if duration_mismatches
        else "",
    )

    checker.check(
        f"All {len(common)} HF audio files are {EXPECTED_SAMPLERATE}Hz mono",
        len(format_issues) == 0,
        "\n".join(format_issues[:10]) + (f"\n... and {len(format_issues) - 10} more" if len(format_issues) > 10 else "")
        if format_issues
        else "",
    )


def phase4_manifest_vs_local(manifest: dict, audio_dir: Path, checker: Checker):
    print("\n--- Phase 4: Manifest vs Local ---")

    manifest_paths = set()
    path_to_utt = {}
    for u in manifest["utterances"]:
        manifest_paths.add(u["audio_path"])
        path_to_utt[u["audio_path"]] = u

    missing_local = []
    for p in sorted(manifest_paths):
        if not (audio_dir.parent / p).exists():
            missing_local.append(p)
    checker.check(
        "Every manifest audio_path exists locally",
        len(missing_local) == 0,
        f"{len(missing_local)} missing locally:\n" + "\n".join(missing_local[:10]) if missing_local else "",
    )

    local_wavs = set()
    for locale_dir in sorted(audio_dir.iterdir()):
        if not locale_dir.is_dir():
            continue
        for wav in locale_dir.glob("*.wav"):
            local_wavs.add(f"audio/{locale_dir.name}/{wav.name}")

    extra_local = local_wavs - manifest_paths
    checker.check(
        "No extra local wav files beyond manifest",
        len(extra_local) == 0,
        f"{len(extra_local)} extra locally:\n" + "\n".join(sorted(extra_local)[:10]) if extra_local else "",
    )

    common = manifest_paths & local_wavs
    duration_mismatches = []
    format_issues = []
    for i, ap in enumerate(sorted(common)):
        utt = path_to_utt[ap]
        wav_path = audio_dir.parent / ap
        try:
            info = sf.info(str(wav_path))
        except Exception as e:
            format_issues.append(f"{ap}: unreadable WAV: {e}")
            continue

        actual_duration = info.frames / info.samplerate
        expected = utt["duration_sec"]
        if abs(actual_duration - expected) > DURATION_TOLERANCE:
            duration_mismatches.append(
                f"{utt['id']}: manifest={expected:.4f}s, actual={actual_duration:.4f}s, "
                f"diff={abs(actual_duration - expected):.4f}s"
            )

        if info.samplerate != EXPECTED_SAMPLERATE:
            format_issues.append(f"{ap}: samplerate={info.samplerate} (expected {EXPECTED_SAMPLERATE})")
        if info.channels != EXPECTED_CHANNELS:
            format_issues.append(f"{ap}: channels={info.channels} (expected {EXPECTED_CHANNELS})")

        if (i + 1) % 500 == 0:
            print(f"  Checked {i + 1}/{len(common)} local audio files...")

    checker.check(
        f"duration_sec matches local audio for all {len(common)} files (tol={DURATION_TOLERANCE}s)",
        len(duration_mismatches) == 0,
        "\n".join(duration_mismatches[:10])
        + (f"\n... and {len(duration_mismatches) - 10} more" if len(duration_mismatches) > 10 else "")
        if duration_mismatches
        else "",
    )

    checker.check(
        f"All {len(common)} local audio files are {EXPECTED_SAMPLERATE}Hz mono",
        len(format_issues) == 0,
        "\n".join(format_issues[:10]) + (f"\n... and {len(format_issues) - 10} more" if len(format_issues) > 10 else "")
        if format_issues
        else "",
    )

    bad_transcripts = []
    for u in manifest["utterances"]:
        t = u.get("transcript", "")
        if not t or not t.strip():
            bad_transcripts.append(f"{u['id']}: empty transcript")
        elif "\x00" in t:
            bad_transcripts.append(f"{u['id']}: contains null bytes")
        else:
            try:
                t.encode("utf-8")
            except UnicodeEncodeError:
                bad_transcripts.append(f"{u['id']}: not valid UTF-8")

    checker.check(
        "All ground truth transcripts are non-empty and well-formed",
        len(bad_transcripts) == 0,
        "\n".join(bad_transcripts[:10]) if bad_transcripts else "",
    )


def phase5_hf_vs_local(hf_wav_files: set[str], audio_dir: Path, temp_dir: Path, checker: Checker):
    print("\n--- Phase 5: HuggingFace vs Local (binary match) ---")

    local_wavs = set()
    for locale_dir in sorted(audio_dir.iterdir()):
        if not locale_dir.is_dir():
            continue
        for wav in locale_dir.glob("*.wav"):
            local_wavs.add(f"{locale_dir.name}/{wav.name}")

    hf_only = hf_wav_files - local_wavs
    local_only = local_wavs - hf_wav_files
    checker.check(
        "HF and local have the exact same set of wav files",
        len(hf_only) == 0 and len(local_only) == 0,
        (f"Only in HF ({len(hf_only)}):\n" + "\n".join(sorted(hf_only)[:10]) + "\n" if hf_only else "")
        + (f"Only in local ({len(local_only)}):\n" + "\n".join(sorted(local_only)[:10]) if local_only else ""),
    )

    common = hf_wav_files & local_wavs
    hash_mismatches = []
    for i, rel_path in enumerate(sorted(common)):
        hf_file = temp_dir / rel_path
        local_file = audio_dir / rel_path
        if not hf_file.exists() or not local_file.exists():
            hash_mismatches.append(
                f"{rel_path}: file missing (HF exists={hf_file.exists()}, local exists={local_file.exists()})"
            )
            continue
        hf_hash = sha256(hf_file)
        local_hash = sha256(local_file)
        if hf_hash != local_hash:
            hash_mismatches.append(f"{rel_path}: HF={hf_hash[:16]}..., local={local_hash[:16]}...")
        if (i + 1) % 500 == 0:
            print(f"  Hashed {i + 1}/{len(common)} files...")

    checker.check(
        f"SHA-256 matches for all {len(common)} files (HF vs local)",
        len(hash_mismatches) == 0,
        "\n".join(hash_mismatches[:10])
        + (f"\n... and {len(hash_mismatches) - 10} more" if len(hash_mismatches) > 10 else "")
        if hash_mismatches
        else "",
    )


def phase_locale_counts(
    manifest: dict,
    hf_utts: dict[str, dict],
    hf_wav_files: set[str],
    audio_dir: Path,
    checker: Checker,
):
    """Cross-check per-locale counts across all four data views."""
    print("\n--- Per-locale count verification ---")

    manifest_counts: dict[str, int] = Counter()
    for u in manifest["utterances"]:
        manifest_counts[u["locale"]] += 1

    hf_meta_counts: dict[str, int] = Counter()
    for key in hf_utts:
        locale = key.split("/", 1)[0]
        hf_meta_counts[locale] += 1

    hf_audio_counts: dict[str, int] = Counter()
    for wav in hf_wav_files:
        locale = wav.split("/", 1)[0]
        hf_audio_counts[locale] += 1

    local_audio_counts: dict[str, int] = Counter()
    for locale_dir in sorted(audio_dir.iterdir()):
        if not locale_dir.is_dir():
            continue
        count = sum(1 for _ in locale_dir.glob("*.wav"))
        if count > 0:
            local_audio_counts[locale_dir.name] = count

    all_locales = sorted(set(manifest_counts) | set(hf_meta_counts) | set(hf_audio_counts) | set(local_audio_counts))

    mismatches = []
    for locale in all_locales:
        mc = manifest_counts.get(locale, 0)
        hm = hf_meta_counts.get(locale, 0)
        ha = hf_audio_counts.get(locale, 0)
        la = local_audio_counts.get(locale, 0)
        if not (mc == hm == ha == la):
            mismatches.append(f"{locale}: manifest={mc}, hf_meta={hm}, hf_audio={ha}, local_audio={la}")

    header = "Per-locale counts:\n" + "\n".join(
        f"  {loc}: manifest={manifest_counts.get(loc, 0)}, hf_meta={hf_meta_counts.get(loc, 0)}, "
        f"hf_audio={hf_audio_counts.get(loc, 0)}, local={local_audio_counts.get(loc, 0)}"
        for loc in all_locales
    )
    print(header)

    checker.check(
        "Per-locale counts match across all 4 sources (manifest, HF metadata, HF audio, local audio)",
        len(mismatches) == 0,
        "\n".join(mismatches) if mismatches else "",
    )


def main():
    parser = argparse.ArgumentParser(description="Three-way data integrity verification")
    parser.add_argument("--manifest", default="manifest.json", type=Path)
    parser.add_argument("--audio-dir", default="audio", type=Path)
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (defaults to HF_TOKEN env var)")
    parser.add_argument(
        "--hf-dir",
        default=None,
        type=Path,
        help="Pre-downloaded HF dataset directory (offline mode, no HF_TOKEN needed)",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        type=Path,
        help="Temp dir for HF downloads (auto-created if not set)",
    )
    args = parser.parse_args()

    offline = args.hf_dir is not None
    token = None
    if not offline:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            print("ERROR: HF_TOKEN env var or --hf-token required (or use --hf-dir for offline mode)")
            sys.exit(1)

    manifest_path = args.manifest.resolve()
    audio_dir = args.audio_dir.resolve()

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        sys.exit(1)
    if not audio_dir.exists():
        print(f"ERROR: audio dir not found: {audio_dir}")
        sys.exit(1)

    hf_dir = args.hf_dir.resolve() if args.hf_dir else None
    if hf_dir and not hf_dir.exists():
        print(f"ERROR: HF dir not found: {hf_dir}")
        sys.exit(1)

    print(f"Manifest:  {manifest_path}")
    print(f"Audio dir: {audio_dir}")
    if hf_dir:
        print(f"HF dir:    {hf_dir} (offline mode)")
    else:
        print(f"HF repo:   {REPO_ID} (online mode)")

    manifest = load_manifest(manifest_path)
    print(f"Loaded {len(manifest['utterances'])} utterances across {len(manifest['locales'])} locales")

    if offline:
        temp_dir = hf_dir
    else:
        temp_dir = (args.temp_dir or Path(tempfile.mkdtemp(prefix="mu-bench-verify-"))).resolve()
        temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"HF data:   {temp_dir}")

    checker = Checker()

    phase0_manifest_consistency(manifest, checker)

    hf_wav_files, hf_all_files = phase1_list_hf(token, hf_dir, checker)

    hf_utts = phase1b_compare_hf_metadata(manifest, hf_all_files, hf_dir, token, temp_dir, checker)

    phase1c_regeneration_check(manifest, hf_all_files, hf_dir, token, temp_dir, checker)

    if not offline:
        phase2_download_hf(hf_wav_files, token, temp_dir, checker)

    phase3_manifest_vs_hf(manifest, hf_wav_files, temp_dir, checker)
    phase4_manifest_vs_local(manifest, audio_dir, checker)
    phase5_hf_vs_local(hf_wav_files, audio_dir, temp_dir, checker)

    phase_locale_counts(manifest, hf_utts, hf_wav_files, audio_dir, checker)

    checker.summary()
    sys.exit(0 if checker.all_passed else 1)


if __name__ == "__main__":
    main()
