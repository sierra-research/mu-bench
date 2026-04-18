"""Tests for ``scoring.validate``.

Covers the rules introduced by the fairness-fixes plan:
  * ``metadata.yaml`` must include a ``config:`` block with exactly the
    six required keys, values must be ``default`` or come with an
    ``override:`` line in ``notes``.
  * ``latency.json`` must use the new schema with ``meta.protocol`` +
    ``meta.region`` and per-entry fields matching the declared protocol.
    The legacy flat ``{"<locale>/<uid>": ms}`` schema was dropped with
    the rollout PR and is now a hard rejection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from scoring.validate import (  # noqa: E402
    REQUIRED_CONFIG_KEYS,
    _validate_config_block,
    _validate_latency_new_schema,
    validate_latency,
    validate_metadata,
)


def _write_metadata(tmp_path: Path, doc: dict) -> Path:
    """Write a YAML mapping under tmp_path/metadata.yaml and return the dir."""
    yaml = pytest.importorskip("yaml")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "metadata.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return sub


def _default_config_block() -> dict:
    return {k: "default" for k in REQUIRED_CONFIG_KEYS}


def test_config_block_accepts_all_defaults():
    issues = _validate_config_block(
        {
            "model": "X",
            "organization": "Y",
            "config": _default_config_block(),
        }
    )
    assert issues == []


def test_config_block_requires_all_keys():
    partial = _default_config_block()
    del partial["beamSize"]
    issues = _validate_config_block({"config": partial})
    assert any("beamSize" in i for i in issues)


def test_config_block_rejects_extra_keys():
    cfg = _default_config_block()
    cfg["mysteryKey"] = "default"
    issues = _validate_config_block({"config": cfg})
    assert any("mysteryKey" in i and "unknown" in i.lower() for i in issues)


def test_config_block_rejects_non_default_without_override_note():
    cfg = _default_config_block()
    cfg["beamSize"] = "10"
    issues = _validate_config_block({"config": cfg, "notes": "no override line here"})
    assert any("override:" in i and "beamSize" in i for i in issues)


def test_config_block_accepts_non_default_with_override_note():
    cfg = _default_config_block()
    cfg["beamSize"] = "10"
    cfg["keywordBoosting"] = "enabled: 42 phrases"
    issues = _validate_config_block(
        {
            "config": cfg,
            "notes": "override: beamSize because our prod default is 10\noverride: keywordBoosting we shipped this\n",
        }
    )
    assert issues == []


def test_validate_metadata_rejects_missing_config_block(tmp_path):
    sub = _write_metadata(
        tmp_path,
        {"model": "X", "organization": "Y", "version": "v1", "date": "2026-01-01"},
    )
    issues = validate_metadata(sub)
    assert any("config" in i for i in issues)


def test_validate_metadata_accepts_default_config(tmp_path):
    sub = _write_metadata(
        tmp_path,
        {
            "model": "X",
            "organization": "Y",
            "version": "v1",
            "date": "2026-01-01",
            "config": _default_config_block(),
        },
    )
    issues = validate_metadata(sub)
    assert issues == []


# ---- latency.json ----


def _submitted(submitted_ids: dict[str, list[str]]) -> dict[str, set[str]]:
    return {k: set(v) for k, v in submitted_ids.items()}


def test_validate_latency_rejects_legacy_flat_schema(tmp_path):
    """Legacy flat ``{"<locale>/<uid>": ms}`` was accepted during the
    fairness-fixes back-compat window; the rollout PR drops it. Now
    submitting a flat map must be a hard rejection with a pointer at
    the new ``meta`` + ``measurements`` shape.
    """
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "latency.json").write_text(
        json.dumps({"en-US/u1": 120.0, "en-US/u2": 150}),
        encoding="utf-8",
    )
    issues, warnings = validate_latency(sub, _submitted({"en-US": ["u1", "u2"]}))
    assert any("meta" in i and "measurements" in i for i in issues), issues
    assert any("legacy" in i.lower() for i in issues), issues
    assert warnings == []


def test_new_schema_batch_requires_roundtrip():
    data = {
        "meta": {"protocol": "batch", "region": "us-east-1"},
        "measurements": {
            "en-US/u1": {"roundTripMs": 120.0},
            "en-US/u2": {"notARealField": 1},
        },
    }
    warnings: list[str] = []
    issues = _validate_latency_new_schema(data, _submitted({"en-US": ["u1", "u2"]}), warnings)
    assert any("roundTripMs" in i for i in issues)


def test_new_schema_streaming_requires_ttft_and_complete():
    data = {
        "meta": {"protocol": "streaming", "region": "us-east-1"},
        "measurements": {
            "en-US/u1": {"ttftMs": 100, "completeMs": 500},
            "en-US/u2": {"ttftMs": 100},  # missing completeMs
        },
    }
    warnings: list[str] = []
    issues = _validate_latency_new_schema(data, _submitted({"en-US": ["u1", "u2"]}), warnings)
    assert any("completeMs" in i for i in issues)


def test_new_schema_rejects_unknown_region():
    data = {
        "meta": {"protocol": "batch", "region": "mars-base-1"},
        "measurements": {"en-US/u1": {"roundTripMs": 120}},
    }
    warnings: list[str] = []
    issues = _validate_latency_new_schema(data, _submitted({"en-US": ["u1"]}), warnings)
    assert any("mars-base-1" in i for i in issues)


def test_new_schema_accepts_us_multi_region():
    """Multi-region 'us' is in the allowlist as an escape hatch for
    providers like Google Chirp-3 whose model under test isn't routable
    via a single region. See ALLOWED_LATENCY_REGIONS in validate.py.
    """
    data = {
        "meta": {"protocol": "batch", "region": "us"},
        "measurements": {"en-US/u1": {"roundTripMs": 120}},
    }
    warnings: list[str] = []
    issues = _validate_latency_new_schema(data, _submitted({"en-US": ["u1"]}), warnings)
    assert issues == []


def test_new_schema_rejects_unknown_protocol():
    data = {
        "meta": {"protocol": "hybrid", "region": "us-east-1"},
        "measurements": {"en-US/u1": {"roundTripMs": 120}},
    }
    warnings: list[str] = []
    issues = _validate_latency_new_schema(data, _submitted({"en-US": ["u1"]}), warnings)
    assert any("protocol" in i for i in issues)


def test_new_schema_accepts_valid_batch_submission():
    data = {
        "meta": {
            "protocol": "batch",
            "region": "us-east-1",
            "clientLocation": "aws:us-east-1",
            "concurrency": 1,
        },
        "measurements": {
            "en-US/u1": {"roundTripMs": 120.0},
            "en-US/u2": {"roundTripMs": 150.2},
        },
    }
    warnings: list[str] = []
    issues = _validate_latency_new_schema(data, _submitted({"en-US": ["u1", "u2"]}), warnings)
    assert issues == []


def test_new_schema_warns_on_non_unit_concurrency():
    data = {
        "meta": {
            "protocol": "batch",
            "region": "us-east-1",
            "concurrency": 8,
        },
        "measurements": {"en-US/u1": {"roundTripMs": 120}},
    }
    warnings: list[str] = []
    issues = _validate_latency_new_schema(data, _submitted({"en-US": ["u1"]}), warnings)
    assert issues == []
    assert any("concurrency=8" in w for w in warnings)


def test_new_schema_coverage_gap_reports_missing_entries():
    data = {
        "meta": {"protocol": "batch", "region": "us-east-1"},
        "measurements": {"en-US/u1": {"roundTripMs": 120.0}},
    }
    warnings: list[str] = []
    issues = _validate_latency_new_schema(data, _submitted({"en-US": ["u1", "u2"]}), warnings)
    assert any("u2" in i for i in issues)


def test_latency_stats_new_schema_produces_complete_fields(tmp_path):
    """End-to-end smoke: feed the new latency schema through and assert
    we get ``completeP95Ms`` on the per-locale bucket.
    """
    from scripts.latency_stats import compute_latency_stats

    manifest = {
        "utterances": [
            {"id": "u1", "locale": "en-US", "transcript": ""},
            {"id": "u2", "locale": "en-US", "transcript": ""},
            {"id": "u3", "locale": "en-US", "transcript": ""},
        ]
    }
    latency = {
        "meta": {"protocol": "batch", "region": "us-east-1"},
        "measurements": {
            "en-US/u1": {"roundTripMs": 100},
            "en-US/u2": {"roundTripMs": 200},
            "en-US/u3": {"roundTripMs": 300},
        },
    }
    stats = compute_latency_stats(latency, manifest)
    en = stats["locales"]["en-US"]
    # completeP95Ms aliases roundTripP95Ms for batch.
    assert en["completeP95Ms"] == pytest.approx(en["roundTripP95Ms"])
    # Legacy ``latencyP95Ms`` alias was dropped with the rollout PR.
    assert "latencyP95Ms" not in en
    assert stats["protocol"] == "batch"
    assert stats["region"] == "us-east-1"


def test_latency_stats_rejects_legacy_flat_schema():
    """The old flat ``{"<locale>/<uid>": ms}`` schema is no longer parsed."""
    from scripts.latency_stats import compute_latency_stats

    manifest = {"utterances": [{"id": "u1", "locale": "en-US", "transcript": ""}]}
    legacy = {"en-US/u1": 100}
    with pytest.raises(ValueError, match="new schema"):
        compute_latency_stats(legacy, manifest)


def test_latency_stats_streaming_emits_ttft_and_complete(tmp_path):
    from scripts.latency_stats import compute_latency_stats

    manifest = {
        "utterances": [
            {"id": "u1", "locale": "en-US", "transcript": ""},
            {"id": "u2", "locale": "en-US", "transcript": ""},
        ]
    }
    latency = {
        "meta": {"protocol": "streaming", "region": "us-east-1"},
        "measurements": {
            "en-US/u1": {"ttftMs": 100, "completeMs": 400},
            "en-US/u2": {"ttftMs": 150, "completeMs": 500},
        },
    }
    stats = compute_latency_stats(latency, manifest)
    en = stats["locales"]["en-US"]
    assert "ttftP95Ms" in en
    assert "completeP95Ms" in en
    assert stats["protocol"] == "streaming"


def test_update_leaderboard_prefers_complete_field(tmp_path):
    """Round-trip: write a scores.json, run update_leaderboard, and
    confirm the completeP95Ms value is in leaderboard.json.
    """
    # Minimal fixture so update_leaderboard has something to chew on.
    # We monkeypatch Path("results") via chdir.
    import scoring.update_leaderboard as ul

    results_root = tmp_path / "results"
    prov = results_root / "acme-stt"
    prov.mkdir(parents=True)
    (prov / "scores.json").write_text(
        json.dumps(
            {
                "model": "Acme-STT",
                "organization": "Acme",
                "date": "2026-01-01",
                "locales": {
                    "en-US": {
                        "wer": 0.05,
                        "significantWer": 0.08,
                        "completeP50Ms": 100,
                        "completeP95Ms": 200,
                        "roundTripP50Ms": 100,
                        "roundTripP95Ms": 200,
                    }
                },
                "overall": {"wer": 0.05, "significantWer": 0.08, "completeP95Ms": 200},
                "latencyMeta": {"protocol": "batch", "region": "us-east-1"},
                "meta": {
                    "config": {
                        "beamSize": "default",
                        "languageHint": "default",
                        "customVocabulary": "default",
                        "noiseSuppression": "default",
                        "domainAdaptation": "default",
                        "keywordBoosting": "default",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    import os

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        ul.main()
        lb = json.loads((results_root / "leaderboard.json").read_text())
    finally:
        os.chdir(cwd)

    prov_entry = next(p for p in lb["providers"] if p["id"] == "acme-stt")
    assert prov_entry["localeResults"]["en-US"]["completeP95Ms"] == 200
    # Legacy ``latencyP95Ms`` alias was dropped from update_leaderboard.py.
    assert "latencyP95Ms" not in prov_entry["localeResults"]["en-US"]
    assert prov_entry["latencyMeta"] == {"protocol": "batch", "region": "us-east-1"}
    assert prov_entry["inferenceConfig"]["beamSize"] == "default"
    # Overall block plumbed through so the UI doesn't re-average.
    assert prov_entry["overall"]["significantWer"] == 0.08
