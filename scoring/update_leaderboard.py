"""Update results/leaderboard.json from all results/*/scores.json files.

Walks the results/ directory, reads each provider's scores.json, and
produces a single leaderboard.json consumed by the leaderboard web app.

Propagates the fields introduced by the fairness-fixes plan so the UI
can render them:

  * ``completeP50Ms`` / ``completeP95Ms`` — unified cross-protocol
    latency metric (batch round-trip or streaming time-to-complete).
  * ``ttftP50Ms`` / ``ttftP95Ms`` — streaming-only, rendered as the
    ``+TTFT`` annotation in the UI.
  * ``latencyMeta`` — ``{protocol, region}`` so the UI can stamp a
    badge on each row. Region is also recorded here (not shown in the
    main row per plan, but available via the data file).
  * ``meta.config`` — inference-config disclosure surfaced in the
    provider detail panel.

Legacy ``latencyP50Ms`` / ``latencyP95Ms`` aliases were dropped with
the rollout PR; the UI sorts on ``completeP95Ms`` directly.

Usage:
    python -m scoring.update_leaderboard
"""

import json
from pathlib import Path

LOCALE_LABELS = {
    "en-US": {"label": "English (US)", "flag": "\U0001f1fa\U0001f1f8"},
    "es-MX": {"label": "Spanish (MX)", "flag": "\U0001f1f2\U0001f1fd"},
    "tr-TR": {"label": "Turkish", "flag": "\U0001f1f9\U0001f1f7"},
    "vi-VN": {"label": "Vietnamese", "flag": "\U0001f1fb\U0001f1f3"},
    "zh-CN": {"label": "Chinese (CN)", "flag": "\U0001f1e8\U0001f1f3"},
}

# Per-locale latency fields propagated into leaderboard.json.
LATENCY_LOCALE_FIELDS = (
    "completeP50Ms",
    "completeP95Ms",
    "roundTripP50Ms",
    "roundTripP95Ms",
    "ttftP50Ms",
    "ttftP95Ms",
)


def _extract_locale_fields(data: dict) -> dict:
    out = {
        "wer": data.get("wer"),
        "significantWer": data.get("significantWer"),
    }
    for field in LATENCY_LOCALE_FIELDS:
        if field in data:
            out[field] = data.get(field)
    return out


def main():
    results_dir = Path("results")
    providers = []
    all_locales = set()

    for scores_path in sorted(results_dir.glob("*/scores.json")):
        provider_id = scores_path.parent.name
        with open(scores_path, "r", encoding="utf-8") as f:
            scores = json.load(f)

        locale_results = {}
        for locale, data in scores.get("locales", {}).items():
            all_locales.add(locale)
            locale_results[locale] = _extract_locale_fields(data)

        overall = scores.get("overall") or {}
        latency_meta = scores.get("latencyMeta") or {}
        meta = scores.get("meta") or {}

        provider_entry = {
            "id": provider_id,
            "model": scores.get("model", provider_id),
            "organization": scores.get("organization", ""),
            "modelDate": scores.get("date", ""),
            "localeResults": locale_results,
        }
        if overall:
            provider_entry["overall"] = _extract_locale_fields(overall) if isinstance(overall, dict) else None
        if latency_meta:
            provider_entry["latencyMeta"] = {
                "protocol": latency_meta.get("protocol", "batch"),
                "region": latency_meta.get("region", "unknown"),
            }
        # Surface the declared inference-config block for the provider
        # detail panel. Empty dict is fine; UI shows nothing in that case.
        config_block = meta.get("config") if isinstance(meta, dict) else None
        if isinstance(config_block, dict) and config_block:
            provider_entry["inferenceConfig"] = config_block
        providers.append(provider_entry)

    locales = []
    for code in sorted(all_locales):
        info = LOCALE_LABELS.get(code, {"label": code, "flag": ""})
        locales.append({"code": code, "label": info["label"], "flag": info["flag"]})

    if not providers:
        print("WARNING: No providers found in results/*/scores.json — leaderboard will be empty")

    leaderboard = {"locales": locales, "providers": providers}

    output_path = results_dir / "leaderboard.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)

    print(f"Updated {output_path} with {len(providers)} providers and {len(locales)} locales")


if __name__ == "__main__":
    main()
