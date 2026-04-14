"""Update results/leaderboard.json from all results/*/scores.json files.

Walks the results/ directory, reads each provider's scores.json, and
produces a single leaderboard.json consumed by the leaderboard web app.

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
            locale_results[locale] = {
                "wer": data.get("wer"),
                "significantWer": data.get("significantWer"),
                "qualityScore": data.get("qualityScore"),
                "latencyP50Ms": data.get("latencyP50Ms"),
                "latencyP95Ms": data.get("latencyP95Ms"),
            }

        providers.append(
            {
                "id": provider_id,
                "model": scores.get("model", provider_id),
                "organization": scores.get("organization", ""),
                "modelDate": scores.get("date", ""),
                "localeResults": locale_results,
            }
        )

    locales = []
    for code in sorted(all_locales):
        info = LOCALE_LABELS.get(code, {"label": code, "flag": ""})
        locales.append({"code": code, "label": info["label"], "flag": info["flag"]})

    leaderboard = {"locales": locales, "providers": providers}

    output_path = results_dir / "leaderboard.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)

    print(f"Updated {output_path} with {len(providers)} providers and {len(locales)} locales")


if __name__ == "__main__":
    main()
