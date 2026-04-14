"""Test whether the updated normalization prompt correctly handles Chinese homophones.

Loads test cases from homophone_test_cases.json and runs the updated
NORMALIZE_AGAINST_GOLD_PROMPT against them.

Usage:
    python3 scripts/test_homophone_normalization.py
    python3 scripts/test_homophone_normalization.py --max-cases 10
"""

import argparse
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

# Add project root to path so we can import prompt.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prompt import NORMALIZE_AGAINST_GOLD_PROMPT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CASES_PATH = os.path.join(SCRIPT_DIR, "homophone_test_cases.json")


def load_config():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Add it to .env or export it.")
    return NORMALIZE_AGAINST_GOLD_PROMPT, api_key


def call_normalize(prompt_template: str, api_key: str, gold: str, predicted: str) -> str | None:
    prompt = prompt_template.replace("{expected_transcript}", gold).replace("{actual_transcript}", predicted)
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4096,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "normalize_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"normalized_actual": {"type": "string"}},
                    "required": ["normalized_actual"],
                    "additionalProperties": False,
                },
            },
        },
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed.get("normalized_actual")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (429, 500, 502, 503, 529) and attempt < 2:
                wait = 2.0 * (2**attempt)
                print(f"  Rate limited ({status}), retrying in {wait:.0f}s...")
                time.sleep(wait)
                continue
            print(f"  API error: {status}")
            return None
        except Exception as e:
            print(f"  Error: {e}")
            return None


def normalize_for_comparison(text: str) -> str:
    for ch in "，。、！？,. !?\"'''：:；;":
        text = text.replace(ch, "")
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Test Chinese homophone normalization")
    parser.add_argument("--max-cases", type=int, default=None, help="Max cases to test")
    args = parser.parse_args()

    prompt_template, api_key = load_config()

    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if args.max_cases:
        cases = cases[: args.max_cases]

    print(f"Testing {len(cases)} cases against updated normalization prompt\n")
    print("=" * 80)

    results = {"pass": 0, "fail": 0, "error": 0}

    for i, case in enumerate(cases):
        case_type = case["type"]
        label = f"[{case_type.upper()}]"
        print(f"\n{label} Case {i + 1}/{len(cases)} ({case['provider']})")
        print(f"  Subs: {', '.join(case['subs'])}")
        print(f"  Gold:      {case['gold']}")
        print(f"  Predicted: {case['predicted']}")

        normalized = call_normalize(prompt_template, api_key, case["gold"], case["predicted"])

        if normalized is None:
            print("  Normalized: ERROR (API call failed)")
            results["error"] += 1
            continue

        print(f"  Normalized: {normalized}")

        gold_clean = normalize_for_comparison(case["gold"])
        norm_clean = normalize_for_comparison(normalized)
        pred_clean = normalize_for_comparison(case["predicted"])

        if case_type == "negative":
            if norm_clean == gold_clean:
                print("  Result: FAIL — incorrectly normalized toward gold (should have kept as-is)")
                results["fail"] += 1
            else:
                print("  Result: PASS — correctly did NOT normalize general vocabulary homophones")
                results["pass"] += 1
        else:
            if norm_clean == gold_clean:
                print("  Result: PASS — correctly normalized homophone name to match gold")
                results["pass"] += 1
            elif norm_clean == pred_clean:
                print("  Result: FAIL — did NOT normalize homophone (kept predicted as-is)")
                results["fail"] += 1
            else:
                print("  Result: PARTIAL — normalized to something different from both gold and predicted")
                results["fail"] += 1

    print("\n" + "=" * 80)
    print(f"\nResults: {results['pass']} pass, {results['fail']} fail, {results['error']} errors")
    total = results["pass"] + results["fail"]
    if total > 0:
        print(f"Pass rate: {100 * results['pass'] / total:.1f}%")


if __name__ == "__main__":
    main()
