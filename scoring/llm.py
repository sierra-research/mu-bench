"""LLM API utilities for scoring.

Handles parallel OpenAI API calls and JSON response parsing.
Requires OPENAI_API_KEY environment variable.

Judge config (model snapshot, temperature, seed) is pinned for
reproducibility via the SCORING_MODEL / SCORING_TEMPERATURE / SCORING_SEED
env vars. The defaults below are the pinned reference values; override in
.env only if you are intentionally running a drift / sensitivity experiment.

Every call records the effective config on the module-level JUDGE_CONFIG
dict so scoring.score can copy it into scores.json for auditability.
"""

import concurrent.futures
import hashlib
import json
import os
import time

import requests

MAX_RETRIES = 6
RETRY_BACKOFF = 2.0
MAX_BACKOFF_SECONDS = 60.0


# Pinned judge configuration. Defaults are the reference values used to
# produce the leaderboard's published numbers. Set SCORING_MODEL /
# SCORING_TEMPERATURE / SCORING_SEED in the environment only when running
# drift experiments; changing these without re-scoring every submission
# produces an inconsistent leaderboard.
DEFAULT_SCORING_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_SCORING_TEMPERATURE = 0.0
DEFAULT_SCORING_SEED = 7


def _load_judge_config() -> dict:
    """Read the pinned judge config from env, falling back to defaults.

    Returns a dict with model, temperature, seed. Values are snapshot at
    import time so a single scoring run uses one config end-to-end.
    """
    return {
        "model": os.environ.get("SCORING_MODEL", DEFAULT_SCORING_MODEL),
        "temperature": float(os.environ.get("SCORING_TEMPERATURE", DEFAULT_SCORING_TEMPERATURE)),
        "seed": int(os.environ.get("SCORING_SEED", DEFAULT_SCORING_SEED)),
    }


JUDGE_CONFIG = _load_judge_config()


def prompt_sha(prompt_text: str) -> str:
    """Return a short sha256 hex digest of a prompt string for scores.json."""
    if prompt_text is None:
        return ""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


NORMALIZE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "normalize_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "normalized_expected": {"type": "string"},
                "normalized_actual": {"type": "string"},
            },
            "required": ["normalized_expected", "normalized_actual"],
            "additionalProperties": False,
        },
    },
}

# Schema for the new canonical-gold-only normalization prompt (item 1).
# Prompt takes only the gold; output is normalized_expected.
NORMALIZE_GOLD_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "normalize_gold_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "normalized_expected": {"type": "string"},
            },
            "required": ["normalized_expected"],
            "additionalProperties": False,
        },
    },
}

# Schema for the new prediction-only normalization prompt (item 1).
# Prompt takes the already-normalized gold (for style reference) and the
# raw prediction; output is only normalized_actual.
NORMALIZE_PRED_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "normalize_pred_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "normalized_actual": {"type": "string"},
            },
            "required": ["normalized_actual"],
            "additionalProperties": False,
        },
    },
}

SIGNIFICANT_WER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "significant_wer_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "error": {"type": "string"},
                            "reason": {"type": "string"},
                            "score": {"type": "integer"},
                        },
                        "required": ["error", "reason", "score"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["scores"],
            "additionalProperties": False,
        },
    },
}


def call_llm(prompt: str, response_format: dict | None = None) -> str:
    """Call the OpenAI Chat Completions API with retries.

    Uses the pinned JUDGE_CONFIG (model, temperature, seed) so every call is
    reproducible. Retries on 429 (rate limit) and 5xx errors with exponential
    backoff. Logs response body on non-retryable failures.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": JUDGE_CONFIG["model"],
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": JUDGE_CONFIG["temperature"],
        "seed": JUDGE_CONFIG["seed"],
        "max_tokens": 8192,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            body = e.response.text[:500] if e.response is not None else "no response body"
            if status in (429, 500, 502, 503, 529) and attempt < MAX_RETRIES - 1:
                # Honor the server's Retry-After header when present (OpenAI
                # sends seconds on 429 rate-limit responses). Fall back to
                # exponential backoff, capped so one stuck call can't stall
                # the whole batch.
                retry_after = None
                if e.response is not None:
                    raw = e.response.headers.get("Retry-After") or e.response.headers.get("x-ratelimit-reset-requests")
                    try:
                        retry_after = float(raw) if raw is not None else None
                    except ValueError:
                        retry_after = None
                exp = RETRY_BACKOFF * (2**attempt)
                wait = min(MAX_BACKOFF_SECONDS, retry_after if retry_after is not None else exp)
                source = "Retry-After" if retry_after is not None else "backoff"
                print(
                    f"  LLM call got {status}, retrying in {wait:.0f}s via {source} "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})..."
                )
                time.sleep(wait)
                continue
            print(f"  LLM call failed (HTTP {status}): {body}")
            raise
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                wait = min(MAX_BACKOFF_SECONDS, RETRY_BACKOFF * (2**attempt))
                print(f"  LLM call timed out, retrying in {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            raise


def get_responses(prompts, num_workers=10, response_format=None):
    """Get responses from the LLM in parallel.

    Individual failures return None instead of crashing the batch.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for prompt in prompts:
            if prompt is not None:
                futures.append(executor.submit(call_llm, prompt, response_format=response_format))
            else:
                futures.append(None)
        outputs = []
        for i, future in enumerate(futures):
            if future is None:
                outputs.append(None)
            else:
                try:
                    outputs.append(future.result())
                except Exception as e:
                    print(f"  LLM call {i} failed: {e}")
                    outputs.append(None)
    return outputs


def load_responses(responses):
    """Parse JSON responses from the LLM.

    Handles markdown code fences (```json ... ```) and plain JSON.

    Args:
        responses: List of response strings

    Returns:
        List of parsed dicts (or None on parse error)
    """
    for i, response in enumerate(responses):
        if response is None:
            responses[i] = None
            continue
        try:
            responses[i] = json.loads(response)
        except json.JSONDecodeError:
            try:
                responses[i] = json.loads(response.split("```json")[1].split("```")[0])
            except Exception as e:
                print(f"JSON parse error for response {i}: {e}")
                responses[i] = None
    return responses
