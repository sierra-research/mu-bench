"""LLM API utilities for scoring.

Handles parallel OpenAI API calls and JSON response parsing.
Requires OPENAI_API_KEY environment variable.
"""

import concurrent.futures
import json
import os
import time

import requests

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


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

    Retries on 429 (rate limit) and 5xx errors with exponential backoff.
    Logs response body on non-retryable failures for debugging.
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
        "model": "gpt-4.1",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
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
                wait = RETRY_BACKOFF * (2**attempt)
                print(f"  LLM call got {status}, retrying in {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            print(f"  LLM call failed (HTTP {status}): {body}")
            raise
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (2**attempt)
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
