"""Transcribe benchmark audio using provider batch APIs directly.

Reads per-utterance .wav files and calls provider APIs with no middleware.

Usage:
    python scripts/transcribe.py --provider deepgram-nova3 --output-dir submissions/raw/deepgram-nova3

    # Limit to one locale
    python scripts/transcribe.py --provider deepgram-nova3 --locale en-US --output-dir /tmp/test

    # Limit conversations (for validation)
    python scripts/transcribe.py --provider deepgram-nova3 --locale en-US --max-conversations 15 --output-dir /tmp/val
"""

import argparse
import asyncio
import base64
import io
import json
import os
import time
import uuid
import wave
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf
import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def read_wav_as_pcm16(wav_path: Path) -> tuple[np.ndarray, int]:
    """Read any WAV file and return (samples_int16, sample_rate)."""
    data, sr = sf.read(wav_path, dtype="int16")
    return data, sr


def pcm16_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Convert int16 numpy array to in-memory WAV file bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Manifest / data helpers
# ---------------------------------------------------------------------------


def build_utterance_items(
    manifest: dict,
    locales: list[str],
    max_conversations: int | None = None,
    target_sr: int | None = None,
):
    """Build list of (utterance_id, locale, wav_bytes) items for utterance mode.

    Reads each utterance's audio_path directly from the manifest.
    If target_sr is set, resamples audio to the given sample rate.
    """
    if target_sr:
        from math import gcd

        from scipy.signal import resample_poly

    convos_by_key: dict[tuple[str, str], list[dict]] = {}
    for utt in manifest["utterances"]:
        if utt["locale"] not in locales:
            continue
        key = (utt["locale"], utt["conversation_id"])
        convos_by_key.setdefault(key, []).append(utt)

    items = []
    convo_count_by_locale: dict[str, int] = {}
    for (locale, convo_id), utts in sorted(convos_by_key.items()):
        convo_count_by_locale.setdefault(locale, 0)
        if max_conversations is not None and convo_count_by_locale[locale] >= max_conversations:
            continue
        convo_count_by_locale[locale] += 1

        for utt in sorted(utts, key=lambda u: u["turn_index"]):
            wav_path = Path(utt["audio_path"])
            if not wav_path.exists():
                print(f"  WARNING: audio not found at {wav_path}, skipping {utt['id']}")
                continue

            samples, sr = read_wav_as_pcm16(wav_path)
            if target_sr and target_sr != sr:
                g = gcd(target_sr, sr)
                samples = resample_poly(samples, target_sr // g, sr // g).astype(np.int16)
                sr = target_sr
            wav_bytes = pcm16_to_wav_bytes(samples, sr)
            items.append({"id": utt["id"], "locale": locale, "wav_bytes": wav_bytes})

    return items


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

LOCALE_TO_ISO639_1 = {
    "en-US": "en",
    "es-MX": "es",
    "tr-TR": "tr",
    "vi-VN": "vi",
    "zh-CN": "zh",
}

LOCALE_TO_SCRIBE = {
    "en-US": "eng",
    "es-MX": "spa",
    "tr-TR": "tur",
    "vi-VN": "vie",
    "zh-CN": "zho",
}

DEEPGRAM_LOCALE_MAP = {
    "en-US": "en-US",
    "es-MX": "es",
    "tr-TR": "tr",
    "vi-VN": "vi",
    "zh-CN": "zh",
}

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
RETRYABLE_STATUSES = {429, 500, 502, 503, 529}


# ---------------------------------------------------------------------------
# Persistent client caches
#
# Google and OpenAI SDKs both expect clients to be constructed once and reused
# across requests. Fresh-per-request construction wastes auth (OAuth token
# mint for Google, httpx pool init for OpenAI), TLS handshakes, and gRPC
# channel setup. Production code always reuses. The caches below hold one
# client per run and are reset at the start of each run_transcription call.
#
# Deepgram, ElevenLabs, and Azure use plain HTTP with static key-in-header
# auth, so the shared aiohttp.ClientSession created in run_transcription is
# already the right pattern — no additional per-provider caching needed.
# ---------------------------------------------------------------------------

_google_client_cache: dict = {}
_openai_client_cache: dict = {}


def _reset_clients() -> None:
    """Clear persistent-client caches. Called at the start of run_transcription."""
    _google_client_cache.clear()
    _openai_client_cache.clear()


def _get_google_client():
    """Return a persistent SpeechAsyncClient, creating it on first call.

    Also caches project_id and region so callers don't re-parse credentials.
    GOOGLE_SPEECH_REGION defaults to "us" (multi-region endpoint). Set to a
    specific region like "us-central1" for strict reproducibility.
    """
    if "client" in _google_client_cache:
        return _google_client_cache

    from google.cloud.speech_v2 import SpeechAsyncClient

    creds_json = os.environ["GOOGLE_SPEECH_CREDENTIALS"]
    account_dict = json.loads(creds_json)
    region = os.environ.get("GOOGLE_SPEECH_REGION", "us")

    client = SpeechAsyncClient.from_service_account_info(
        account_dict,
        client_options={"api_endpoint": f"{region}-speech.googleapis.com"},
    )
    _google_client_cache["client"] = client
    _google_client_cache["project_id"] = account_dict["project_id"]
    _google_client_cache["region"] = region
    return _google_client_cache


def _get_openai_client():
    """Return a persistent AsyncOpenAI client, creating it on first call.

    The OpenAI SDK explicitly recommends reusing a single client instance
    so its internal httpx connection pool can keep TLS connections warm.
    """
    if "client" in _openai_client_cache:
        return _openai_client_cache["client"]

    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    _openai_client_cache["client"] = client
    return client


async def _post_with_retry(session: aiohttp.ClientSession, url: str, headers: dict, data, provider: str):
    """POST with exponential backoff on rate-limit and server errors."""
    for attempt in range(MAX_RETRIES):
        async with session.post(url, headers=headers, data=data) as resp:
            if resp.status in RETRYABLE_STATUSES and attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (2**attempt)
                print(f"  {provider} {resp.status}, retrying in {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                await asyncio.sleep(wait)
                continue
            if resp.status != 200:
                body = await resp.text()
                raise Exception(f"{provider} {resp.status}: {body[:300]}")
            return await resp.json()


async def transcribe_deepgram(session: aiohttp.ClientSession, wav_bytes: bytes, locale: str) -> str:
    api_key = os.environ["DEEPGRAM_API_KEY"]
    dg_locale = DEEPGRAM_LOCALE_MAP.get(locale, locale)
    url = f"https://api.deepgram.com/v1/listen?model=nova-3&language={dg_locale}&punctuate=true&smart_format=false"
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"}
    data = await _post_with_retry(session, url, headers, wav_bytes, "Deepgram")
    return data["results"]["channels"][0]["alternatives"][0]["transcript"]


GOOGLE_LOCALE_MAP = {"zh-CN": "cmn-Hans-CN", "zh-HK": "yue-Hant-HK"}


async def transcribe_google(session: aiohttp.ClientSession, wav_bytes: bytes, locale: str) -> str:
    # Session is unused: Google Speech v2 uses gRPC via SpeechAsyncClient, not HTTP.
    # The client is persistent (see _get_google_client) so auth, gRPC channel, and
    # TLS handshake are paid once per run rather than per request.
    from google.cloud.speech_v2.types import cloud_speech

    cache = _get_google_client()
    client = cache["client"]
    project_id = cache["project_id"]
    region = cache["region"]
    google_locale = GOOGLE_LOCALE_MAP.get(locale, locale)

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[google_locale],
        model="chirp_3",
    )
    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{project_id}/locations/{region}/recognizers/_",
        config=config,
        content=wav_bytes,
    )
    response = await client.recognize(request=request)
    for result in response.results:
        if result.alternatives:
            return result.alternatives[0].transcript.strip()
    return ""


async def transcribe_azure(session: aiohttp.ClientSession, wav_bytes: bytes, locale: str) -> str:
    # Azure region is baked into AZURE_SPEECH_ENDPOINT (e.g. eastus2 in the host).
    # Subscription-key auth means no token to mint/cache, so the shared aiohttp
    # ClientSession is the right pattern here.
    endpoint = os.environ["AZURE_SPEECH_ENDPOINT"]
    api_key = os.environ["AZURE_SPEECH_KEY"]
    session_id = str(uuid.uuid4()).replace("-", "")

    data = aiohttp.FormData()
    data.add_field("audio", wav_bytes, filename="audio.wav", content_type="audio/wav")
    definition = {"locales": [locale]}
    data.add_field("definition", json.dumps(definition), content_type="application/json")
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    url = f"{endpoint}/speechtotext/transcriptions:transcribe?api-version=2025-10-15&X-ConnectionId={session_id}"

    async with session.post(url, headers=headers, data=data) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise Exception(f"Azure {resp.status}: {body[:300]}")
        json_data = await resp.json()

    if json_data.get("combinedPhrases") and len(json_data["combinedPhrases"]) > 0:
        return json_data["combinedPhrases"][0].get("text", "")
    return ""


async def transcribe_elevenlabs(session: aiohttp.ClientSession, wav_bytes: bytes, locale: str) -> str:
    api_key = os.environ["ELEVENLABS_API_KEY"]
    scribe_locale = LOCALE_TO_SCRIBE.get(locale, locale)

    data = aiohttp.FormData()
    data.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
    data.add_field("model_id", "scribe_v2")
    if scribe_locale:
        data.add_field("language_code", scribe_locale)
    data.add_field("tag_audio_events", "false")
    headers = {"xi-api-key": api_key}
    url = "https://api.elevenlabs.io/v1/speech-to-text?enable_logging=false"

    async with session.post(url, headers=headers, data=data) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise Exception(f"ElevenLabs {resp.status}: {body[:300]}")
        json_data = await resp.json()
    return json_data.get("text", "")


async def transcribe_openai(session: aiohttp.ClientSession, wav_bytes: bytes, locale: str) -> str:
    # OpenAI SDK has its own httpx pool — _get_openai_client returns a persistent
    # AsyncOpenAI so connections stay warm across requests (SDK-recommended).
    iso_locale = LOCALE_TO_ISO639_1.get(locale, locale[:2])
    client = _get_openai_client()
    response = await client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=("audio.wav", wav_bytes, "audio/wav"),
        language=iso_locale,
    )
    return response.text.strip()


async def transcribe_openai_mini(session: aiohttp.ClientSession, wav_bytes: bytes, locale: str) -> str:
    iso_locale = LOCALE_TO_ISO639_1.get(locale, locale[:2])
    client = _get_openai_client()
    response = await client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=("audio.wav", wav_bytes, "audio/wav"),
        language=iso_locale,
    )
    return response.text.strip()


LOCALE_NAMES = {
    "en-US": "English",
    "es-MX": "Spanish",
    "tr-TR": "Turkish",
    "vi-VN": "Vietnamese",
    "zh-CN": "Chinese",
}


async def transcribe_openai_gpt_audio(session: aiohttp.ClientSession, wav_bytes: bytes, locale: str) -> str:
    lang_name = LOCALE_NAMES.get(locale, "")
    lang_hint = f" The audio is in {lang_name}." if lang_name else ""

    client = _get_openai_client()
    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    response = await client.chat.completions.create(
        model="gpt-audio-1.5",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "system",
                "content": f"Transcribe the provided audio exactly. Return only the transcript text.{lang_hint}",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav",
                        },
                    }
                ],
            },
        ],
    )
    return response.choices[0].message.audio.transcript.strip()


PROVIDERS = {
    "deepgram-nova3": transcribe_deepgram,
    "google-chirp3": transcribe_google,
    "azure": transcribe_azure,
    "elevenlabs-scribe-v2": transcribe_elevenlabs,
    "openai-gpt4o-transcribe": transcribe_openai,
    "openai-gpt4o-mini-transcribe": transcribe_openai_mini,
    "openai-gpt-audio-1.5": transcribe_openai_gpt_audio,
}

PROVIDER_METADATA = {
    "deepgram-nova3": {"model": "Nova-3", "organization": "Deepgram"},
    "google-chirp3": {"model": "Chirp-3", "organization": "Google"},
    "azure": {"model": "Azure-Speech-v1", "organization": "Microsoft"},
    "elevenlabs-scribe-v2": {"model": "Scribe-v2", "organization": "ElevenLabs"},
    "openai-gpt4o-transcribe": {"model": "GPT-4o-Transcribe", "organization": "OpenAI"},
    "openai-gpt4o-mini-transcribe": {"model": "GPT-4o-Mini-Transcribe", "organization": "OpenAI"},
    "openai-gpt-audio-1.5": {"model": "GPT-Audio-1.5", "organization": "OpenAI"},
}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_transcription(args):
    _reset_clients()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    all_locales = manifest["locales"]
    locales = [args.locale] if args.locale else all_locales

    provider_fn = PROVIDERS[args.provider]

    required_env = {
        "deepgram-nova3": ["DEEPGRAM_API_KEY"],
        "google-chirp3": ["GOOGLE_SPEECH_CREDENTIALS"],
        "azure": ["AZURE_SPEECH_ENDPOINT", "AZURE_SPEECH_KEY"],
        "elevenlabs-scribe-v2": ["ELEVENLABS_API_KEY"],
        "openai-gpt4o-transcribe": ["OPENAI_API_KEY"],
        "openai-gpt4o-mini-transcribe": ["OPENAI_API_KEY"],
        "openai-gpt-audio-1.5": ["OPENAI_API_KEY"],
    }
    missing = [k for k in required_env.get(args.provider, []) if not os.environ.get(k)]
    if missing:
        print(f"ERROR: Missing required environment variables for {args.provider}: {', '.join(missing)}")
        print("Set them in your .env file or export them before running.")
        return

    print(f"Provider: {args.provider}")
    print(f"Locales: {locales}")
    print(f"Concurrency: {args.concurrency}")
    print()

    resample_hz = getattr(args, "resample_hz", None)
    if resample_hz:
        print(f"Resampling audio to {resample_hz} Hz")

    print("Loading audio items...")
    items = build_utterance_items(manifest, locales, args.max_conversations, target_sr=resample_hz)

    # Filter out items that already have output files (skip existing)
    filtered = []
    skipped = 0
    for item in items:
        out_path = output_dir / item["locale"] / f"{item['id']}.txt"
        if out_path.exists():
            skipped += 1
        else:
            filtered.append(item)
    items = filtered
    if skipped:
        print(f"Skipped {skipped} items with existing output files")

    print(f"Loaded {len(items)} items to transcribe")
    if not items:
        print("Nothing to do.")
        return

    # Write metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.yaml"
    if not meta_path.exists():
        today = time.strftime("%Y-%m-%d")
        meta = {
            **PROVIDER_METADATA[args.provider],
            "version": today,
            "date": today,
        }
        with open(meta_path, "w") as f:
            yaml.dump(meta, f, default_flow_style=False)

    sem = asyncio.Semaphore(args.concurrency)
    completed = 0
    failed = 0
    total = len(items)
    start_time = time.time()
    latency_records: dict[str, float] = {}

    async def process_one(item: dict, session: aiohttp.ClientSession):
        nonlocal completed, failed
        async with sem:
            item_id = item["id"]
            locale = item["locale"]
            out_path = output_dir / locale / f"{item_id}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                t0 = time.monotonic()
                transcript = await provider_fn(session, item["wav_bytes"], locale)
                latency_ms = (time.monotonic() - t0) * 1000
                out_path.write_text(transcript, encoding="utf-8")
                latency_records[f"{locale}/{item_id}"] = round(latency_ms, 1)
                completed += 1
            except Exception as e:
                failed += 1
                print(f"  ERROR {item_id}: {e}")

            done = completed + failed
            if done % 10 == 0 or done == total:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  Progress: {done}/{total} ({completed} ok, {failed} err) [{rate:.1f} items/s]")

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [process_one(item, session) for item in items]
        await asyncio.gather(*tasks)

    # Merge with any existing latency data and write latency.json
    latency_path = output_dir / "latency.json"
    if latency_path.exists():
        with open(latency_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(latency_records)
        latency_records = existing
    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(latency_records, f, indent=2, ensure_ascii=False)
    print(f"\nLatency recorded for {len(latency_records)} utterances -> {latency_path}")

    elapsed = time.time() - start_time
    print(f"Done. {completed} succeeded, {failed} failed in {elapsed:.1f}s")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe benchmark audio using provider batch APIs")
    parser.add_argument(
        "--provider",
        required=True,
        choices=list(PROVIDERS.keys()),
        help="Transcription provider",
    )
    parser.add_argument("--manifest", default="manifest.json", help="Path to manifest.json")
    parser.add_argument("--output-dir", required=True, help="Output submission directory")
    parser.add_argument("--concurrency", type=int, default=10, help="Max parallel API calls")
    parser.add_argument("--locale", default=None, help="Limit to a single locale (e.g., en-US)")
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Max conversations per locale",
    )
    parser.add_argument(
        "--resample-hz",
        type=int,
        default=None,
        help="Resample audio to this sample rate before transcription (e.g. 8000)",
    )
    args = parser.parse_args()

    asyncio.run(run_transcription(args))


if __name__ == "__main__":
    main()
