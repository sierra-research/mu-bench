# Voice Transcription Benchmark

An open benchmark for evaluating speech-to-text providers across multiple locales and metrics.

## Overview

This benchmark compares transcription providers on real customer service phone conversations recorded at 24kHz mono. Human annotators produce word-level ground truth transcripts for each caller utterance.

The dataset covers 5 locales with 4,270 utterances total:

| Locale | Language | Utterances |
|--------|----------|------------|
| en-US | English (US) | 817 |
| es-MX | Spanish (Mexico) | 792 |
| tr-TR | Turkish | 846 |
| vi-VN | Vietnamese | 975 |
| zh-CN | Chinese (Mandarin) | 840 |

## Getting Started

### 1. Install System Dependencies

```bash
brew install ffmpeg   # macOS
# or: apt-get install ffmpeg   # Linux
```

### 2. Download Audio

The audio is hosted as a gated dataset on HuggingFace. You'll need to:

1. [Request access](https://huggingface.co/datasets/sierra-research/mu-bench) to the `sierra-research/mu-bench` dataset
2. Create a [HuggingFace token](https://huggingface.co/settings/tokens) and set it as an environment variable

```bash
export HF_TOKEN=your_token_here
pip install -r scripts/requirements.txt
python scripts/download_audio.py
```

This exports `.wav` files to `audio/<locale>/`.

### 3. View the Leaderboard Locally

```bash
cd leaderboard
npm install
npm run dev
```

### 4. Explore the Manifest

`manifest.json` contains every utterance in the benchmark with its ground truth transcript:

```json
{
  "id": "conv-0-turn-0",
  "locale": "en-US",
  "conversation_id": "conv-0",
  "turn_index": 0,
  "transcript": "Hi. I'm calling to check the status of my card.",
  "audio_path": "audio/en-US/conv-0-turn-0.wav",
  "duration_sec": 2.246
}
```

## Submitting Results

### Submission Format

Create a directory under `submissions/raw/<your-model-name>/` with the following structure:

```
submissions/raw/your-model-name/
  metadata.yaml
  latency.json              # Optional — per-utterance API latency
  en-US/
    conv-0-turn-0.txt
    conv-0-turn-1.txt
    ...
  es-MX/
    ...
  tr-TR/
    ...
  vi-VN/
    ...
  zh-CN/
    ...
```

**Transcript files:** One `.txt` file per utterance containing only the transcript text. File names must match the utterance IDs in `manifest.json` (e.g., `conv-0-turn-0.txt`).

**latency.json (optional):** Per-utterance API response time in milliseconds, measured as wall-clock time from request send to response received:

```json
{
  "conv-0-turn-0": 234.5,
  "conv-0-turn-1": 187.2
}
```

If included, p50 and p95 latency statistics will be computed and displayed on the leaderboard.

**metadata.yaml:** Include basic model information:

```yaml
model: Your-Model-Name
organization: Your Organization
version: "1.0"
date: "2026-04-01"
contact: your-email@example.com
notes: ""
```

### Required Fields

| Field | Description |
|-------|-------------|
| `model` | Name of the transcription model |
| `organization` | Company or team name |
| `version` | Model version string |
| `date` | Submission date (YYYY-MM-DD) |
| `contact` | Email for questions (optional) |
| `notes` | Any additional context (optional) |

### Step-by-Step

1. **Download the audio** from the HuggingFace dataset (or use `scripts/download_audio.py`)
2. **Run your model** on all audio files listed in `manifest.json`
3. **Save transcripts** as `.txt` files — one per utterance, plain text, no headers
4. **Create `metadata.yaml`** with your model info
5. **(Optional) Create `latency.json`** mapping utterance IDs to API response times in milliseconds
6. **Open a pull request** to this repo adding your submission under `submissions/raw/`

You do not need to submit transcripts for every locale. Missing locales will be noted but the submission will still be accepted. Missing utterances within a locale will be scored as full errors.

### Validation

You can validate your submission locally before opening a PR:

```bash
pip install -r scoring/requirements.txt
python scoring/validate.py submissions/raw/your-model-name --manifest manifest.json
```

When you open a PR, CI runs the same check automatically. Validation verifies:
- `metadata.yaml` is present with required fields
- All `.txt` file names match utterance IDs in the manifest
- Warns about missing utterances (scored as full errors)
- Warns about empty transcript files

### Scoring

Scoring is run by maintainers — submitters do not need to run it locally. When a maintainer comments `/score` on your PR, the CI pipeline runs automatically:

1. **Normalization** — Your raw transcripts are LLM-normalized toward the ground truth format (handling differences in number formatting, punctuation, contractions, etc.)
2. **Scoring** — Four metrics are computed:

| Metric | Description | Direction |
|--------|-------------|-----------|
| **WER** | Word Error Rate after LLM normalization | Lower is better |
| **Significant WER** | Rate of semantically significant word errors | Lower is better |
| **Quality Score** | LLM-judged quality on a 0-3 scale (computed on raw transcripts) | Higher is better |
| **Latency (p95)** | 95th percentile API response time per utterance (ms) | Lower is better |

Results are posted as a comment on your PR and added to the leaderboard on merge. The scoring prompts and API keys are stored as GitHub secrets and are not available outside CI.

See `submissions/raw/deepgram-nova3/` for a complete example submission.

## Repository Structure

```
manifest.json              # Audio file list + ground truth transcripts
packages.txt               # System-level dependencies (ffmpeg)
scripts/
  download_audio.py        # Download audio from HuggingFace
  transcribe.py            # Runs provider APIs to generate transcripts + latency
  compare_transcripts.py   # Compare transcripts between two submission bases
  latency_stats.py         # Compute p50/p95 latency from latency.json files
  requirements.txt         # Python dependencies
submissions/
  raw/                     # Raw transcripts + latency.json, one directory per provider
  normalized/              # LLM-normalized transcripts (auto-generated)
scoring/
  validate.py              # Submission format validation
  normalize.py             # LLM-based transcript normalization
  score.py                 # Metrics computation (WER, quality, sig. WER)
  metrics.py               # Core metric implementations
  prompts.py               # LLM prompt templates
results/
  leaderboard.json         # Aggregated leaderboard data
  <provider>/scores.json   # Per-provider score breakdowns (includes latency)
leaderboard/               # React/Vite leaderboard web app
.github/workflows/         # CI for validation, scoring, and deployment
```

## License

This repository uses a dual-license model:

* **Code** (scripts, scoring, leaderboard, workflows, and all other software) is licensed under the [Apache License 2.0](LICENSE).
* **Data** (audio files and transcripts from Hugging Face and manifest.json) is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
