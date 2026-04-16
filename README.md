# MU-Bench

**M**ultilingual **U**tterances Transcription Benchmark — an open benchmark for evaluating speech-to-text providers across multiple locales and metrics.

## Overview

This benchmark compares transcription providers on real customer service phone conversations recorded at 8kHz mono. Human annotators produce word-level ground truth transcripts for each caller utterance.

The dataset covers 5 locales with 4,270 utterances total:

| Locale | Language | Utterances |
|--------|----------|------------|
| en-US | English (US) | 817 |
| es-MX | Spanish (Mexico) | 792 |
| tr-TR | Turkish | 846 |
| vi-VN | Vietnamese | 975 |
| zh-CN | Chinese (Mandarin) | 840 |

Live leaderboard: [research.sierra.ai/mubench](https://research.sierra.ai/mubench)

## For Submitters (Benchmarking Your Model)

If you want to evaluate your speech-to-text model on MU-Bench, read **[`submissions/SUBMITTING.md`](submissions/SUBMITTING.md)**. The short version:

1. Request access to the [HuggingFace dataset](https://huggingface.co/datasets/sierra-research/mu-bench) and download the audio.
2. Run your model, producing one `.txt` per utterance plus a `latency.json`.
3. Drop a directory under `submissions/raw/<your-model-name>/` and open a PR.
4. CI validates the format; a maintainer comments `/score` to run scoring; the leaderboard redeploys on merge.

### Local validation (before opening a PR)

```bash
pip install pyyaml              # or: pip install -e . from the repo root
python scoring/validate.py submissions/raw/<your-model-name> --manifest manifest.json
```

### Downloading the audio

```bash
export HF_TOKEN=your_token_here
pip install -e .[tools]
python scripts/download_audio.py
```

Audio files land in `audio/<locale>/`.

## Metrics

| Metric | Direction | Description |
|---|---|---|
| **WER** (Word Error Rate) | Lower is better | Percentage of words incorrectly transcribed after LLM normalization |
| **UER** (Utterance Error Rate) | Lower is better | Fraction of utterances containing at least one meaning-changing error |
| **Latency p95** | Lower is better | 95th percentile of per-request API response time (ms), measured at concurrency=1 |

The LLM prompt templates used for normalization and scoring are published alongside the audio on the HuggingFace dataset: [`sierra-research/mu-bench/blob/main/scoring/prompts.py`](https://huggingface.co/datasets/sierra-research/mu-bench/blob/main/scoring/prompts.py). See [`submissions/SUBMITTING.md`](submissions/SUBMITTING.md) for how each metric is computed.

## Repository Structure

```
manifest.json                # Audio list + ground truth transcripts
pyproject.toml               # Python deps (base, [tools], [scoring], [transcribe] extras)
submissions/
  SUBMITTING.md              # Submitter guide (read this first)
  raw/                       # One dir per provider: .txt transcripts + latency.json + metadata.yaml
  normalized/                # LLM-normalized transcripts (auto-generated on merge)
scoring/
  validate.py                # Submission format validation (run locally, also by CI)
  normalize.py               # LLM-based transcript normalization (CI only)
  score.py                   # WER / UER / quality computation (CI only)
  metrics.py                 # Core metric implementations
  update_leaderboard.py      # Regenerates results/leaderboard.json
scripts/
  download_audio.py          # Fetch audio from HuggingFace
  transcribe.py              # Example: run provider APIs end-to-end
  latency_stats.py           # Compute p50/p95 and patch scores.json
  compare_transcripts.py     # Diff two submission bases utterance-by-utterance
  significance_test.py       # Statistical significance between providers
results/
  leaderboard.json           # Aggregated leaderboard
  <provider>/scores.json     # Per-provider breakdown
  <provider>/details/        # Per-utterance scoring detail
leaderboard/                 # React/Vite web app for the public leaderboard (maintainer concern)
.github/workflows/           # CI for validation, scoring, and deployment
```

## License

This repository uses a dual-license model:

* **Code** (scripts, scoring, leaderboard, workflows, and all other software) is licensed under the [Apache License 2.0](LICENSE).
* **Data** (audio files and transcripts from Hugging Face and manifest.json) is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
