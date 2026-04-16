# Submitting to MU-Bench

This guide is for anyone who wants to benchmark their speech-to-text model on MU-Bench. You'll create a directory under `submissions/raw/`, open a PR, and a maintainer will run automated scoring.

## What you need to ship

```
submissions/raw/<your-model-name>/
  metadata.yaml            # required
  latency.json             # required — per-utterance API response time in ms
  en-US/
    conv-0-turn-0.txt      # one .txt per utterance in the manifest
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

**Naming:** `<your-model-name>` should be a short kebab-case identifier, e.g. `acme-stt-v2`. The same name appears as the directory under `submissions/raw/` and later under `results/`.

## File-by-file contract

### Transcripts (`<locale>/<utterance_id>.txt`)

- **One `.txt` file per utterance** you want scored. The filename is the utterance ID from `manifest.json` (e.g. `conv-0-turn-0.txt`).
- **Content is the raw transcript** — nothing else. No JSON, no speaker tags, no timing. UTF-8.
- **Empty files are allowed** for silence or inaudible speech. They'll be scored as empty transcripts.
- **UTF-8 only, under 10 KB each, under 50 MB total.**

### `latency.json` (required)

A flat JSON object mapping `"<locale>/<utterance_id>"` to per-request API latency in milliseconds:

```json
{
  "en-US/conv-0-turn-0": 269.1,
  "en-US/conv-0-turn-1": 102.0,
  "es-MX/conv-0-turn-0": 318.4
}
```

Measurement guidelines:
- **Wall-clock time** from when your client sends the request to when it receives the complete response.
- **One request at a time** (concurrency = 1). Bulk/parallel measurements inflate latency and aren't comparable across submissions.
- **Milliseconds, as a number** (int or float).
- **Keys must match exactly** the utterances you shipped `.txt` files for. Every `.txt` needs a matching latency entry (the validator enforces this).
- Keys without a `<locale>/` prefix are rejected — same utterance IDs appear across locales so bare IDs are ambiguous.

### `metadata.yaml` (required)

```yaml
model: Your-Model-Name
organization: Your Organization
version: "2026-04-16"
date: "2026-04-16"
contact: "optional@example.com"
notes: "Optional free-form notes about config, beam size, decoding params, etc."
```

`model`, `organization`, `version`, `date` are required. `contact` and `notes` are optional. See `submissions/raw/EXAMPLE.md` for a copy-pasteable example.

## Partial submissions

You don't have to cover all 5 locales. If you only want to benchmark `en-US`, ship only the `en-US/` directory and only en-US keys in `latency.json`. Skipped locales simply won't be scored. **Within each locale you do include, every utterance in the manifest must have a `.txt` file and a matching `latency.json` entry.**

## Before you open a PR: validate locally

```bash
pip install pyyaml           # or: pip install -e . from the repo root
python scoring/validate.py submissions/raw/<your-model-name> --manifest manifest.json
```

The validator checks directory structure, file encoding, metadata fields, latency key format, and coverage. Fix any reported issues before pushing — CI runs the same validator, and will block the PR with the same messages if it fails.

## What happens on your PR

1. A validation script runs immediately on any PR that touches `submissions/raw/**` checking for missing files. It posts a comment with the validator output and passes or blocks the PR on the result.
2. **A maintainer will review and comment `/score`** to run the scoring pipeline (WER, Utterance Error Rate, Latency p95). Results are posted back as a PR comment.
3. **On merge**, `post-merge-score.yml` re-scores from scratch, commits `results/<your-model-name>/` and `submissions/normalized/<your-model-name>/` to main, and the leaderboard redeploys with your model added.

Scoring requires an OpenAI key (for the LLM-based normalization pass), so you can't run the full pipeline in this repo locally — `scoring/validate.py` is the most comprehensive local check we support. The LLM prompt templates themselves are fully public: they're published alongside the audio on the HuggingFace dataset at [`sierra-research/mu-bench/blob/main/scoring/prompts.py`](https://huggingface.co/datasets/sierra-research/mu-bench/blob/main/scoring/prompts.py), so you can reproduce our scoring end-to-end with your own OpenAI key if you want.

## How we measure

| Metric | Source | Notes |
|---|---|---|
| **WER** (Word Error Rate) | `scoring/score.py` after LLM-based normalization | Percentage of words incorrectly transcribed |
| **UER** (Utterance Error Rate / Significant WER) | `scoring/score.py` | Fraction of utterances with ≥1 meaning-changing error |
| **Latency p50 / p95** | `scripts/latency_stats.py` on your `latency.json` | Per-locale and overall percentiles |
