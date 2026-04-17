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

`latency.json` now uses a schema with a top-level `meta` block and a per-utterance `measurements` map. The shape depends on your API protocol:

**Batch (single request/response, e.g. a POST with the full wav file):**

```json
{
  "meta": {
    "protocol": "batch",
    "region": "us-east-1",
    "clientLocation": "aws:us-east-1",
    "measuredAt": "2026-04-17T12:00:00Z",
    "concurrency": 1
  },
  "measurements": {
    "en-US/conv-0-turn-0": {"roundTripMs": 269.1},
    "en-US/conv-0-turn-1": {"roundTripMs": 102.0}
  }
}
```

**Streaming (partial transcripts + final transcript):**

```json
{
  "meta": {
    "protocol": "streaming",
    "region": "us-east-1",
    "clientLocation": "aws:us-east-1",
    "measuredAt": "2026-04-17T12:00:00Z",
    "concurrency": 1
  },
  "measurements": {
    "en-US/conv-0-turn-0": {"ttftMs": 190.0, "completeMs": 780.0},
    "en-US/conv-0-turn-1": {"ttftMs": 155.0, "completeMs": 620.0}
  }
}
```

Measurement guidelines:
- **Wall-clock time** from when your client sends the request to when it receives the relevant response event. For batch that's the complete response; for streaming that's both the first partial (`ttftMs`) and the final transcript (`completeMs`).
- **One request at a time** (`concurrency: 1`). Bulk/parallel measurements inflate latency and aren't comparable across submissions.
- **Milliseconds, as a number** (int or float).
- **Single pinned region.** Set `meta.region` to the AWS-style region your client is running in. The validator accepts a small allowlist today (`us-east-1`, `us-east-2`, `us-west-1`, `us-west-2`, `eu-west-1`, `eu-central-1`, `ap-southeast-1`, `ap-northeast-1`). If you need a new region, ask a maintainer to add it in `scoring/validate.py`.
- **Keys must match exactly** the utterances you shipped `.txt` files for. Every `.txt` needs a matching `measurements` entry (the validator enforces this).
- Keys without a `<locale>/` prefix are rejected — same utterance IDs appear across locales so bare IDs are ambiguous.

**Legacy flat schema** (`{"en-US/conv-0-turn-0": 269.1, ...}`) is still accepted during the rollout window but warns; it is interpreted as `protocol=batch`, `region=unknown`. Migrate to the new schema when you next re-measure.

### `metadata.yaml` (required)

```yaml
model: Your-Model-Name
organization: Your Organization
version: "2026-04-16"
date: "2026-04-16"
contact: "optional@example.com"
notes: "Optional free-form notes. If config has any non-default value, add one 'override: <key> because ...' line per such key."
config:
  beamSize: default
  languageHint: default
  customVocabulary: default
  noiseSuppression: default
  domainAdaptation: default
  keywordBoosting: default
```

`model`, `organization`, `version`, `date`, and `config` are required. `contact` and `notes` are optional (but `notes` is required if any `config` value is not `default`; see below).

**About the `config` block.** This is a required disclosure of any inference-time configuration that affects the transcript — so that "Nova-3" with beam size 10 and 42 custom-vocab terms isn't compared to vanilla "Nova-3" without it. Each of the six keys must be either the literal string `default` (meaning you sent the provider's out-of-the-box value) or an explicit disclosure like `"10"` or `"enabled: 42 terms"`. If you declare a non-default value for key *X*, your `notes:` must include a line starting with `override: X` that explains the override in free text. The validator enforces both the schema and the override note. We recommend keeping everything at `default` so your submission is directly comparable to the provider's advertised capabilities.

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

Scoring requires an OpenAI key so you can't run the full pipeline in this repo locally — `scoring/validate.py` is the most comprehensive local check we support. The LLM prompt templates themselves are published alongside the audio on the HuggingFace dataset at [`sierra-research/mu-bench/blob/main/scoring/prompts.py`](https://huggingface.co/datasets/sierra-research/mu-bench/blob/main/scoring/prompts.py).

## How we measure

| Metric | Source | Notes |
|---|---|---|
| **WER** (Word Error Rate) | `scoring/score.py` after LLM-based normalization | Per-locale: total word edits divided by total reference words across the locale's utterances. Overall: unweighted mean of the five per-locale WERs. Silent clips with non-`<unintelligible>` empty gold count any hypothesis words as insertion errors (both numerator and denominator receive the hyp count) so hallucinations on silence show up in WER. |
| **UER** (Utterance Error Rate) | `scoring/score.py` | Per-locale: fraction of utterances with ≥1 meaning-changing error. Overall: unweighted mean of per-locale UERs (same locale-macro convention as WER). |
| **Latency p95** | `scripts/latency_stats.py` on your `latency.json` | Time-to-complete-transcript p95 (batch round-trip or streaming send-to-final). For streaming submissions, TTFT is also surfaced as a `+TTFT` annotation but is not used in the sort (batch has no analog). |

The scoring pipeline is pinned for reproducibility: every `scores.json` records the judge model snapshot, temperature, seed, and SHAs of the normalization / significant-errors prompts under a top-level `judge` block. `scripts/check_judge_drift.py` catches partial re-scoring batches where providers disagree on judge config.
