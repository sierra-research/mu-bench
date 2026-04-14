# MU-Bench: A Multilingual Transcription Benchmark from Real Phone Calls

**Andrea Li, Soham Ray** · April 2026

Representing work by Katie Echavia, Venu Satuluri, Ola Zytek, Victor Barres, Mindy Long, Nishita Jain, Nittai Malchin, Lydia Zarcone, Kelly Cooke

---

[Most of the world doesn't speak English.](https://sierra.ai/blog/multilingual-voice-agents) Most public ASR benchmarks focus on English, and the ones that cover other languages typically use read speech recorded in quiet studios. For a voice agent handling real customer calls across Turkish, Vietnamese, Mandarin, Spanish, and English, that leaves a gap in what you can actually measure before deploying.

At Sierra, we run a multilingual voice AI agent that speaks 70+ languages. A provider's transcription accuracy in Mandarin is not the same as in English -- the gap can be 3x or more. We can't route traffic to the right provider, tune our pipeline, or debug agent failures without per-locale measurements. So we measure.

Today, we're open-sourcing part of that evaluation: **5 locales** and **5 providers**, totaling **4,270 human-annotated utterances** from **250 real phone conversations** -- recorded at 24 kHz mono from real phone calls.

This is **MU-Bench** ([`sierra-research/mu-bench`](https://huggingface.co/datasets/sierra-research/mu-bench)) -- an open benchmark built from actual telephony audio, evaluated with metrics that go beyond raw WER.

<!-- widget:globe -->

<!-- widget:coverage-matrix -->

## Why another benchmark?

Existing ASR benchmarks fall into two camps. Academic datasets like LibriSpeech, Common Voice, and FLEURS use clean, read speech in controlled recording conditions -- useful for model development, but not representative of production telephony. More recent benchmarks like Earnings-21, AA-WER v2.0, Voicegain 2025, and ConnexAI 2026 use more realistic audio, but they're all English-only and measure nothing beyond WER.

No existing public benchmark simultaneously covers real phone conversations, multiple languages, and metrics that distinguish meaningful transcription errors from cosmetic ones. MU-Bench addresses all three.

## The dataset

MU-Bench is hosted on HuggingFace as [`sierra-research/mu-bench`](https://huggingface.co/datasets/sierra-research/mu-bench). It comprises 250 conversations across 5 locales, yielding 4,270 caller utterances totaling 5.1 hours of audio.

### Source audio

The calls are scripted recordings in which participants called an AI banking agent built on Sierra's voice AI platform. Callers followed a banking journey -- checking a card order status, confirming a case tracking code, providing personal details (name, email, phone, address), disputing a transaction, and requesting a credit limit increase. Each call was conducted entirely in a single language.

Despite following a scripted scenario, callers used their own phones from their own environments (home, car, office, outdoors), producing the natural acoustic variation found in real customer service traffic:

- **Background noise** -- traffic, household sounds, wind, other voices.
- **Disfluencies** -- "uh," "you know," pauses, corrections, and false starts.
- **Emotional variation** -- frustration, hesitation, urgency -- affecting pronunciation, speaking rate, and clarity.
- **Interruptions** at turn boundaries where the caller spoke over the agent.
- **Diverse speaking styles** -- curt and impatient to chatty and rambling.

All calls were recorded at 24 kHz mono from real phone calls -- lower fidelity than studio-recorded ASR benchmarks, with real-world noise and compression artifacts from production telephony.

### Annotation

Ground truth transcripts were produced by professional native-speaking annotators in two passes:

1. **Transcription pass.** An annotator segmented the audio by the human speaker's turns and produced a clean verbatim transcript for each segment.
2. **Review pass.** A second annotator independently reviewed every transcript against the audio, correcting discrepancies.

The style guide prescribed **clean verbatim** transcription: faithful to the speaker's words but omitting unnecessary hesitations, fillers, and false starts (e.g., "and uhhhh yeah I I I'm just wond- wondering why" becomes "And yeah, I'm just wondering why"). Each locale had language-specific formatting conventions for numbers, spelled sequences, and punctuation.

### Processing

The full call recordings were processed into the final dataset:

1. **Utterance clipping**: Audio segmented into individual turns using annotator-provided timestamps.
2. **Agent filtering**: AI agent turns discarded, retaining only human caller utterances.
3. **Unintelligible handling**: Segments flagged as unintelligible were excluded.

<!-- widget:audio-samples -->

### Dataset statistics

<!-- widget:dataset-stats -->

### Privacy

All personal details in the conversations -- names, email addresses, phone numbers, addresses -- are **fictional**, created for the scripted scenarios. No real personal information was used. Audio files are clipped to individual utterances and stripped of any metadata.

## How we measure

Not all transcription errors carry equal weight. "Um" versus "umm" has the same edit distance as "Mason" versus "Jason," but one is a trivial spelling variant while the other refers to a completely different person. A transcription that drops a filler word is functionally perfect; one that gets a single digit wrong in a phone number can render the utterance useless for downstream processing.

We use three complementary metrics: **Word Error Rate (WER)**, **Utterance Error Rate (UER)**, and **Quality Score**. All LLM-based scoring uses GPT-4.1 at temperature 0.

### LLM-based normalization

Before computing WER, predicted transcripts must be normalized to match the surface form of the ground truth. Traditional deterministic normalization -- lowercasing, punctuation stripping, and rule-based number expansion -- works reasonably well for English, but breaks down in multilingual settings. Variations in number formats, abbreviations, and punctuation conventions can introduce spurious mismatches and inflate error rates.

To address this, we use an LLM-based normalization step powered by GPT-4.1 (temperature 0). Given a (gold, predicted) pair, the model rewrites the prediction to align with the gold transcript's formatting conventions -- without correcting underlying transcription errors. This approach handles cases that deterministic pipelines struggle with, including:

- Locale-specific number and date formats (e.g., "ciento uno" → "101")
- Abbreviation expansion and contraction (e.g., "Dr." ↔ "doctor")
- Variants of disfluencies (e.g., "Ummm" → "umm")
- Script normalization in CJK languages

The last category is particularly important. In Chinese, many characters share identical pronunciations. For example, the names 羽凡, 宇凡, and 雨繁 are all pronounced "yǔ fán." When a speaker introduces themselves, the ASR system must choose a character sequence, but the audio signal alone is insufficient to disambiguate (imagine multiple distinct copies of the letter "A" -- a listener hearing the sound cannot know which one was intended).

Consider:

> **Actual:** <br>
> 我叫羽凡。 *(My name is Yǔfán.)* <br>
> <span class="pinyin">wǒ jiào yǔ fán.</span>
>
> **Predicted:** <br>
> 我叫宇凡。 *(My name is Yǔfán.)* <br>
> <span class="pinyin">wǒ jiào yǔ fán.</span>

A deterministic normalizer treats these as distinct strings and counts an error. An LLM, however, can infer that both are phonetically equivalent proper nouns and normalize the prediction to match the actual transcript.

Crucially, this normalization must remain selective. Not all homophones are interchangeable. Consider 买卖 (mǎi mài) versus 买麦 (mǎi mài):

> **Actual:** <br>
> 他做买卖很多年了。 *(He's been in business for many years.)* <br>
> <span class="pinyin">tā zuò mǎi mài hěn duō nián le.</span>
>
> **Predicted:** <br>
> 农民去市场买麦。 *(The farmer goes to the market to buy wheat.)* <br>
> <span class="pinyin">nóng mín qù shì chǎng mǎi mài.</span>

买卖 ("business") and 买麦 ("buy wheat") sound identical, but swapping one for the other is a genuine transcription error. An LLM can use context to distinguish these cases -- normalizing when ambiguity is inherent to the audio (e.g., proper nouns), and preserving differences when they reflect true semantic errors.

This distinction allows us to measure transcription quality more faithfully, reducing false positives while still penalizing errors that affect meaning or downstream usability.

### Word Error Rate (WER)

WER measures the minimum edit distance between reference and hypothesis transcripts at the word level -- substitutions, deletions, and insertions divided by the number of reference words. WER is computed on LLM-normalized transcripts. For CJK locales (zh-CN), we insert spaces around each character before alignment, converting to approximate character-level WER. Per-locale WER is the arithmetic mean of per-utterance values.

### Quality Score

Quality Score provides a holistic assessment of transcription quality on a **0--3 integer scale**, computed on **raw (un-normalized) transcripts** by an LLM (GPT-4.1, temperature 0):

| Score | Label | Description |
|-------|-------|-------------|
| 0 | Missing | Empty or no recognizable speech content |
| 1 | Poor | Largely incorrect, intent misunderstood, or wrong language |
| 2 | Acceptable | Core intent preserved, most words correct |
| 3 | Near-Perfect | Closely matches reference, ignoring trivial formatting |

### Utterance Error Rate (UER)

Standard WER treats all errors equally -- a dropped "uh" counts the same as a misheard phone number digit. Utterance Error Rate isolates **meaning-changing errors** from cosmetic ones.

After word alignment, each error (substitution, deletion, insertion) is scored by an LLM into one of three categories:

- **Significant** -- meaning changes. "Mason" transcribed as "Jason," a wrong digit in a phone number, or a negation flipped.
- **Minor** -- a real transcription difference, but meaning is preserved. "Got" instead of "get," or "gonna" versus "going to."
- **No error** -- the words differ on the surface but are semantically identical. "Um" versus "umm," "okay" versus "OK," or "Dr." versus "doctor."

UER is the **fraction of utterances containing at least one significant error**.

<!-- widget:error-examples -->

### Latency

We measure total round-trip API latency per utterance -- the wall-clock time from sending the request to receiving the complete response. We report both median (p50) and p95 (95th percentile) latency in milliseconds. p50 reflects typical performance, while p95 captures worst-case tail latency under production conditions. Measurements were taken using batch (non-streaming) APIs with sequential (non-concurrent) requests from a single location.

## Providers evaluated

Initial results use batch (non-streaming) APIs -- each utterance sent as a complete WAV file in a single HTTP request. No post-processing was applied to provider responses; the raw transcript text returned by each API was captured as-is.

| Provider | Model |
|----------|-------|
| Deepgram | Nova-3 |
| Google | Chirp-3 |
| Microsoft | Azure Speech |
| ElevenLabs | Scribe v2 |
| OpenAI | GPT-4o Transcribe |

The benchmark accepts any transcription mode (batch, streaming, or real-time), and we welcome submissions from other providers and approaches.

## Results

Results are drawn from the current leaderboard as of April 2026.

<!-- widget:heatmap -->

<!-- widget:radar -->

### Key findings

**WER alone is misleading.** Deepgram Nova-3 has 7.3% WER on en-US with 6.9% UER -- nearly all its errors change meaning. Microsoft Azure has 4.6% WER but only 3.3% UER -- most of its errors are cosmetic. Raw WER cannot distinguish providers that make many harmless errors from those that make fewer but more consequential ones. This is why we introduced Utterance Error Rate.

**English is the strongest locale, but no language is fully solved.** All five providers achieve UER between 3--7% on en-US. The gap narrows dramatically for structured inputs like names and tracking codes.

**Chinese remains the most challenging locale.** zh-CN sees UER between 21--37%, meaning a substantial fraction of utterances have their meaning compromised. Vietnamese varies widely -- Google Chirp-3 achieves 10.0% WER while Deepgram Nova-3 reaches 44.8%.

**Google Chirp-3 leads across all quality metrics** with the lowest overall WER (10.7%), UER (10.6%), and highest Quality Score (2.75). ElevenLabs Scribe v2 is second-best on UER (11.2%), while Azure achieves the lowest UER on en-US (3.3%) but struggles on vi-VN and zh-CN.

**Quality and speed do not correlate.** Deepgram Nova-3 has the best p50 latency (244ms) and p95 (690ms), while Google Chirp-3, the quality leader, has the highest p50 (1213ms) and p95 (2041ms). ElevenLabs Scribe v2 offers a good balance with strong quality (UER 11.2%) and moderate latency (p50 420ms, p95 832ms). The right choice depends on whether the deployment prioritizes accuracy or throughput.

## What's next

We're actively expanding MU-Bench along several axes:

- **More locales.** We have data collection underway for additional locales and plan to open-source them as annotation is completed.
- **Streaming evaluation.** The current benchmark evaluates batch transcription only. Streaming APIs produce different latency and quality characteristics that matter for real-time voice agents.
- **Open scoring toolkit.** We plan to release the full evaluation toolkit as a standalone package so the community can run scoring locally.
- **Community submissions.** The leaderboard is open to any provider. If you'd like to submit your model's results, see the [submission guide](https://github.com/sierra-research/mu-bench).

## Try it

- **Dataset**: [`sierra-research/mu-bench`](https://huggingface.co/datasets/sierra-research/mu-bench) on HuggingFace
- **Code & submissions**: [github.com/sierra-research/mu-bench](https://github.com/sierra-research/mu-bench)
- **Leaderboard**: [research.sierra.ai](https://research.sierra.ai)

---

## References

1. V. Panayotov, G. Chen, D. Povey, S. Khudanpur. "LibriSpeech: An ASR corpus based on public domain audio books." *ICASSP*, 2015.
2. R. Ardila et al. "Common Voice: A massively-multilingual speech corpus." *LREC*, 2020.
3. A. Conneau et al. "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech." *IEEE SLT*, 2023.
4. S. H. Del Rio et al. "Earnings-21: A practical benchmark for ASR in the wild." *Interspeech*, 2021.
5. HuggingFace. "Open ASR Leaderboard." [huggingface.co/spaces/hf-audio/open_asr_leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), 2024.
6. Artificial Analysis. "AA-WER v2.0: Speech-to-Text Benchmark." [artificialanalysis.ai/speech-to-text](https://artificialanalysis.ai/speech-to-text), 2026.
7. Voicegain. "STT Benchmark 2025." [voicegain.ai/benchmark](https://www.voicegain.ai/post/stt-benchmark-2025-q1-telephony-en-us), 2025.
8. ConnexAI. "ASR Benchmark 2026." [connexai.com/asr-benchmark](https://www.connexai.com/blog/asr-benchmark-2026), 2026.
