# MU-Bench: A Multilingual Transcription Benchmark from Real Phone Calls

**Andrea Li, Soham Ray** · April 2026

Representing work by Katie Echavia, Venu Satuluri, Ola Zytek, Victor Barres, Mindy Long, Nishita Jain, Nittai Malchin, Lydia Zarcone, Kelly Cooke

---

[Only about a quarter of the world speaks English.](https://sierra.ai/blog/multilingual-voice-agents) Yet it's the basis for most public automatic speech recognition (ASR) benchmarks — and the ones that cover other languages typically rely on read speech recorded in quiet studios. That leaves a huge gap in what you can measure before deploying a voice agent to handle real customer conversations in Turkish, Vietnamese, Mandarin, Spanish, or English.

To support voice across 70+ languages, Sierra uses a constellation of models as no single provider performs best across them all. When we started measuring ASR accuracy across them, we saw that transcription accuracy in Mandarin can be 5x worse than in English.

Without per-locale measurements, we can't choose the right model for each language, route traffic between providers, tune our pipeline for different conditions, or pinpoint where and why the agent is failing. So we built our own benchmark. Internally, we benchmark 79 locale variants across 42 languages and 13+ providers.

Today, we're open-sourcing a subset of that evaluation: five locales and five providers, totaling 4,270 human-annotated utterances from 250 real phone conversations — recorded at 8 kHz mono.

<!-- widget:globe -->

<!-- widget:coverage-matrix -->

## Why another benchmark?

Existing ASR benchmarks fall into two camps. Academic datasets like LibriSpeech, Common Voice, and FLEURS use clean, read speech in controlled recording conditions — useful for model development, but not representative of real conversations.

More recent benchmarks like Earnings-21, AA-WER v2.0, Voicegain 2025, and ConnexAI 2026 use more realistic audio, but they're English-only and measure nothing beyond word error rate (WER).

MU-Bench addresses all three gaps in existing public benchmarks: real phone conversations, multiple languages, and metrics that distinguish meaningful transcription errors from surface-level ones.

## The dataset

MU-Bench is hosted on HuggingFace as [`sierra-research/mu-bench`](https://huggingface.co/datasets/sierra-research/mu-bench). It comprises 250 conversations across five locales, yielding 4,270 caller utterances totaling 5.1 hours of audio.

### Source audio

The calls are scripted interactions in which participants called an AI banking agent built on Sierra's voice AI platform — checking a card order status, confirming a case tracking code, providing personal details (name, email, phone, address), disputing a transaction, and requesting a credit limit increase. Each call was conducted entirely in a single language.

Despite following a scripted scenario, callers used their own phones from their own environments (home, car, office, outdoors), producing the challenges found in real customer service conversations:

- **Background noise**: traffic, household sounds, wind, other voices.
- **Disfluencies**: "uh," "you know," pauses, corrections, and false starts.
- **Emotional variation**: frustration, hesitation, urgency.
- **Interruptions**: callers speaking over the agent.
- **Diverse speaking styles**: curt and impatient to chatty and rambling.

All calls were recorded at 8 kHz mono, with real-world noise and telephony compression artifacts.

### Annotation

Ground truth transcripts were produced by professional native-speaking annotators in two passes:

- **Transcription**: segment audio by speaker turn and produce a clean verbatim transcript.
- **Review**: a second annotator independently verifies and corrects the transcript.

The style guide specifies clean verbatim transcription: faithful to the speaker's words while omitting unnecessary fillers and false starts (e.g., "uhhh… I'm just wond—wondering why" → "I'm just wondering why"). Each locale follows language-specific conventions for numbers, spelling, and punctuation.

### Processing

Full call recordings are processed into the final dataset:

- **Utterance clipping**: segment audio into turns using annotator timestamps.
- **Agent filtering**: remove agent turns, retaining only caller speech.
- **Unintelligible filtering**: exclude segments flagged as unintelligible.

<!-- widget:audio-samples -->

### Dataset statistics

<!-- widget:dataset-stats -->

### Privacy

All personal details in the conversations — names, email addresses, phone numbers, addresses — are fictional, created for the scripted scenarios. No real personal information was used. Audio files are clipped to individual utterances and stripped of any metadata.

## How we measure

Not all transcription errors carry equal weight. "Um" versus "umm" has the same edit distance as "Mason" versus "Jason," but one is a trivial spelling variant while the other refers to a completely different person. A transcription that drops a filler word is functionally perfect; one that gets a single digit wrong in a phone number can render the utterance useless for downstream processing.

We report a new metric, Utterance Error Rate (UER), alongside the traditional Word Error Rate (WER) and latency. UER supplements WER by scoring on semantic meaning, while WER remains a token-level measure of transcription accuracy.

### LLM-based normalization

Before computing any metrics, transcripts must be normalized to a common surface form. This is traditionally done with deterministic rules such as lowercasing, punctuation stripping, and rule-based number expansion. While that approach works reasonably well for English, it breaks down in multilingual settings.

Take Chinese, where many characters are homophones and the audio alone is often insufficient to disambiguate them. When there is little semantic context — for example when someone states their name — it is unreasonable to expect an ASR system to recover the exact character sequence:

> **Actual:** <br>
> 我叫羽凡。 *(My name is Yǔfán.)* <br>
> <span class="pinyin">wǒ jiào yǔ fán.</span>
>
> **Predicted:** <br>
> 我叫宇凡。 *(My name is Yǔfán.)* <br>
> <span class="pinyin">wǒ jiào yǔ fán.</span>

At the same time, we cannot simply normalize all homophones indiscriminately. When enough semantic context is available, collapsing distinct homophones would erase genuine errors. Consider 买卖 (mǎi mài), meaning "business," and 买麦 (mǎi mài), meaning "buy wheat":

> **Sentence 1:** <br>
> 他做买卖很多年了。 *(He's been in business for many years.)* <br>
> <span class="pinyin">tā zuò mǎi mài hěn duō nián le.</span>
>
> **Sentence 2:** <br>
> 农民去市场买麦。 *(The farmer goes to the market to buy wheat.)* <br>
> <span class="pinyin">nóng mín qù shì chǎng mǎi mài.</span>

Failing to distinguish these cases would be a real transcription error, not a formatting difference.

We address these issues with LLM-based normalization. Given a reference transcript and predicted transcript, the model rewrites both to a consistent format so that variance between styling amongst providers is scored fairly. The normalizer handles cases that deterministic pipelines struggle with, including:

- Locale-specific number and date formats: digits are spelled in the locale's native language (e.g. "2" → "two" in en-US, "uno dos" in es-MX, "bir iki" in tr-TR, "một hai" in vi-VN, "一 二" in zh-CN), never cross-translated
- Abbreviation expansion (e.g., "Dr." → "doctor")
- Filler word removal (e.g., "um", "uh" → removed from both sides)
- Script normalization in Chinese, Japanese, and Korean (CJK) languages

This normalization is powered by GPT-4.1 with temperature 0. The full prompt templates are available in the [dataset repository](https://huggingface.co/datasets/sierra-research/mu-bench/blob/main/scoring/prompts.py).

### Error Scoring

#### Word Error Rate (WER) vs Utterance Error Rate (UER)

Word Error Rate is the standard metric used by most ASR benchmarks. It measures the minimum edit distance between reference and hypothesis at the word level — substitutions, deletions, and insertions divided by the number of reference words.

But WER treats all errors equally — a dropped "uh" counts the same as a misheard phone number digit. This makes it an unreliable signal for production use, where what matters is whether the meaning of an utterance was preserved. To address this, we introduce Utterance Error Rate, which isolates meaning-changing errors from surface-level ones.

After word alignment, each error (substitution, deletion, insertion) is scored by an LLM into one of three categories:

- **Significant**: a change in meaning, such as "C N 8 7 2" transcribed as "D N 8 7 2" in an account code, a wrong digit in a phone number, or a negation flipped.
- **Minor**: a real transcription difference, but meaning is preserved, such as "got" instead of "get."
- **No error**: the words differ on the surface but are semantically identical, such as "um" versus "umm," or "Dr." versus "doctor."

UER is the fraction of utterances containing at least one significant error.

<!-- widget:error-examples -->

#### Implementation notes

- WER is computed on LLM-normalized transcripts. For CJK locales (zh-CN in this set) we tokenize at the character level instead, since Chinese doesn't use whitespace — equivalent to character error rate (CER).
- Per-locale WER is corpus WER (sum of edits over sum of reference words). Overall WER is the unweighted mean of the five per-locale rates so each language counts equally; UER uses the same locale-macro aggregation.

### Latency

In a real conversation, the only latency that matters is how long the user waits before the assistant can act on what they said. We measure that as wall-clock time per utterance, from a single pinned client location at concurrency=1. The headline **Latency p95** column is time-to-complete-transcript: for batch providers, the full request-to-response round-trip; for streaming providers, the time from sending the request to receiving the final transcript. Defining it this way keeps batch and streaming submissions on the same axis.

Streaming has a second number worth knowing — TTFT, the time to the first partial transcript, which is what governs whether downstream systems can start reacting before the user finishes speaking. We surface it as a `+TTFT` annotation under the p95 cell, but we don't sort on it, since batch APIs have no analog.

## Providers evaluated

Initial results use batch (non-streaming) APIs — each utterance sent as a complete WAV file in a single HTTP request. No post-processing was applied to provider responses; the raw transcript text returned by each API was captured as-is.

| Provider | Model |
|----------|-------|
| Deepgram | Nova-3 |
| Google | Chirp-3 |
| Microsoft | Azure Speech |
| ElevenLabs | Scribe v2 |
| OpenAI | GPT-4o Mini Transcribe |

The benchmark accepts any transcription mode (batch, streaming, or real-time), and we welcome submissions from other providers and approaches.

## Results

Results are drawn from the current leaderboard as of April 2026.

<!-- widget:heatmap -->

<!-- widget:radar -->

**WER alone is misleading.** Deepgram Nova-3 has 5.4% WER on en-US with 5.8% UER — nearly every utterance with errors contains at least one meaning-changing one. Microsoft Azure has 4.2% WER but only 3.1% UER — most of its errors are surface-level. Raw WER cannot distinguish providers that make many harmless errors from those that make fewer but more consequential ones. This is why we introduced Utterance Error Rate.

**English is the strongest locale, but no language is fully solved.** All five providers achieve UER between 3–7% on en-US (Azure best at 3.1%, OpenAI weakest at 6.2%).

**Chinese remains the most challenging locale.** Mandarin sees UER between 29–55%, meaning a substantial-to-majority fraction of utterances have their meaning compromised. Vietnamese varies widely — Google Chirp-3 achieves 5.6% WER while Deepgram Nova-3 reaches 24.7%, driven heavily by Deepgram returning empty transcripts for 33% of Vietnamese utterances.

**Google Chirp-3 leads across all accuracy metrics** with the lowest overall WER (8.0%) and UER (14.1%). ElevenLabs Scribe v2 is second on UER (21.2%), with Azure third (24.6%). Azure achieves the lowest UER on en-US (3.1%) but struggles on tr-TR (30.7%) and vi-VN (33.9%). OpenAI GPT-4o Mini Transcribe comes in fourth (UER 27.3%), and Deepgram Nova-3 last (UER 32.6%) — Deepgram's headline number is dragged down by its Vietnamese (61% UER) and Chinese (55% UER) performance.

**Accuracy and speed do not correlate.** Deepgram Nova-3 has the best p50 latency (107ms) and p95 (376ms), while the slower providers — Google Chirp-3, OpenAI, and Azure — all sit above 1,000ms p95 (Google p50 833ms / p95 1,281ms; OpenAI p95 1,376ms; Azure p95 1,179ms). ElevenLabs Scribe v2 offers a strong balance: second-best UER (21.2%) with moderate latency (p50 415ms, p95 847ms). Azure's overall p95 is dominated by Turkish, where its p95 is 1,781ms — roughly 3–4× its other locales — so for non-Turkish workloads its tail is closer to ~600ms. The right choice depends on whether the deployment prioritizes accuracy or throughput.

## Statistical validity

Are these ranking differences real, or could they be artifacts of which conversations we happened to include? We answer this with paired bootstrap resampling at the **conversation level**: resample 250 conversations with replacement 10,000 times, and check how often the ranking between each provider pair flips. We resample conversations rather than individual utterances because utterances within the same conversation are correlated (same speaker, audio quality, and call conditions).

<!-- widget:significance -->

On UER, **all 10 pairwise differences are statistically significant (p < 0.001)** — the full ranking Google → ElevenLabs → Azure → OpenAI → Deepgram (best to worst) is robust under conversation-level resampling. On WER, 9 of 10 pairs are significant; the one exception is ElevenLabs vs Azure (p ≈ 0.12), which are statistically tied on WER even though they're cleanly separated on UER. Google's lead and Deepgram's last place are robust across both metrics.

We also ran the full pipeline end-to-end four times — fresh transcripts, fresh scoring — to check that the headline numbers don't depend on a lucky run. Standard deviations stay under 1 percentage point across providers and locales, and the ranking never flipped.

## Beyond accuracy: production considerations

MU-Bench measures transcription accuracy in isolation, but accuracy alone doesn't determine which provider is the best fit for production voice agents. In our deployments, several other factors shape the choice:

**Latency matters as much as accuracy.** A voice agent needs transcripts fast enough to respond naturally. Deepgram Nova-3's p50 of 107ms gives it nearly an 8× edge over Google Chirp-3's 833ms — a difference that directly affects conversation flow, even though Google leads on accuracy.

**Features like keyword boosting and noise cancellation shift rankings.** Deepgram excels at keyword boosting for domain-specific terms (product names, account codes, proper nouns), while Azure and Google struggle. Noise cancellation (e.g., Krisp) and sample rate both shift results — helping some providers while degrading others.

**Reliability and integration vary widely.** Rate limits, uptime SLAs, streaming API quality, and error handling all differ across providers. A provider with slightly worse accuracy but rock-solid reliability and low-latency streaming may be the better production choice.

In practice, we don't rely on a single provider. We combine multiple providers and techniques to push transcription quality beyond what any single API delivers out of the box. More on these techniques soon.

## What's next

We're expanding MU-Bench to more locales — we benchmark 42 languages internally and plan to open-source additional locales over time. We're also working on incorporating production-realistic conditions into the benchmark, including keyword boosting, noise cancellation, and latency-aware evaluation.

## Try it

- **Dataset**: [`sierra-research/mu-bench`](https://huggingface.co/datasets/sierra-research/mu-bench) on HuggingFace
- **Code & submissions**: [github.com/sierra-research/mu-bench](https://github.com/sierra-research/mu-bench)
- **Leaderboard**: [research.sierra.ai/mubench](https://research.sierra.ai/mubench)

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
