"""Core metrics implementation: WER and Significant WER.

Self-contained module adapted from the benchmark evaluation pipeline.
Uses LLM-based normalization via OpenAI for fair transcript comparison.

WER is reported as **corpus WER**: per utterance we record the count of
edit operations (substitutions + deletions + insertions) and the
reference word count. Per-locale WER is the ratio of summed edits to
summed reference words across the locale's utterances; overall WER is
the unweighted mean of per-locale corpus WERs (computed in scoring.score).

Silent clips with a non-``<unintelligible>`` empty gold contribute any
hypothesis words as pure insertion errors. Both numerator and denominator
receive the hyp word count so silence hallucinations show up in corpus WER
directly rather than being silently dropped.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from jiwer import process_words

from scoring.llm import (
    NORMALIZE_SCHEMA,
    SIGNIFICANT_WER_SCHEMA,
    get_responses,
    load_responses,
)


def _load_prompts():
    """Lazy-load scoring prompts.

    Returns ``(NORMALIZE_PRED_AGAINST_GOLD_PROMPT, SIGNIFICANT_WORD_ERRORS_PROMPT)``.
    Kept lazy so modules that don't touch the LLM (validator, leaderboard
    update) can import ``scoring.metrics`` without the secret prompts file.
    """
    from scoring.prompts import (  # type: ignore[attr-defined]
        NORMALIZE_PRED_AGAINST_GOLD_PROMPT,
        SIGNIFICANT_WORD_ERRORS_PROMPT,
    )

    return NORMALIZE_PRED_AGAINST_GOLD_PROMPT, SIGNIFICANT_WORD_ERRORS_PROMPT


def _load_gold_prompt() -> str:
    """Return the canonical gold-only normalization prompt."""
    from scoring.prompts import NORMALIZE_GOLD_PROMPT  # type: ignore[attr-defined]

    return NORMALIZE_GOLD_PROMPT


@dataclass
class TranscriptRow:
    """A paired ground truth + predicted transcript for scoring."""

    locale: str
    utterance_id: str
    gold: str
    predicted: str


@dataclass
class WERResult:
    """Result from WER computation.

    `wer` is the per-utterance rate (edits / ref_words), preserved for
    distribution plots and back-compat. Corpus aggregation should use
    `edits` and `ref_words` directly.
    """

    locale: str
    utterance_id: str
    gold: str
    predicted: str
    normalized_gold: Optional[str] = None
    normalized_predicted: Optional[str] = None
    wer: Optional[float] = None
    edits: Optional[int] = None
    ref_words: Optional[int] = None


@dataclass
class SignificantWERResult:
    """Result from significant WER computation."""

    locale: str
    utterance_id: str
    gold: str
    predicted: str
    normalized_gold: Optional[str] = None
    normalized_predicted: Optional[str] = None
    major_errors_count: Optional[int] = None
    total_words_count: Optional[int] = None
    all_errors_with_scores: Optional[List[Dict[str, Any]]] = None


def is_unintelligible(text: str) -> bool:
    return "<unintelligible>" in text.strip().lower()


_CJK_RANGES = [
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x3000, 0x303F),  # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),  # Fullwidth Forms
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0xAC00, 0xD7AF),  # Hangul Syllables
]


def _is_cjk_char(c: str) -> bool:
    cp = ord(c)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def tokenize_for_alignment(text: str, locale: str) -> str:
    """Character-level tokenization for CJK locales.

    CJK languages don't use spaces between words, so jiwer's word-level
    alignment treats entire sentences as single tokens. This inserts spaces
    between each CJK character so jiwer can align at character level.
    """
    if not locale.startswith(("zh", "ja", "ko")):
        return text
    result = []
    for char in text:
        if _is_cjk_char(char):
            result.append(f" {char} ")
        else:
            result.append(char)
    return " ".join("".join(result).split())


def normalize_for_match(text: str) -> str:
    """Simple normalization: lowercase, remove punctuation/spaces.

    Used for legacy string-equality matching; not appropriate for word-level
    WER because it collapses whitespace.
    """
    if text is None:
        return ""
    t = str(text).lower()
    for ch in [",", ".", "-", "_", "/", "\\", "(", ")", "[", "]", ":", ";", " "]:
        t = t.replace(ch, "")
    return t


def normalize_for_simple_wer(text: str) -> str:
    """Whitespace-preserving simple normalization for WER.

    Lowercases and strips punctuation but keeps spaces so that the result
    has meaningful word boundaries for jiwer's word-level alignment and
    for the corpus-WER reference word count.
    """
    if text is None:
        return ""
    t = str(text).lower()
    for ch in [",", ".", "-", "_", "/", "\\", "(", ")", "[", "]", ":", ";"]:
        t = t.replace(ch, " ")
    return " ".join(t.split())


def normalize_transcript_pairs(
    rows: List[TranscriptRow],
    num_workers: int = 1,
    normalization_prompt: str | None = None,
) -> Dict[int, Optional[Tuple[str, str]]]:
    """Normalize predicted transcripts toward gold format using LLM.

    Returns dict mapping row index to (gold, normalized_predicted),
    or None if normalization failed.
    """
    if normalization_prompt is None:
        normalization_prompt, _ = _load_prompts()
    normalization_pairs: List[Tuple[int, str]] = []

    for row_idx, r in enumerate(rows):
        if not r.gold and not r.predicted:
            continue
        normalization_pairs.append(
            (
                row_idx,
                normalization_prompt.format(expected_transcript=r.gold, actual_transcript=r.predicted),
            )
        )

    row_idx_to_normalized: Dict[int, Optional[Tuple[str, str]]] = {}

    if normalization_pairs:
        chunk_size = max(1, min(num_workers, len(normalization_pairs)))
        for i in range(0, len(normalization_pairs), chunk_size):
            chunk = normalization_pairs[i : i + chunk_size]
            prompts = [p for _, p in chunk]
            responses = get_responses(prompts, num_workers=num_workers, response_format=NORMALIZE_SCHEMA)
            loaded = load_responses(responses)

            for j, (row_idx, _) in enumerate(chunk):
                if j < len(loaded) and loaded[j] is not None:
                    resp = loaded[j]
                    if isinstance(resp, dict):
                        normalized_pred = resp.get("normalized_actual")
                        if normalized_pred is not None:
                            row_idx_to_normalized[row_idx] = (
                                rows[row_idx].gold,
                                normalized_pred,
                            )
                        else:
                            row_idx_to_normalized[row_idx] = None
                    else:
                        row_idx_to_normalized[row_idx] = None
                else:
                    row_idx_to_normalized[row_idx] = None

    if row_idx_to_normalized:
        none_count = sum(1 for v in row_idx_to_normalized.values() if v is None)
        if none_count > 0:
            total = len(row_idx_to_normalized)
            print(f"Normalization: {total - none_count}/{total} successful, {none_count} errors (using original text)")

    return row_idx_to_normalized


def _wer_components(tok_gold: str, tok_pred: str) -> Tuple[int, int]:
    """Return (edits, ref_words) for a tokenized gold/hyp pair.

    edits = substitutions + deletions + insertions, the numerator of WER.
    ref_words = number of tokens in the reference, the denominator.
    """
    res = process_words(tok_gold, tok_pred)
    edits = res.substitutions + res.deletions + res.insertions
    ref_words = len(tok_gold.split())
    return edits, ref_words


def _silence_hallucination_components(tok_pred: str) -> Tuple[int, int]:
    """Return (edits, ref_words) when the gold is empty (silent clip).

    Every hypothesis word is a pure insertion, so both the numerator
    (edits) and denominator (ref_words) get the hyp word count. This
    folds silence hallucinations into corpus WER instead of dropping them.

    Returns (0, 0) for a truly silent pair (both gold and pred empty);
    `compute_wer` maps that to a no-op contribution.
    """
    hyp_words = len(tok_pred.split()) if tok_pred else 0
    return hyp_words, hyp_words


def compute_wer(rows: List[TranscriptRow]) -> List[WERResult]:
    """Compute per-utterance WER components on pre-normalized transcripts.

    Records per-utterance edits and reference word count so that callers can
    aggregate corpus WER (sum(edits) / sum(ref_words)) per locale.

    Expects rows with gold and predicted already normalized (via
    scoring.normalize). Uses character-level tokenization for CJK locales,
    so CJK WER is effectively character-level.

    Silent-clip handling (item 6): when ``r.gold`` is empty but not
    ``<unintelligible>``, every hyp word contributes as a pure insertion
    to both the numerator and denominator; the per-utterance ``wer`` stays
    ``None`` for silent/silent pairs (``0/0``) to avoid corrupting
    distribution plots, but corpus sums are unaffected because we add 0/0.
    """
    results = []
    for r in rows:
        if is_unintelligible(r.gold):
            w, edits, ref_words = None, None, None
        elif not r.gold:
            tok_pred = tokenize_for_alignment(r.predicted, r.locale)
            edits, ref_words = _silence_hallucination_components(tok_pred)
            w = 1.0 if edits > 0 else None
        else:
            tok_gold = tokenize_for_alignment(r.gold, r.locale)
            tok_pred = tokenize_for_alignment(r.predicted, r.locale)
            edits, ref_words = _wer_components(tok_gold, tok_pred)
            w = (edits / ref_words) if ref_words > 0 else None

        results.append(
            WERResult(
                locale=r.locale,
                utterance_id=r.utterance_id,
                gold=r.gold,
                predicted=r.predicted,
                normalized_gold=r.gold,
                normalized_predicted=r.predicted,
                wer=w,
                edits=edits,
                ref_words=ref_words,
            )
        )

    return results


def compute_simple_wer(rows: List[TranscriptRow]) -> List[WERResult]:
    """Compute WER with simple normalization (no LLM calls needed).

    Uses `normalize_for_simple_wer` so word boundaries are preserved and
    both the per-utterance rate and corpus components are meaningful.

    Silent-clip handling matches ``compute_wer`` — see that docstring.
    """
    results = []
    for r in rows:
        if is_unintelligible(r.gold):
            w, edits, ref_words = None, None, None
            norm_g, norm_p = None, None
        else:
            norm_g = normalize_for_simple_wer(r.gold)
            norm_p = normalize_for_simple_wer(r.predicted)
            if not norm_g:
                # Silent (non-unintelligible) gold — count hyp words as insertions.
                edits, ref_words = _silence_hallucination_components(norm_p)
                w = 1.0 if edits > 0 else None
            else:
                edits, ref_words = _wer_components(norm_g, norm_p)
                w = (edits / ref_words) if ref_words > 0 else None

        results.append(
            WERResult(
                locale=r.locale,
                utterance_id=r.utterance_id,
                gold=r.gold,
                predicted=r.predicted,
                normalized_gold=norm_g,
                normalized_predicted=norm_p,
                wer=w,
                edits=edits,
                ref_words=ref_words,
            )
        )
    return results


def compute_significant_wer(
    rows: List[TranscriptRow],
    num_workers: int = 8,
) -> List[SignificantWERResult]:
    """Compute Significant WER — rate of semantically significant word errors.

    Expects rows with gold and predicted already normalized (via scoring.normalize).
    Word-level errors are found via jiwer alignment, then each error is scored
    by an LLM as significant (1), minor (2), or none (3).
    """
    _, SIGNIFICANT_WORD_ERRORS_PROMPT = _load_prompts()
    # Find word-level errors and prepare scoring prompts
    error_scoring_prompts: List[Tuple[int, str, Dict[int, Dict[str, Any]]]] = []
    row_idx_to_result: Dict[int, SignificantWERResult] = {}

    for row_idx, r in enumerate(rows):
        if not r.gold:
            row_idx_to_result[row_idx] = SignificantWERResult(
                locale=r.locale,
                utterance_id=r.utterance_id,
                gold=r.gold,
                predicted=r.predicted,
                major_errors_count=0,
                total_words_count=0,
            )
            continue

        norm_g = tokenize_for_alignment(r.gold, r.locale)
        norm_p = tokenize_for_alignment(r.predicted, r.locale)

        result = process_words(norm_g, norm_p)
        ref_words = norm_g.split()
        hyp_words = norm_p.split()

        errors = []
        if result.alignments and len(result.alignments) > 0:
            for chunk in result.alignments[0]:
                if chunk.type == "substitute":
                    ref_word = ref_words[chunk.ref_start_idx] if chunk.ref_start_idx < len(ref_words) else ""
                    hyp_word = hyp_words[chunk.hyp_start_idx] if chunk.hyp_start_idx < len(hyp_words) else ""
                    errors.append(
                        {
                            "error": f"Substitution: '{ref_word}' to '{hyp_word}' at position {chunk.ref_start_idx}",
                            "type": "substitution",
                            "position": chunk.ref_start_idx,
                            "truth_word": ref_word,
                            "hyp_word": hyp_word,
                        }
                    )
                elif chunk.type == "delete":
                    ref_word = ref_words[chunk.ref_start_idx] if chunk.ref_start_idx < len(ref_words) else ""
                    errors.append(
                        {
                            "error": f"Deletion: '{ref_word}' at position {chunk.ref_start_idx}",
                            "type": "deletion",
                            "position": chunk.ref_start_idx,
                            "truth_word": ref_word,
                        }
                    )
                elif chunk.type == "insert":
                    hyp_word = hyp_words[chunk.hyp_start_idx] if chunk.hyp_start_idx < len(hyp_words) else ""
                    errors.append(
                        {
                            "error": f"Insertion: '{hyp_word}' at position {chunk.ref_start_idx}",
                            "type": "insertion",
                            "position": chunk.ref_start_idx,
                            "hyp_word": hyp_word,
                        }
                    )

        if errors:
            position_to_error = {err["position"]: err for err in errors}
            error_descriptions = []
            for err in errors:
                error_descriptions.append(
                    f'    {{\n      "error": "{err["error"]}",\n      "reason": "<TO_BE_DETERMINED>",\n      "score": <TO_BE_DETERMINED>\n    }}'
                )
            errors_str = ",\n".join(error_descriptions)
            prompt = SIGNIFICANT_WORD_ERRORS_PROMPT.format(
                expected_transcript=norm_g, actual_transcript=norm_p, errors=errors_str
            )
            error_scoring_prompts.append((row_idx, prompt, position_to_error))
        else:
            total_words = len(norm_g.split()) if norm_g else 0
            row_idx_to_result[row_idx] = SignificantWERResult(
                locale=r.locale,
                utterance_id=r.utterance_id,
                gold=r.gold,
                predicted=r.predicted,
                normalized_gold=norm_g,
                normalized_predicted=norm_p,
                major_errors_count=0,
                total_words_count=total_words,
                all_errors_with_scores=[],
            )

    # Score errors via LLM
    row_idx_to_scored_errors: Dict[int, List[Dict[str, Any]]] = {}

    if error_scoring_prompts:
        print(
            f"Significant WER scoring: {len(error_scoring_prompts)} utterances have "
            f"alignment errors and need LLM scoring (chunk_size={min(num_workers, len(error_scoring_prompts))})",
            flush=True,
        )
        chunk_size = max(1, min(num_workers, len(error_scoring_prompts)))
        for i in range(0, len(error_scoring_prompts), chunk_size):
            chunk = error_scoring_prompts[i : i + chunk_size]
            prompts = [p for _, p, _ in chunk]
            responses = get_responses(prompts, num_workers=num_workers, response_format=SIGNIFICANT_WER_SCHEMA)
            loaded = load_responses(responses)

            for j, (row_idx, _, position_to_error) in enumerate(chunk):
                if j < len(loaded) and loaded[j] is not None:
                    resp = loaded[j]
                    if isinstance(resp, dict) and "scores" in resp:
                        scored_errors_from_llm = resp["scores"]
                        scored_errors_list = []

                        for scored_err in scored_errors_from_llm:
                            if isinstance(scored_err, dict) and "error" in scored_err:
                                error_str = scored_err["error"]
                                pos_match = re.search(r"at position (\d+)", error_str)
                                if pos_match:
                                    pos = int(pos_match.group(1))
                                    if pos in position_to_error:
                                        err_copy = position_to_error[pos].copy()
                                        err_copy["score"] = scored_err.get("score", 2)
                                        err_copy["reason"] = scored_err.get("reason", "")
                                        scored_errors_list.append(err_copy)
                                        continue

                        # Fallback to order if position matching failed
                        if len(scored_errors_list) != len(position_to_error):
                            scored_errors_list = []
                            positions = sorted(position_to_error.keys())
                            for idx, pos in enumerate(positions):
                                err_copy = position_to_error[pos].copy()
                                if idx < len(scored_errors_from_llm) and isinstance(scored_errors_from_llm[idx], dict):
                                    err_copy["score"] = scored_errors_from_llm[idx].get("score", 2)
                                    err_copy["reason"] = scored_errors_from_llm[idx].get("reason", "")
                                else:
                                    err_copy["score"] = 2
                                    err_copy["reason"] = "infra_error"
                                scored_errors_list.append(err_copy)

                        row_idx_to_scored_errors[row_idx] = scored_errors_list
                    else:
                        row_idx_to_scored_errors[row_idx] = None
                else:
                    row_idx_to_scored_errors[row_idx] = None

            print(
                f"Significant WER scoring: {min(i + chunk_size, len(error_scoring_prompts))}/{len(error_scoring_prompts)} processed",
                flush=True,
            )

    # Build final results for rows with errors
    for row_idx, _, position_to_error in error_scoring_prompts:
        r = rows[row_idx]
        norm_g = tokenize_for_alignment(r.gold, r.locale)
        norm_p = tokenize_for_alignment(r.predicted, r.locale)

        scored_errors = row_idx_to_scored_errors.get(row_idx)
        if scored_errors is None:
            row_idx_to_result[row_idx] = SignificantWERResult(
                locale=r.locale,
                utterance_id=r.utterance_id,
                gold=r.gold,
                predicted=r.predicted,
                normalized_gold=norm_g,
                normalized_predicted=norm_p,
                major_errors_count=None,
                total_words_count=None,
                all_errors_with_scores=None,
            )
            continue

        total_words = len(norm_g.split()) if norm_g else 0
        all_errors_with_scores = []
        major_errors_count = 0
        for err in scored_errors:
            all_errors_with_scores.append(
                {
                    "error": err.get("error", ""),
                    "reason": err.get("reason", ""),
                    "score": err.get("score", 2),
                }
            )
            if err.get("score") == 1:
                major_errors_count += 1

        row_idx_to_result[row_idx] = SignificantWERResult(
            locale=r.locale,
            utterance_id=r.utterance_id,
            gold=r.gold,
            predicted=r.predicted,
            normalized_gold=norm_g,
            normalized_predicted=norm_p,
            major_errors_count=major_errors_count,
            total_words_count=total_words,
            all_errors_with_scores=all_errors_with_scores,
        )

    return [row_idx_to_result[i] for i in range(len(rows))]
