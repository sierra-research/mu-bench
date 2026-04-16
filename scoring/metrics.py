"""Core metrics implementation: WER, Quality Score, Significant WER.

Self-contained module adapted from the benchmark evaluation pipeline.
Uses LLM-based normalization via OpenAI for fair transcript comparison.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from jiwer import process_words, wer

from scoring.llm import (
    NORMALIZE_SCHEMA,
    QUALITY_SCORE_SCHEMA,
    SIGNIFICANT_WER_SCHEMA,
    get_responses,
    load_responses,
)


def _load_prompts():
    """Lazy-load scoring prompts so modules that don't need LLM calls can import metrics freely."""
    from scoring.prompts import (
        NORMALIZE_AGAINST_GOLD_PROMPT,
        SCORE_TRANSCRIPT_PROMPT,
        SIGNIFICANT_WORD_ERRORS_PROMPT,
    )

    return NORMALIZE_AGAINST_GOLD_PROMPT, SCORE_TRANSCRIPT_PROMPT, SIGNIFICANT_WORD_ERRORS_PROMPT


@dataclass
class TranscriptRow:
    """A paired ground truth + predicted transcript for scoring."""

    locale: str
    utterance_id: str
    gold: str
    predicted: str


@dataclass
class WERResult:
    """Result from WER computation."""

    locale: str
    utterance_id: str
    gold: str
    predicted: str
    normalized_gold: Optional[str] = None
    normalized_predicted: Optional[str] = None
    wer: Optional[float] = None


@dataclass
class QualityResult:
    """Result from quality scoring."""

    locale: str
    utterance_id: str
    gold: str
    predicted: str
    score: Optional[int] = None  # 0-3


@dataclass
class SignificantWERResult:
    """Result from significant WER computation."""

    locale: str
    utterance_id: str
    gold: str
    predicted: str
    normalized_gold: Optional[str] = None
    normalized_predicted: Optional[str] = None
    major_error_rate: Optional[float] = None
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
    """Simple normalization: lowercase, remove punctuation/spaces."""
    if text is None:
        return ""
    t = str(text).lower()
    for ch in [",", ".", "-", "_", "/", "\\", "(", ")", "[", "]", ":", ";", " "]:
        t = t.replace(ch, "")
    return t


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
        normalization_prompt, _, _ = _load_prompts()
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


def compute_wer(rows: List[TranscriptRow]) -> List[WERResult]:
    """Compute Word Error Rate on pre-normalized transcript pairs.

    Expects rows with gold and predicted already normalized (via scoring.normalize).
    Uses character-level tokenization for CJK locales.
    """
    results = []
    for r in rows:
        if is_unintelligible(r.gold) or not r.gold:
            w = None
        else:
            tok_gold = tokenize_for_alignment(r.gold, r.locale)
            tok_pred = tokenize_for_alignment(r.predicted, r.locale)
            w = float(wer(tok_gold, tok_pred))

        results.append(
            WERResult(
                locale=r.locale,
                utterance_id=r.utterance_id,
                gold=r.gold,
                predicted=r.predicted,
                normalized_gold=r.gold,
                normalized_predicted=r.predicted,
                wer=w,
            )
        )

    return results


def compute_simple_wer(rows: List[TranscriptRow]) -> List[WERResult]:
    """Compute WER with simple normalization (no LLM calls needed)."""
    results = []
    for r in rows:
        if is_unintelligible(r.gold) or not r.gold:
            w = None
        else:
            norm_g = normalize_for_match(r.gold)
            norm_p = normalize_for_match(r.predicted)
            if not norm_g:
                w = None
            else:
                w = float(wer(norm_g, norm_p))

        results.append(
            WERResult(
                locale=r.locale,
                utterance_id=r.utterance_id,
                gold=r.gold,
                predicted=r.predicted,
                normalized_gold=normalize_for_match(r.gold) if r.gold else None,
                normalized_predicted=normalize_for_match(r.predicted) if r.predicted else None,
                wer=w,
            )
        )
    return results


def compute_quality(rows: List[TranscriptRow], num_workers: int = 8) -> List[QualityResult]:
    """Compute LLM-judged quality score (0-3 scale)."""
    _, SCORE_TRANSCRIPT_PROMPT, _ = _load_prompts()
    pairs: List[Tuple[int, str]] = []
    for row_idx, r in enumerate(rows):
        if not r.gold:
            continue
        pairs.append(
            (
                row_idx,
                SCORE_TRANSCRIPT_PROMPT.format(gold_transcript=r.gold, llm_transcript=r.predicted),
            )
        )

    row_idx_to_score: Dict[int, Optional[int]] = {}
    if pairs:
        chunk_size = max(1, min(num_workers, len(pairs)))
        for i in range(0, len(pairs), chunk_size):
            chunk = pairs[i : i + chunk_size]
            prompts = [p for _, p in chunk]
            responses = get_responses(prompts, num_workers=num_workers, response_format=QUALITY_SCORE_SCHEMA)
            loaded = load_responses(responses)

            for j, (row_idx, _) in enumerate(chunk):
                score = None
                if is_unintelligible(rows[row_idx].gold):
                    row_idx_to_score[row_idx] = None
                    continue
                if j < len(loaded) and loaded[j] is not None:
                    resp = loaded[j]
                    if isinstance(resp, dict) and "score" in resp:
                        try:
                            score = int(resp["score"])
                        except Exception:
                            score = None
                row_idx_to_score[row_idx] = score

            print(f"Quality scoring: {min(i + chunk_size, len(pairs))}/{len(pairs)} processed")

    results = []
    for row_idx, r in enumerate(rows):
        results.append(
            QualityResult(
                locale=r.locale,
                utterance_id=r.utterance_id,
                gold=r.gold,
                predicted=r.predicted,
                score=row_idx_to_score.get(row_idx),
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
    _, _, SIGNIFICANT_WORD_ERRORS_PROMPT = _load_prompts()
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
                major_error_rate=None,
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
                major_error_rate=0.0,
                major_errors_count=0,
                total_words_count=total_words,
                all_errors_with_scores=[],
            )

    # Score errors via LLM
    row_idx_to_scored_errors: Dict[int, List[Dict[str, Any]]] = {}

    if error_scoring_prompts:
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
                f"Significant WER scoring: {min(i + chunk_size, len(error_scoring_prompts))}/{len(error_scoring_prompts)} processed"
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
                major_error_rate=None,
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

        major_error_rate = major_errors_count / total_words if total_words > 0 else 0.0

        row_idx_to_result[row_idx] = SignificantWERResult(
            locale=r.locale,
            utterance_id=r.utterance_id,
            gold=r.gold,
            predicted=r.predicted,
            normalized_gold=norm_g,
            normalized_predicted=norm_p,
            major_error_rate=major_error_rate,
            major_errors_count=major_errors_count,
            total_words_count=total_words,
            all_errors_with_scores=all_errors_with_scores,
        )

    return [row_idx_to_result[i] for i in range(len(rows))]
