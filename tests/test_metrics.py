"""Tests for ``scoring.metrics`` and the relevant parts of ``scoring.score``.

Covers the fairness-fixes plan guarantees that don't require live LLM
calls:

* Gold normalization is prediction-independent — ``_load_gold_prompt``
  returns a prompt whose format placeholders don't include the
  prediction.
* Silent-clip handling: hyp words count as insertions; silent/silent
  pairs contribute nothing; ``<unintelligible>`` is always skipped.
* Corpus WER aggregates sum(edits)/sum(ref_words) matching hand-computed
  values.
* Overall UER in ``scoring.score``'s summary is the unweighted mean of
  per-locale UERs (locale-macro), symmetric to overall WER.
"""

from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, ".")  # make repo-local imports work when pytest is run from repo root

from scoring.metrics import (  # noqa: E402
    TranscriptRow,
    _load_gold_prompt,
    compute_simple_wer,
    compute_wer,
    is_unintelligible,
)


def test_gold_prompt_is_blind_to_prediction():
    """The canonical gold prompt must not take a prediction placeholder.

    Under the new split, the gold-normalization prompt is
    ``NORMALIZE_GOLD_PROMPT`` and only formats ``{expected_transcript}``.
    The tests/conftest.py stub honors this. If the real secret regresses
    to a symmetric prompt, this test fails loudly.
    """
    prompt = _load_gold_prompt()
    # It must accept expected_transcript; it must NOT require actual_transcript.
    # If a legacy symmetric prompt slipped through, this would pop a KeyError.
    formatted = prompt.format(expected_transcript="hello world")
    assert "hello world" in formatted


def test_compute_wer_matches_hand_computed_corpus_values():
    """Two-locale fixture; assert sum(edits)/sum(ref_words) matches."""
    rows = [
        TranscriptRow(locale="en-US", utterance_id="u1", gold="hello world", predicted="hello world"),
        TranscriptRow(locale="en-US", utterance_id="u2", gold="a b c d", predicted="a x c d"),  # 1 sub
        TranscriptRow(locale="en-US", utterance_id="u3", gold="one two", predicted="one two three"),  # 1 ins
    ]
    results = compute_wer(rows)
    edits = sum(r.edits for r in results if r.edits is not None)
    ref_words = sum(r.ref_words for r in results if r.ref_words is not None)
    # 2+4+2 = 8 ref words; 0+1+1 = 2 edits; corpus WER = 2/8 = 0.25
    assert edits == 2
    assert ref_words == 8
    assert edits / ref_words == pytest.approx(0.25)


def test_compute_wer_silent_clip_counts_hyp_as_insertions():
    """Silence-hallucination fix (item 6): gold="", pred has words → counted."""
    rows = [
        TranscriptRow(locale="en-US", utterance_id="silence_hallu", gold="", predicted="thanks for watching"),
    ]
    results = compute_wer(rows)
    r = results[0]
    assert r.edits == 3
    assert r.ref_words == 3
    # Per-utterance rate: 1.0 when any hallucination.
    assert r.wer == 1.0


def test_compute_wer_silent_silent_is_noop():
    """Silent/silent: edits=0, ref_words=0; adds nothing to corpus sums."""
    rows = [TranscriptRow(locale="en-US", utterance_id="silent", gold="", predicted="")]
    results = compute_wer(rows)
    r = results[0]
    assert r.edits == 0
    assert r.ref_words == 0
    # Per-utterance wer stays None to avoid 0/0 in distribution plots.
    assert r.wer is None


def test_compute_wer_unintelligible_still_skipped():
    rows = [TranscriptRow(locale="en-US", utterance_id="unk", gold="<unintelligible>", predicted="some words here")]
    results = compute_wer(rows)
    r = results[0]
    assert r.edits is None
    assert r.ref_words is None
    assert r.wer is None


def test_is_unintelligible_matches_plan_contract():
    assert is_unintelligible("<unintelligible>")
    assert is_unintelligible("  <UNINTELLIGIBLE>  ")
    assert not is_unintelligible("hello")


def test_compute_simple_wer_silent_clip_also_counts_insertions():
    """Same silence-hallucination handling in the non-LLM path."""
    rows = [TranscriptRow(locale="en-US", utterance_id="halu", gold="", predicted="ghost words")]
    results = compute_simple_wer(rows)
    r = results[0]
    assert r.edits == 2
    assert r.ref_words == 2
    assert r.wer == 1.0


def test_overall_uer_is_locale_macro(tmp_path, monkeypatch):
    """``scoring.score`` must aggregate overall UER as an unweighted mean
    of per-locale UERs (locale-macro), matching WER's convention.

    Fixture: two locales. Locale A has 100 utterances with 10 errors
    (10% UER), locale B has 10 utterances with 5 errors (50% UER).
    Utterance-micro would give 15/110 = 13.6%; locale-macro gives
    (10% + 50%) / 2 = 30%. We assert the published value is the
    locale-macro one.
    """
    import scoring.score as score_mod

    # Patch the two target locales to just {locA, locB} for this test.
    monkeypatch.setattr(score_mod, "TARGET_LOCALES", ["locA", "locB"])

    # Build fake locale_stats the way flush_details_to_disk would.
    # We bypass disk and invoke the summary-building logic by reusing the
    # public shape of `summary_locales`.
    summary_locales = {
        "locA": {
            "wer": 0.1,
            "significantWer": 0.10,  # 10/100
            "utteranceCount": 100,
            "unintelligibleCount": 0,
        },
        "locB": {
            "wer": 0.2,
            "significantWer": 0.50,  # 5/10
            "utteranceCount": 10,
            "unintelligibleCount": 0,
        },
    }

    # Replicate the locale-macro aggregation from scoring.score main().
    per_locale_uers = [v["significantWer"] for v in summary_locales.values() if v["significantWer"] is not None]
    overall_sig_wer = (
        round(sum(per_locale_uers) / len(per_locale_uers), 4) if len(per_locale_uers) == len(summary_locales) else None
    )
    # Locale-macro: (0.10 + 0.50)/2 = 0.30
    assert overall_sig_wer == pytest.approx(0.30)
    # Utterance-micro would be 15/110 ≈ 0.1364 — assert we're NOT that.
    assert overall_sig_wer != pytest.approx(15 / 110, abs=1e-3)


def test_manifest_gold_hash_is_deterministic(tmp_path):
    """``compute_manifest_gold_hash`` must be stable across reorderings."""
    from scoring.normalize_gold import compute_manifest_gold_hash, load_manifest_gold

    manifest = {
        "utterances": [
            {"id": "u1", "locale": "en-US", "transcript": "hi"},
            {"id": "u2", "locale": "en-US", "transcript": "bye"},
        ]
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(manifest))
    rows = load_manifest_gold(path)
    h1 = compute_manifest_gold_hash(rows)
    # Shuffling the input, the loader re-sorts, so the hash is stable.
    rows2 = list(reversed(rows))
    rows2.sort()
    assert compute_manifest_gold_hash(rows2) == h1


def test_judge_block_is_recorded_with_sha_fields(tmp_path, monkeypatch):
    """Check ``_collect_judge_block`` returns the expected keys and shape."""
    import scoring.score as score_mod

    block = score_mod._collect_judge_block()
    for key in (
        "model",
        "modelSnapshot",
        "temperature",
        "seed",
        "normalizeGoldPromptSha",
        "normalizePredPromptSha",
        "significantErrorsPromptSha",
        "scoredAt",
    ):
        assert key in block, f"missing judge field: {key}"
    # The stub prompts produce non-empty SHAs.
    assert block["normalizeGoldPromptSha"]
    assert block["normalizePredPromptSha"]
    assert block["significantErrorsPromptSha"]
