"""Test fixtures shared across ``tests/``.

The gitignored ``scoring/prompts.py`` file is only present at CI / scoring
time (injected from the ``SCORING_PROMPTS_PY`` GitHub secret). For unit
tests we install a lightweight stub module so ``scoring.metrics``'s
``_load_prompts`` / ``_load_gold_prompt`` helpers resolve without needing
the real secret. The stubbed strings are deliberately minimal — tests
that care about prompt SHAs assert on the SHAs of these exact strings.
"""

from __future__ import annotations

import sys
import types

_STUB_NORMALIZE_GOLD_PROMPT = "Stub gold-only normalization prompt. expected_transcript={expected_transcript}"
_STUB_NORMALIZE_PRED_AGAINST_GOLD_PROMPT = (
    "Stub prediction-against-normalized-gold prompt. "
    "expected_transcript={expected_transcript}\nactual_transcript={actual_transcript}"
)
_STUB_SIGNIFICANT_WORD_ERRORS_PROMPT = (
    "Stub significant-error scoring prompt. "
    "expected_transcript={expected_transcript}\nactual_transcript={actual_transcript}\nerrors={errors}"
)


def _install_prompt_stub() -> None:
    if "scoring.prompts" in sys.modules:
        return
    mod = types.ModuleType("scoring.prompts")
    mod.NORMALIZE_GOLD_PROMPT = _STUB_NORMALIZE_GOLD_PROMPT
    mod.NORMALIZE_PRED_AGAINST_GOLD_PROMPT = _STUB_NORMALIZE_PRED_AGAINST_GOLD_PROMPT
    mod.SIGNIFICANT_WORD_ERRORS_PROMPT = _STUB_SIGNIFICANT_WORD_ERRORS_PROMPT
    sys.modules["scoring.prompts"] = mod


_install_prompt_stub()
