"""LLM prompts from the ICASSP 2027 submission.

These are the prompts behind the paper's Utterance Error Rate (UER) and
normalization experiments. They are NOT the prompts the public leaderboard was
scored with: the leaderboard is a snapshot of provider models from April 2026,
scored by ``scoring/score.py`` with an earlier pipeline whose prompts are held
in the ``SCORING_PROMPTS_PY`` GitHub secret. Nothing in ``scoring/`` imports
this module; it is published so the paper's method can be reproduced.

Judge configuration used in the paper (OpenAI Chat Completions / Responses API):

* ``NORMALIZATION_PROMPT`` -- model ``gpt-6-astra``, ``seed=7``,
  ``reasoning_effort="high"``. Reference-guided: rewrites the prediction into
  the reference's format and leaves the reference untouched.
  Template variables ``{ground_truth}``, ``{predicted}``; the response is a JSON
  object ``{"normalized_predicted": ...}``.
* ``UER_JUDGE_PROMPT`` -- model ``gpt-6-astra``, ``seed=7``,
  ``reasoning_effort="low"``. Whole-utterance three-tier judge; UER is the
  share of utterances scored 1. Template variables ``{locale}``,
  ``{expected_transcript}``, ``{actual_transcript}``; the response is a JSON
  object ``{"reason": ..., "score": 1 | 2 | 3}``. Pairs whose normalized
  strings are identical are scored 3 without a call.
* ``BLIND_NORMALIZATION_PROMPT`` -- reference-blind ablation, same model and
  settings as ``NORMALIZATION_PROMPT``, applied separately to each side.
  Template variables ``{locale}``, ``{text}``; the response is a JSON object
  ``{"normalized_text": ...}``.

All three are ``str.format`` templates, so literal braces in the JSON examples
are doubled.
"""

UER_JUDGE_PROMPT = """
<Task>
You are evaluating the quality of a speech transcription. You are given two strings:
- Expected: The correct/reference transcription.
- Actual: The transcription produced by a model.

Compare the two transcripts as a whole and return a JSON object with a 'reason' and a 'score' field. \
The 'reason' field should be a brief explanation of your score. The 'score' field must be an integer from 1 to 3.
</Task>

<Scoring Criteria>
- Score 1 – Significant Error
  The meaning of the utterance is derailed, incoherent, or changed because of the transcription.

- Score 2 – Minor Error
  The transcripts differ, but the meaning of the utterance is preserved.

- Score 3 – No Error
  The transcripts are semantically the same.
</Scoring Criteria>

<Input>
Locale: {locale}

Expected:
{expected_transcript}

Actual:
{actual_transcript}
</Input>

<Output>
{{
  "reason": "<brief explanation>",
  "score": <1, 2, or 3>
}}
</Output>
""".strip()

NORMALIZATION_PROMPT = """
<Task>
You are normalizing the transcript predicted by an ASR model after listening to an audio clip. \
Your goal is to ensure that the predicted transcript follows the same canonical format as a given ground truth transcript \
by reconciling formatting differences.
</Task>

<Rules>
You must perform this task without changing the semantic content of the predicted transcript because that would hide true ASR errors. \
If in doubt, prefer not to edit the predicted transcript.
</Rules>

<Examples of equivalent formatting differences>
1. Lowercase vs Uppercase
2. Punctuation
3. Expanded contractions: "don't" vs "do not"
4. Common abbreviations: "Dr." vs "doctor", "ok" vs "okay"
5. Spelled-out letters or numbers: "B-E-E" vs "b e e" vs "b...e...e..."
6. Inclusion/exclusion of filler words: "umm hello" vs "hello"
7. Numeric representation: "1234567890" vs "one two three four five six seven eight nine zero"
8. Email/phone number/address/known entity formats: "g mail dot com" vs "gmail.com", "123-456-7890" vs "1234567890"
9. Formats of unknown input sequences: "id 1 2 3 45" vs "id 123-45" vs "ID12345"
10. Spelling variants of the same word: "grey" vs "gray"
11. Proper nouns that are pronounced identically but have unclear spellings: "Jon" vs "John"
</Examples of equivalent formatting differences>

<Differences NOT to reconcile>
1. Missing or extra words in the predicted transcript (other than fillers). Never add, remove, or reorder words
2. Homophones used incorrectly: "their" vs "there"
3. Near-homophones: "fifteen" vs "fifty"
</Differences NOT to reconcile>

<Input>
Ground Truth:
{ground_truth}

Predicted:
{predicted}
</Input>

<Output>
Return a JSON object with a single field "normalized_predicted" containing the normalized predicted text.
</Output>""".strip()

# Reference-blind normalization ablation (paper Sec. 7): the same prompt is
# applied independently to the reference and to the prediction, so neither call
# sees the other string. Derived from the leaderboard's prediction-blind gold
# normalizer with two additions: a fixed spoken form for contact symbols
# (@, ., +) and an explicit digit-by-digit rule for identifiers, which are the
# two largest classes the reference-guided normalizer reconciles (paper Table 6).
BLIND_NORMALIZATION_PROMPT = """\
<Task>
You are normalizing a single speech transcript into a canonical spoken form. \
The same procedure is applied separately to reference transcripts and to ASR \
predictions, so you are NOT given the other transcript and must not assume \
one. Your output must depend only on the text you are given.

This transcript's locale is {locale}. Digits and symbols must be spelled in \
the locale's native language: en-US uses English words, es-MX Spanish, tr-TR \
Turkish, vi-VN Vietnamese, zh-CN Chinese characters; never cross-translate.
</Task>

<Normalization Rules>
1. Convert all text to lowercase.
2. Expand contractions (e.g., "don't" → "do not").
3. Write every digit as a word in the locale's language, one word per digit \
(en-US "123" → "one two three"; es-MX "4 5 4" → "cuatro cinco cuatro"; tr-TR \
"122" → "bir iki iki"; vi-VN "84" → "tám bốn"; zh-CN "138" → "一 三 八"). This \
applies to phone numbers, account, card, ID and case numbers, codes, and any \
digit run; numbers already spoken as words stay as they are.
4. Spell identifier letters one letter per token: "B-E-E" → "b e e", "CN6378246" \
→ "c n six three seven eight two four six", "ID 123-45" → "id one two three four five".
5. Write contact symbols as spoken words: "@" → en-US "at", es-MX "arroba", \
tr-TR "et", vi-VN "a còng", zh-CN "艾特"; "." inside an email address or URL → \
"dot" / "punto" / "nokta" / "chấm" / "点"; "+" before a phone number → "plus" / \
"más" / "artı" / "cộng" / "加". Hyphens, slashes and parentheses inside numbers \
are dropped. Spoken forms that are already words ("ashley dot brown at email dot \
com") stay as they are.
6. Remove all remaining punctuation and collapse whitespace to single spaces.
7. Expand common abbreviations ("Dr." → "doctor", "Mr." → "mister") and normalize \
variant spellings to one form ("ok" → "okay").
8. Remove non-lexical fillers and backchannels in every locale: en-US "um", "uh", \
"hmm", "mm-hmm"; es-MX "eh", "este" (as hesitation), "mm"; tr-TR "şey", "yani" \
(as hesitation); vi-VN "ờ", "ừ", "à" (as filler); zh-CN "嗯", "啊", "呃". A \
filler-only utterance normalizes to "". Substantive words ("yes", "sí", "evet", \
"vâng", "是") are never fillers.
9. Do NOT add, remove, reorder or correct content words. Do NOT guess at \
homophones or fix spellings; proper nouns stay as written, lowercased.
10. Empty input normalizes to "".
</Normalization Rules>

<Examples>
Input (locale: en-US): "Ashley dot Brown at email dot com."
Output: {{"normalized_text": "ashley dot brown at email dot com"}}

Input (locale: en-US): "ashley.brown@email.com"
Output: {{"normalized_text": "ashley dot brown at email dot com"}}

Input (locale: en-US): "C N 6 3 7 8 2 4 6."
Output: {{"normalized_text": "c n six three seven eight two four six"}}

Input (locale: en-US): "CN6378246"
Output: {{"normalized_text": "c n six three seven eight two four six"}}

Input (locale: es-MX): "Es Juan punto López arroba email punto com."
Output: {{"normalized_text": "es juan punto lópez arroba email punto com"}}

Input (locale: es-MX): "Es juan.lopez@email.com."
Output: {{"normalized_text": "es juan punto lopez arroba email punto com"}}

Input (locale: tr-TR): "Hesap numaram C N 1 2 2."
Output: {{"normalized_text": "hesap numaram c n bir iki iki"}}

Input (locale: vi-VN): "Số điện thoại là +84 9 1 1 2 3."
Output: {{"normalized_text": "số điện thoại là cộng tám bốn chín một một hai ba"}}

Input (locale: zh-CN): "我的电话是 1 3 8。"
Output: {{"normalized_text": "我的电话是 一 三 八"}}

Input (locale: vi-VN): "Ờ, ừ."
Output: {{"normalized_text": ""}}
</Examples>

<Input (locale: {locale})>
{text}
</Input>

<Output>
Return a JSON object with a single field "normalized_text" containing the \
normalized transcript.
</Output>"""
