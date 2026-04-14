NORMALIZE_AGAINST_GOLD_PROMPT = """\
<Task>
You are normalizing transcripts for Word Error Rate (WER) calculation. Given two \
transcripts (expected and actual), reformat the actual transcript to a consistent \
surface form with the expected transcript to ensure fair comparison.
</Task>

<Guidelines>
- Treat the expected transcript as a formatting reference, not as a target to copy from
- Preserve the semantic content while normalizing format
- Do not correct transcription errors in the predicted transcript towards ground truth
- Be consistent in your normalization choices
- Empty transcripts should be normalized to empty string (""), not null
- If the predicted transcript is nothing like the expected transcript, return the predicted transcript as is
</Guidelines>

<Allowed Transformations>
1. Converting text to lowercase or uppercase
2. Convert between contracted and expanded forms (e.g., "don't" → "do not", "it is" → "it's")
3. Converting between numbers and their word forms in the correct language \
(e.g., "2" → "two", "first" → "1st", "101" -> "ciento uno")
4. Removing or adding punctuation (e.g., "(123) 543-5987" → "123-543-5987")
5. Normalizing spacing between words or characters
6. Expanding or contracting common abbreviations (e.g., "Dr." → "doctor", "mister" → "Mr.")
7. Normalizing variant spellings to consistent forms (e.g., "okay"/"ok" → "okay")
8. Normalizing spelled-out letters/numbers (e.g., "B-E-E" → "b e e")
9. Changing the spelling of filler words (e.g., "Ummm" → "umm")
10. Using consistent alphabetization when appropriate (e.g.,"日本" → "にほん")
11. Correcting homophones for proper nouns (names, emails) in CJK locales (e.g., "我的名字是羽凡" → "我的名字是雨繁")
</Allowed Transformations>

<Disallowed Transformations>
1. Adding omitted words
2. Removing extra words, including disfluencies and stuttering
3. Substituting gmail and email
4. Swapping between filler words (e.g., "Yes. Yep." → "Yeah. Yes.")
5. Changing numbers or letters from a spelled out name, phone number, date, etc.
6. Correcting homophones for general vocabulary or phrases, \
even if pronunciation (e.g., pinyin in Chinese) is identical but meaning differs.
</Disallowed Transformations>

<Examples>
Input:
Expected: "Yes, my ID is 123-ABC."
Actual: "yeah my i.d. is one two three a b c"

Output:
{
  "normalized_actual": "Yeah, my ID is 123-ABC."
}

Input:
Expected: "Dr. Smith's office, 2nd floor."
Actual: "Doctor Smith office second floor"

Output:
{
  "normalized_actual": "Dr. Smith's office, 2nd floor."
}

Input:
Expected: "Please enter your account number."
Actual: ""

Output:
{
  "normalized_actual": ""
}

Input:
Expected: "My tracking code is C N 8 6 1."
Actual: "My tracking code is umm CM861."

Output:
{
  "normalized_actual": "My tracking code is umm C M 8 6 1."
}

Input:
Expected: "我叫雨繁"
Actual: "我叫羽凡"

Output:
{
  "normalized_actual": "我叫雨繁"
}
</Examples>

<Input>
Expected:
{expected_transcript}

Actual:
{actual_transcript}
</Input>

<Output>
Return a JSON object with one field: "normalized_actual" containing the normalized \
version of the actual transcript against the expected transcript.
</Output>"""

SCORE_TRANSCRIPT_PROMPT = """\
<Task>
You are evaluating the quality of a transcription task. You are given two strings:
- **Expected**: The correct/reference transcription.
- **Actual**: The transcription produced by a model or annotator.

Your job is to compare the two strings and return a JSON object with a 'score' field. \
This field must be an integer from 0 to 3, indicating how closely the actual \
transcription matches the expected one.
</Task>

<Scoring Criteria>
- **Score 0** – *Missing Transcription*
  The actual output is empty, null, or contains no recognizable speech content at all.

- **Score 1** – *Really Bad but Not Missing*
  The actual output contains some speech content but is largely incorrect, with most \
words being wrong, the intent being completely misunderstood, or inputs being incorrect.

- **Score 2** – *Acceptable*
  The core intent is preserved. Most words are correct, though there may be noticeable \
omissions, substitutions, or reordering.

- **Score 3** – *Near-Perfect Match*
  The actual output closely matches the expected output, ignoring differences in \
punctuation, capitalization, spacing, and trivial formatting.
</Scoring Criteria>

<Guidance>
- Ignore differences in:
  - Punctuation (e.g., periods, commas, dashes)
  - Capitalization (e.g., "hello" vs. "HELLO")
  - Spacing (e.g., "icecream" vs. "ice cream")
- Treat numeric equivalents as equal (e.g., "2" == "two")
- Consider equivalent variants (e.g., "z" == "zed")
- Check for correctness of inputs, such as names, email addresses, spellings, phone \
numbers, etc. This should be a major error if at all wrong.
- If the actual transcription is in a different language, it is a major error, and \
should be scored as 1.
- For spelled out words, check for correctness of spelling, but differences in format \
(all caps, spaces, etc.) should be ignored.
- If a transcript is inaudible, this is not considered missing.
- If both actual and expected transcriptions are empty, score as 3 (perfect match)
</Guidance>

<Examples>

<Example Score 0 - Missing Transcription>
Expected:
Bom dia. De onde é que estar falando?

Actual:


Response:
{
    "reason":"Missing Transcription",
    "score": 0,
}
</Example>

<Example Score 1 - Really Bad but Not Missing>
Expected:
Good morning. Where are you calling from?

Actual:
Increase the credit card limit.

Response:
{
    "reason":"the meaning is completely different",
    "score": 1,
}
</Example>
<Example Score 1 - Incorrect inputs>
Expected:
ID1593029

Actual:
I D one five nine three eight
</Example>

<Example Score 1 - Incorrect language>
Expected:
vale vale

Actual:
भले भले
</Example>

<Example Score 2 - Acceptable>
Expected:
Attends une minute, attends une minute!

Actual:
Veuillez patienter un instant, s'il vous plaît.

Response:
{
    "reason":"the meaning is preserved, but there are some substitutions",
    "score": 2,
}
</Example>

<Example Score 3 - Near-Perfect Match>
Expected:
नमस्ते। आप कहाँ से बोल रहे हैं?

Actual:
नमस्ते... आप कहाँ से बोल रहे हैं?

Response:
{
    "reason":"the utterances are semantically equivalent",
    "score": 3,
}
</Example>

<Example Score 3 - Near-Perfect Match>
Expected:
Oi bom dia

Actual:
Oi, bom dia!

Response:
{
    "reason":"the utterances are semantically equivalent",
    "score": 3,
}
</Example>
</Examples>

<Input>

These are the input transcripts, do not use any of the information from the examples \
to score the output.

Expected:
{gold_transcript}

Actual (this maybe empty, according to the scoring criteria):
{llm_transcript}
</Input>"""

SIGNIFICANT_WORD_ERRORS_PROMPT = """\
<Task>
You are evaluating the quality of a transcription task. You are given two strings:
- Expected: The correct/reference transcription.
- Actual: The transcription produced by a model.

Additionally, you are given the words where the mistranscription occurred.
The mistranscription can be in the form of a substitution, deletion, or insertion.

Your job is to compare the two words and return a JSON object with a 'score' field \
for every error. This field must be an integer from 1 to 3, indicating how closely \
the actual transcription matches the expected one.
</Task>

<Scoring Criteria>
- Score 1 – Significant Error
  The meaning of the whole sentence is derailed or incoherent because of the error.

- Score 2 – Minor Error
  The meaning of the overall sentence doesn't change. The words may be completely \
different in meaning, but it doesnt derail the meaning of the whole sentence.

- Score 3 – No Error
  The words are semantically the same.
</Scoring Criteria>

<Guidance>
- Ignore differences in:
  - Punctuation (e.g., periods, commas, dashes)
  - Capitalization (e.g., "hello" vs. "HELLO")
  - Spacing (e.g., "icecream" vs. "ice cream")
- Treat numeric equivalents as equal (e.g., "2" == "two")
- Consider equivalent variants (e.g., "z" == "zed")
- Check for correctness of inputs, such as names, email addresses, spellings, phone \
numbers, etc. If it is spelled out letters or numbers (H E N R Y), a single letter \
difference is considered a significant error. If it is a proper noun as a whole \
(henry), a single letter difference should be considered a minor error.
- If the actual transcription is in a different language, it is a major error, and \
should be scored as 1.
- For spelled out words, check for correctness of spelling, but differences in format \
(all caps, spaces, etc.) should be ignored.
- If a transcript is inaudible, this is not considered missing.
- If the expected transcript contains "<unintelligible>", that comparison should \
receive a score of 3.
</Guidance>
</Task>

<Input>
Expected:
{expected_transcript}

Actual:
{actual_transcript}
</Input>

<Examples>
<Example>
Expected:
I want to check my balance

Actual:
I wanted to check my balance

Response:
{
  "scores": [
    {
      "error": "want substituted for wanted",
      "score": 3,
    }
  ]
}
</Example>
<Example>
Expected:
That is right

Actual:
Thats not right, no

Response:
{
  "scores": [
    {
      "error": "Substitution: 'that' to 'thats' at position 0",
      "score": 3,
    },
    {
      "error": "Substitution: 'is' to 'not' at position 2",
      "score": 1,
    },
    {
      "error": "Insertion: 'no' at position 4",
      "score": 1,
    }
  ]
}
</Example>
<Example>
Expected:
john franklin

Actual:
john franklim

Response:
{
  "scores": [
    {
      "error": "Substitution: 'franklin' to 'franklim'",
      "score": 2,
    }
  ]
}
</Example>
<Example>
Expected:
R O S E

Actual:
R O S D

Response:
{
  "scores": [
    {
      "error": "Substitution: 'D' to 'E'",
      "score": 1,
    }
  ]
}
</Example>

</Examples>

<Input>
Expected:
{expected_transcript}

Actual:
{actual_transcript}
</Input>

<Output>
Return a JSON object.
{
  "scores": [
    {errors}
  ]
}
</Output>"""


if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys

    import requests
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the normalize-against-gold prompt through the LLM.")
    parser.add_argument("--expected", required=True, help="The expected (gold) transcript")
    parser.add_argument("--actual", required=True, help="The actual (predicted) transcript")
    parser.add_argument("--dry-run", action="store_true", help="Print the prompt without calling the LLM")
    args = parser.parse_args()

    rendered = NORMALIZE_AGAINST_GOLD_PROMPT.replace("{expected_transcript}", args.expected).replace(
        "{actual_transcript}", args.actual
    )

    if args.dry_run:
        print(rendered)
        sys.exit(0)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    print(f"Expected:  {args.expected}")
    print(f"Actual:    {args.actual}")
    print("Calling gpt-4.1...")

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": rendered}],
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "normalize_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "normalized_actual": {"type": "string"},
                        },
                        "required": ["normalized_actual"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    result = json.loads(resp.json()["choices"][0]["message"]["content"])

    print(f"\nResult:    {result['normalized_actual']}")
