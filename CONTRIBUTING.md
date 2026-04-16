# Contributing to MU-Bench

Thank you for your interest in contributing to MU-Bench!

## Submitting benchmark results

If you want to add your model to the leaderboard, see **[`submissions/SUBMITTING.md`](submissions/SUBMITTING.md)** for the full submission format and process.

## Reporting issues

- **Bugs or scoring questions** — [Open a GitHub issue](https://github.com/sierra-research/mu-bench/issues)
- **Security vulnerabilities** — See [SECURITY.md](SECURITY.md)

## Contributing code or documentation

1. Fork the repo and create a branch from `main`.
2. Install dev dependencies:

```bash
pip install -e ".[scoring]"
pre-commit install
```

3. Make your changes. The codebase uses:
   - **Python**: [ruff](https://docs.astral.sh/ruff/) for linting and formatting
   - **Leaderboard (JS)**: eslint + prettier — run `npm run lint` and `npm run format:check` in `leaderboard/`

4. Run validation locally if you touched scoring or submission code:

```bash
python scoring/validate.py submissions/raw/deepgram-nova3 --manifest manifest.json
```

5. Open a pull request. CI will run linting and validation automatically.

## Code style

- Python: `ruff` handles both linting and formatting. Pre-commit hooks run automatically if installed.
- JavaScript: Prettier for formatting, ESLint for linting. Both are configured in `leaderboard/`.

## What needs `scoring/prompts.py`?

The file `scoring/prompts.py` contains LLM prompt templates and is **not included in the repo** — it's injected via GitHub secret in CI. You do **not** need it for:

- Running `scoring/validate.py`
- Importing `scoring.metrics` (WER, simple WER, data classes)
- Importing `scoring.normalize.load_ground_truth_from_manifest`

You **do** need it for running the full normalization or scoring pipeline (LLM-based functions).
