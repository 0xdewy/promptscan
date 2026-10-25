# Agent guidelines for promptscan

promptscan detects prompt injection in text and in files that AI agents read (CLAUDE.md,
AGENTS.md, .cursorrules, prompts/). One fine-tuned DeBERTa-v3-small classifier plus a
deterministic check for hidden Unicode text. Tuned for low false positives over recall.
English only.

## Layout

```
promptscan/
├── __init__.py      version, exports Classifier
├── classifier.py    Classifier.load/score/predict/save (training agent owns this)
├── scan.py          file walking, hidden-Unicode check, findings, JSON/SARIF output
├── cli.py           `promptscan` entry point; exit 0 clean, 1 findings, 2 error
├── mcp_server.py    `promptscan-mcp`: scan_text, scan_many, scan_path
└── utils/colors.py
scripts/             build_dataset.py, train_v2.py, eval_suite.py, selfplay.py, rl_attacker.py
models/release/      current model (config.json, model.safetensors, tokenizer, promptscan.json)
tests/test_scan.py   runs without a model (keyword stub)
docs/INTEGRATIONS.md Action, pre-commit, CLI, runtime guard, MCP, Claude Code hook
action.yml           composite GitHub Action
archive/             old CNN/LSTM/DistilBERT ensemble code, gitignored
```

## Model

- Hugging Face `transformers` format: `config.json` + `model.safetensors` + tokenizer files.
  `config.json` carries a `promptscan` block with `threshold` and `max_length`; the CLI
  honours the stored threshold unless `--threshold` is passed.
- Default repo is `DEFAULT_REPO` in `classifier.py`. Local override: `--model DIR` or
  `PROMPTSCAN_MODEL=DIR`. For development: `PROMPTSCAN_MODEL=models/release`.
- Never load pickle `.pt` files; safetensors only.

## Commands

```bash
uv run --with pytest pytest tests/          # tests
uv run ruff check promptscan/ tests/        # lint (88 cols)
PROMPTSCAN_MODEL=models/release uv run promptscan .   # dogfood; .promptscanignore skips README and fixtures
uv run python scripts/eval_suite.py         # held-out evaluation
uv build && twine check dist/*              # package
HF_TOKEN=... uv run python scripts/publish_model.py models/release   # replace the HF model repo
uv run python scripts/fetch_feedback.py     # pull website feedback into data/v2/raw/feedback.parquet (unverified labels)
uv lock --upgrade && uv run pytest tests/   # dependency refresh; pip-audit via `uvx pip-audit -r <(uv export --no-hashes)`
```

## Rules

- Keep `scan.py` free of model-specific code; it only needs `.score(list[str])` and
  `.threshold`, which is what the test stub provides.
- Any text file the Action scans by default must be listed in `TEXT_SUFFIXES` in `scan.py`
  and in the `files:` regex of `.pre-commit-hooks.yaml`; keep them in sync.
- Changes to detection output (findings, exit codes, SARIF) need a test in `tests/test_scan.py`.
- Version lives in `pyproject.toml` and `promptscan/__init__.py`.
