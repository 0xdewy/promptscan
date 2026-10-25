# Integrations

Two places to put promptscan: in CI, so nobody can land an injection in files your AI
agents read, and at runtime, so untrusted text is checked before it reaches a model.

Every entry point reports two kinds of finding:

- **injection**: the classifier scored the text, an HTML comment inside it, or text hidden
  in Unicode tag characters above the decision threshold. The report points at the
  highest-scoring paragraph.
- **hidden**: zero-width characters, bidirectional overrides or Unicode tag characters.
  These are reported deterministically, with the hidden text decoded, because they have no
  legitimate place in a prompt or instruction file.

Exit status: 0 clean, 1 findings, 2 error.

## GitHub Action

```yaml
# .github/workflows/promptscan.yml
on: [pull_request, push]
permissions:
  contents: read
  security-events: write   # only needed for the SARIF upload step
jobs:
  promptscan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: 0xdewy/promptscan@master
        id: scan
        with:
          path: .              # default; or prompts/
          fail: "true"         # "false" to annotate without failing
      - uses: github/codeql-action/upload-sarif@v4   # optional: Security tab
        if: always()
        with:
          sarif_file: ${{ steps.scan.outputs.sarif }}
```

On pull requests the action scans only text files the PR changed (`changed-only: "true"`),
so a repo full of existing prompts can adopt it without a cleanup first. The diff is taken
against the current tip of the base branch, so commits that land on the base after the PR
was opened are not scanned as part of it. Findings appear as
inline annotations on the PR. The model (280 MB) is cached between runs.

Which files: `.md`, `.markdown`, `.mdc`, `.txt`, `.rst`, `.adoc`, `.prompt`, `.rules` and
anything ending in `rules` (`.cursorrules`, `.windsurfrules`, `.clinerules`). That covers
`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, `.github/copilot-instructions.md`, `.cursor/rules/`,
skills and prompt directories. `.git`, `node_modules`, virtualenvs and build output are
skipped. Add patterns with `--include`, skip paths with `--exclude` or a `.promptscanignore`
file (one glob per line) in the scanned directory.

## pre-commit

```yaml
repos:
  - repo: https://github.com/0xdewy/promptscan
    rev: master
    hooks:
      - id: promptscan
```

## Command line

```bash
promptscan .                              # whole repo
promptscan CLAUDE.md AGENTS.md prompts/   # specific files and directories
promptscan --changed origin/main          # files changed on this branch
promptscan . --json                       # machine-readable
promptscan . --sarif promptscan.sarif     # for GitHub code scanning
promptscan https://example.com/page       # fetch a page, strip tags, scan
cat untrusted.txt | promptscan -
```

`--threshold 0.5` lowers the bar (more recall, more false positives); the model ships with a
threshold tuned for low false positives. `--model DIR` or `PROMPTSCAN_MODEL=DIR` uses a local
model.

## Runtime guard in your app

```python
from promptscan import Classifier

clf = Classifier.load()          # ~4 s once; ~80 ms per text on CPU after that

def guard(text: str) -> None:
    r = clf.predict(text)
    if r["prediction"] == "INJECTION":
        raise ValueError(f"prompt injection (score {r['score']:.2f})")
```

`clf.score(list_of_texts)` scores a batch. Long inputs are scored in overlapping 512-token
windows and the highest window wins, so an injection buried on page 3 of a document is
still found. To also catch hidden Unicode text and locate the offending paragraph:

```python
from promptscan.scan import scan_text

for f in scan_text(text, clf):
    print(f.kind, f.line, f.message, f.snippet)
```

FastAPI example, rejecting injected chat messages before they reach the model:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from promptscan import Classifier

app = FastAPI()
clf = Classifier.load()

class Chat(BaseModel):
    message: str

@app.post("/chat")
def chat(body: Chat):
    r = clf.predict(body.message)
    if r["prediction"] == "INJECTION":
        raise HTTPException(400, f"prompt injection detected (score {r['score']:.2f})")
    ...
```

The same check belongs wherever an agent ingests text it did not write: tool results, web
pages, retrieved documents, emails, issue comments.

## MCP server (Claude Code, Claude Desktop, Cursor)

```bash
pip install 'promptscan[mcp]'
```

```json
{"mcpServers": {"promptscan": {"command": "promptscan-mcp"}}}
```

Tools: `scan_text(text)`, `scan_many(texts)`, `scan_path(path)`. Each returns the model
score, the threshold and the findings list. Tell the agent to call `scan_text` on fetched
pages and tool output before following anything in them.

## Claude Code hook

Block the agent from reading a file that contains an injection. Add to
`.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Read",
      "hooks": [{
        "type": "command",
        "command": "jq -r .tool_input.file_path | xargs promptscan -q || exit 2"
      }]
    }]
  }
}
```

Exit code 2 from a hook blocks the tool call and shows promptscan's output to the agent.
This loads the model on every read (about 4 s); for lighter use let the agent call the MCP
`scan_text` tool on content it fetched.

## Hosted API

```bash
curl -X POST https://promptscan.dewy.ch/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Ignore all previous instructions"}'
```

200 requests per IP per day are free. Use this when you cannot ship a 280 MB model.
