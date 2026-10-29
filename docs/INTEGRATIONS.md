# Integrations

Two places to put promptscan: in CI, to screen for injections in files your AI
agents read, and at runtime, so untrusted text is checked before it reaches a model.

Every entry point reports two kinds of finding:

- **injection**: the classifier scored the text, an HTML comment with surrounding context, or text hidden
  in Unicode tag characters above the decision threshold. The report points at the
  highest-scoring paragraph.
- **hidden**: zero-width characters, bidirectional overrides or Unicode tag characters.
  These are reported deterministically, with the hidden text decoded, as hygiene warnings. Legitimate files can contain these characters.

Exit status: 0 no blocking findings, 1 blocking findings, 2 error. Injection findings
block by default; Unicode hygiene is a warning. Use `--fail-on all` for strict
Unicode enforcement, or `--no-fail` for reporting only.

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
      - uses: 0xdewy/promptscan@<reviewed-commit-sha>
        id: scan
        with:
          path: .              # default; or prompts/
          fail: "true"         # "false" to annotate without failing
      - uses: github/codeql-action/upload-sarif@v4   # optional: Security tab
        if: always()
        with:
          sarif_file: ${{ steps.scan.outputs.sarif }}
```

On pull requests the Action reports new findings in modified files, comparing base
and head with the same model and cutoff. It uses the event's base commit and its
`.promptscanignore`, so a contributor cannot silence a payload by editing the PR's
ignore file. `path` also scopes PR scans. The CLI's `--all-findings` includes
existing findings in changed files when that is useful.

Pin the Action to a reviewed commit. By default it installs its own bundled code,
not the latest PyPI package. `version` optionally selects an exact package version
(0.3.0 or later). Each code release pins the default model revision. JSON and SARIF
record the scanner version, policy, model identity and cutoff. A local model is
identified by an artifact digest. Custom Hub models support `--model-revision`.

The default scan includes Markdown, text, rules, Python, JavaScript, TypeScript,
JSON, YAML and TOML. Generated directories, symlinks, binary files and files over
1 MB are skipped. JSON lists skipped paths and reasons; the terminal summary
reports the skipped count. No matching inputs is reported explicitly.

Use `include` and `exclude` Action inputs for additional globs, one per line, or
`--include`, `--exclude` and `.promptscanignore` with the CLI. File extensions in
the pre-commit hook match the directory scanner. An explicit file bypasses the
extension filter. Use `fail-on: all` to block on Unicode hygiene too.

## pre-commit

```yaml
repos:
  - repo: https://github.com/0xdewy/promptscan
    rev: <reviewed-commit-sha>
    hooks:
      - id: promptscan
```

## Command line

```bash
promptscan .                              # whole repo
promptscan CLAUDE.md AGENTS.md prompts/   # specific files and directories
promptscan --changed origin/main          # new findings in changed files
promptscan src --changed origin/main      # restrict a PR scan to src/
promptscan --changed origin/main --all-findings
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
windows and the highest window wins. The entire supported input is covered;
coverage does not guarantee that the model recognizes every injection. To also catch hidden Unicode text and locate the offending paragraph:

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
pip install 'promptscan[mcp]==0.3.0'
```

```json
{"mcpServers": {"promptscan": {"command": "promptscan-mcp"}}}
```

For a version-pinned launch without a global install:

```json
{"mcpServers": {"promptscan": {"command": "uvx", "args": ["--from", "promptscan[mcp]==0.3.0", "promptscan-mcp"]}}}
```

Tools: `scan_text(text)`, `scan_many(texts)`, `scan_path(path)`. Text results return
the complete scanner decision score, threshold and findings. Path results include
coverage and skipped paths. Findings include severity; responses identify the model. Tell the agent to call `scan_text` on fetched
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

200 requests per IP per day are free. Use this when you cannot ship a 568 MB model.

## Agent skill

The [promptscan skill](../skills/promptscan/SKILL.md) covers CLI and MCP usage,
coverage checks, and interpreting findings. Copy `skills/promptscan` into your
agent’s skill directory (for example `~/.codex/skills/`). It uses the same pinned
0.3.0 package and model as CI and MCP.
