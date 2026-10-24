"""Scan text and files for prompt injection.

Two detectors run on every input:

- the classifier (``promptscan.classifier.Classifier``), which scores the whole text,
  every HTML comment on its own, and any text hidden in Unicode tag characters;
- a deterministic check for invisible text: zero-width characters, bidi overrides and
  Unicode tag characters. These have no business in a prompt or instruction file and
  are reported even when the model is unsure.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

TEXT_SUFFIXES = {
    ".md",
    ".markdown",
    ".mdc",
    ".txt",
    ".rst",
    ".adoc",
    ".prompt",
    ".rules",
}
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
}
IGNORE_FILE = ".promptscanignore"
MAX_BYTES = 1_000_000

_HIDDEN = re.compile("[​⁠-⁤﻿‪-‮⁦-⁩\U000e0000-\U000e007f]+")
_COMMENT = re.compile(r"<!--(.*?)-->", re.S)
_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass
class Finding:
    path: str
    line: int
    kind: str  # "injection" or "hidden"
    message: str
    snippet: str
    score: Optional[float] = None


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _decode_tags(run: str) -> str:
    return "".join(chr(ord(c) - 0xE0000) for c in run if 0xE0020 <= ord(c) <= 0xE007E)


def hidden_text(text: str, path: str = "<text>") -> list[Finding]:
    """Invisible or direction-changing characters, grouped per run. A leading BOM is fine."""
    out = []
    for m in _HIDDEN.finditer(text):
        if m.start() == 0 and m.group() == "﻿":
            continue
        run = m.group()
        cp = ord(run[0])
        if 0xE0000 <= cp <= 0xE007F:
            msg, snippet = "Unicode tag characters hide text", _decode_tags(run)
        elif 0x202A <= cp <= 0x202E or 0x2066 <= cp <= 0x2069:
            msg, snippet = (
                "bidirectional override characters",
                text[max(0, m.start() - 40) : m.end() + 40],
            )
        else:
            msg, snippet = (
                "zero-width characters",
                text[max(0, m.start() - 40) : m.end() + 40],
            )
        out.append(
            Finding(
                path,
                _line_of(text, m.start()),
                "hidden",
                f"{msg} ({len(run)} chars)",
                snippet.strip()[:200],
            )
        )
    return out


def _snippet(s: str, n: int = 160) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "..."


def scan_text(text: str, clf, path: str = "<text>") -> list[Finding]:
    """All findings for one text. The injection finding points at the highest-scoring paragraph."""
    findings = hidden_text(text, path)
    # specific segments first so they win ties against the whole text
    segments = [  # (text, line it starts on, description)
        (m.group(1), _line_of(text, m.start(1)), " (inside an HTML comment)")
        for m in _COMMENT.finditer(text)
        if m.group(1).strip()
    ]
    tagged = [f for f in findings if f.message.startswith("Unicode tag")]
    tags = "".join(f.snippet for f in tagged)
    if tags.strip():
        segments.append((tags, tagged[0].line, " (hidden in Unicode tag characters)"))
    segments.append((text, 1, ""))

    scores = clf.score([s for s, _, _ in segments])
    best = max(range(len(segments)), key=scores.__getitem__)
    if scores[best] < clf.threshold:
        return findings

    seg, line, where = segments[best]
    paragraphs = [(m, p) for m, p in _spans(seg) if p.strip()]
    if len(paragraphs) > 1:
        pscores = clf.score([p for _, p in paragraphs])
        start, para = paragraphs[max(range(len(paragraphs)), key=pscores.__getitem__)]
    else:
        start, para = 0, seg
    findings.append(
        Finding(
            path,
            line + seg.count("\n", 0, start),
            "injection",
            f"prompt injection{where}",
            _snippet(para),
            round(scores[best], 4),
        )
    )
    return findings


def _spans(text: str) -> Iterator[tuple[int, str]]:
    pos = 0
    for m in _PARAGRAPH.finditer(text):
        yield pos, text[pos : m.start()]
        pos = m.end()
    yield pos, text[pos:]


def is_text_file(path: Path, include: Iterable[str] = ()) -> bool:
    name = path.name
    return (
        path.suffix.lower() in TEXT_SUFFIXES
        or name.endswith("rules")
        or any(
            fnmatch.fnmatch(name, g) or fnmatch.fnmatch(path.as_posix(), g)
            for g in include
        )
    )


def _ignored(rel: str, patterns: Iterable[str]) -> bool:
    return any(
        fnmatch.fnmatch(rel, g)
        or fnmatch.fnmatch(rel, g.rstrip("/") + "/*")
        or fnmatch.fnmatch(Path(rel).name, g)
        for g in patterns
    )


def _ignore_patterns(root: Path, extra: Iterable[str]) -> list[str]:
    pats = list(extra)
    f = root / IGNORE_FILE if root.is_dir() else root.parent / IGNORE_FILE
    if f.is_file():
        pats += [
            ln.strip()
            for ln in f.read_text(errors="replace").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    return pats


def iter_files(
    paths: Iterable[str], include: Iterable[str] = (), exclude: Iterable[str] = ()
) -> Iterator[Path]:
    """Files to scan. Explicit files are always scanned; directories are walked for text files."""
    for raw in paths:
        root = Path(raw)
        patterns = _ignore_patterns(root, exclude)
        if root.is_file():
            if not _ignored(root.as_posix(), patterns):
                yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in sorted(filenames):
                p = Path(dirpath, name)
                rel = p.relative_to(root).as_posix()
                if is_text_file(p, include) and not _ignored(rel, patterns):
                    yield p


def changed_files(base: str, include: Iterable[str] = ()) -> list[Path]:
    """Text files added or modified since ``base`` (a git ref), for pull-request scans."""
    out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AMR", base],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        Path(ln)
        for ln in out.splitlines()
        if ln and Path(ln).is_file() and is_text_file(Path(ln), include)
    ]


def read_text(path: Path) -> Optional[str]:
    """File contents, or None for binary or oversized files."""
    if path.stat().st_size > MAX_BYTES:
        return None
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def scan_files(files: Iterable[Path], clf) -> tuple[int, list[Finding]]:
    n, findings = 0, []
    for p in files:
        text = read_text(p)
        if text is None or not text.strip():
            continue
        n += 1
        findings += scan_text(text, clf, p.as_posix())
    return n, findings


def load_classifier(
    model: Optional[str] = None, threshold: Optional[float] = None, device: str = "cpu"
):
    """Load the classifier from a path, ``PROMPTSCAN_MODEL`` or the default Hugging Face repo."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    from .classifier import DEFAULT_REPO, Classifier

    model = model or os.environ.get("PROMPTSCAN_MODEL")
    try:
        clf = Classifier.load(model, device=device)
    except Exception as e:  # OSError from transformers/hub, or a bad local dir
        print(
            f"promptscan: could not load model '{model or DEFAULT_REPO}': {e}\n"
            "Pass --model DIR or set PROMPTSCAN_MODEL to a local model directory, "
            "or check network access to huggingface.co.",
            file=sys.stderr,
        )
        raise SystemExit(2) from e
    if threshold is not None:
        clf.threshold = threshold
    return clf


def to_sarif(findings: list[Finding], version: str) -> dict:
    rules = {
        "injection": "Text that reads as an instruction to an AI model rather than content for a human.",
        "hidden": "Invisible or direction-changing Unicode characters that can hide text from reviewers.",
    }
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "promptscan",
                        "version": version,
                        "informationUri": "https://github.com/0xdewy/promptscan",
                        "rules": [
                            {"id": k, "shortDescription": {"text": v}}
                            for k, v in rules.items()
                        ],
                    }
                },
                "results": [
                    {
                        "ruleId": f.kind,
                        "level": "error",
                        "message": {
                            "text": f.message
                            + (f" (score {f.score})" if f.score is not None else "")
                        },
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": f.path},
                                    "region": {
                                        "startLine": f.line,
                                        "snippet": {"text": f.snippet},
                                    },
                                }
                            }
                        ],
                    }
                    for f in findings
                ],
            }
        ],
    }


def to_json(files: int, findings: list[Finding], threshold: float) -> str:
    return json.dumps(
        {
            "files": files,
            "threshold": threshold,
            "findings": [asdict(f) for f in findings],
        },
        indent=1,
    )
