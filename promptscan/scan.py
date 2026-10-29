"""Scan text and files for prompt injection.

Two detectors run on every input:

- the classifier (``promptscan.classifier.Classifier``), which scores the whole text,
  contextual views of HTML comments, and decoded Unicode tag text in context;
- a deterministic check for invisible text: zero-width characters, bidi overrides and
  Unicode tag characters. These are hygiene warnings even when the model is unsure.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional
from urllib.parse import quote

TEXT_SUFFIXES = {
    ".md",
    ".markdown",
    ".mdc",
    ".txt",
    ".rst",
    ".adoc",
    ".prompt",
    ".rules",
    ".py",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
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
    ".pytest_cache",
    ".cache",
}
IGNORE_FILE = ".promptscanignore"
MAX_BYTES = 1_000_000
SCANNER_POLICY_VERSION = "document-context-2"

_HIDDEN = re.compile(
    "[\u200b\u2060-\u2064\ufeff\u202a-\u202e\u2066-\u2069\U000e0000-\U000e007f]+"
)
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
    fingerprint: Optional[str] = None

    @property
    def level(self) -> str:
        return "error" if self.kind == "injection" else "warning"


def fingerprint(kind: str, evidence: str) -> str:
    value = kind + "\0" + evidence.strip("\r\n")
    return hashlib.sha256(value.encode()).hexdigest()


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _decode_tags(run: str) -> str:
    return "".join(chr(ord(c) - 0xE0000) for c in run if 0xE0020 <= ord(c) <= 0xE007E)


def hidden_text(text: str, path: str = "<text>") -> list[Finding]:
    """Find invisible characters, grouped per run; allow a leading BOM."""
    out = []
    for m in _HIDDEN.finditer(text):
        if m.start() == 0 and m.group() == "\ufeff":
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
                fingerprint=fingerprint("hidden", run),
            )
        )
    return out


def _snippet(s: str, n: int = 160) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "..."


def classifier_segments(
    text: str, findings: list[Finding]
) -> list[tuple[str, int, str]]:
    """Inputs scored for a file, shared by scanning and detector training."""
    # A comment excerpt can lose the fact that it belongs to a quoted fixture.
    # Keep complete documents; the model's window encoder carries their frame.
    segments = []
    # Decode each tag run in its own surrounding context. Never join unrelated
    # runs into a fabricated instruction, and never truncate the decoded payload.
    for match in _HIDDEN.finditer(text):
        decoded = _decode_tags(match.group())
        if decoded.strip():
            segment = text[: match.start()] + decoded + text[match.end() :]
            segments.append((segment, 1, " (hidden in Unicode tags)"))
    if not any(segment == text for segment, _, _ in segments):
        segments.append((text, 1, ""))
    return segments


def scan_text(
    text: str, clf, path: str = "<text>", *, decision: Optional[dict] = None
) -> list[Finding]:
    """Find hidden text and injections; locate the highest-scoring paragraph."""
    findings = hidden_text(text, path)
    segments = classifier_segments(text, findings)

    scores = clf.score([s for s, _, _ in segments])
    best = max(range(len(segments)), key=scores.__getitem__)
    if decision is not None:
        decision["score"] = scores[best]
    if scores[best] < clf.threshold:
        return findings

    seg, line, where = segments[best]
    paragraphs = [(m, p) for m, p in _spans(seg) if p.strip()]
    if len(paragraphs) > 1:
        pscores = clf.score([p for _, p in paragraphs])
        start, para = paragraphs[max(range(len(paragraphs)), key=pscores.__getitem__)]
    else:
        start, para = 0, seg
    if not where and ("<!--" in para or "-->" in para):
        where = " (near an HTML comment)"
    findings.append(
        Finding(
            path,
            line + seg.count("\n", 0, start),
            "injection",
            f"prompt injection{where}",
            _snippet(para),
            round(scores[best], 4),
            fingerprint("injection", para),
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
    paths: Iterable[str],
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
    skipped: Optional[list[dict]] = None,
) -> Iterator[Path]:
    """Walk supported text files; explicit files bypass extension filtering."""
    for raw in paths:
        root = Path(raw)
        patterns = _ignore_patterns(root, exclude)
        if root.is_symlink():
            if skipped is not None:
                skipped.append({"path": str(root), "reason": "symlink"})
            continue
        if root.is_file():
            if not _ignored(root.as_posix(), patterns):
                yield root
            elif skipped is not None:
                skipped.append({"path": str(root), "reason": "excluded"})
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            kept = []
            for name in sorted(dirnames):
                path = Path(dirpath, name)
                if name in SKIP_DIRS or path.is_symlink():
                    if skipped is not None:
                        skipped.append({"path": str(path), "reason": "directory"})
                else:
                    kept.append(name)
            dirnames[:] = kept
            for name in sorted(filenames):
                p = Path(dirpath, name)
                rel = p.relative_to(root).as_posix()
                reason = (
                    "symlink"
                    if p.is_symlink()
                    else "excluded"
                    if _ignored(rel, patterns)
                    else "unsupported_type"
                    if not is_text_file(p, include)
                    else None
                )
                if reason is None:
                    yield p
                elif skipped is not None:
                    skipped.append({"path": str(p), "reason": reason})


def changed_files(
    base: str, include: Iterable[str] = (), exclude: Iterable[str] = ()
) -> list[Path]:
    """Text files added or modified since ``base`` (a git ref), for pull-request scans.

    Honours ``exclude`` and the ``.promptscanignore`` in the current directory, like a
    directory scan does.
    """
    out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AMR", base],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    patterns = _ignore_patterns(Path("."), exclude)
    return [
        Path(ln)
        for ln in out.splitlines()
        if ln
        and not Path(ln).is_symlink()
        and Path(ln).is_file()
        and is_text_file(Path(ln), include)
        and not _ignored(ln, patterns)
    ]


def read_text(path: Path) -> Optional[str]:
    """File contents, or None for binary or oversized files."""
    if path.is_symlink() or path.stat().st_size > MAX_BYTES:
        return None
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def scan_files(
    files: Iterable[Path], clf, skipped: Optional[list[dict]] = None
) -> tuple[int, list[Finding]]:
    n, findings = 0, []
    for p in files:
        text = read_text(p)
        if text is None or not text.strip():
            if skipped is not None:
                reason = (
                    "symlink"
                    if p.is_symlink()
                    else "oversized"
                    if p.stat().st_size > MAX_BYTES
                    else "binary"
                    if text is None
                    else "empty"
                )
                skipped.append({"path": str(p), "reason": reason})
            continue
        n += 1
        findings += scan_text(text, clf, p.as_posix())
    return n, findings


def load_classifier(
    model: Optional[str] = None,
    threshold: Optional[float] = None,
    device: str = "cpu",
    revision: Optional[str] = None,
):
    """Load from a path, ``PROMPTSCAN_MODEL``, or the default model repo."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    from .classifier import DEFAULT_REPO, Classifier

    model = model or os.environ.get("PROMPTSCAN_MODEL")
    try:
        clf = Classifier.load(model, device=device, revision=revision)
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


def to_sarif(findings: list[Finding], version: str, release=None) -> dict:
    rules = {
        "injection": (
            "Suspected attempt to redirect an agent beyond the authorized task."
        ),
        "hidden": (
            "Invisible or direction-changing Unicode characters "
            "that can hide text from reviewers."
        ),
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
                "properties": {"release": release},
                "results": [
                    {
                        "ruleId": f.kind,
                        "level": f.level,
                        "partialFingerprints": {
                            "promptscan/v1": f.fingerprint
                            or fingerprint(f.kind, f.snippet)
                        },
                        "message": {
                            "text": f.message
                            + (f" (score {f.score})" if f.score is not None else "")
                        },
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {
                                        "uri": Path(f.path).as_uri()
                                        if Path(f.path).is_absolute()
                                        else quote(
                                            f.path.replace(os.sep, "/"), safe="/"
                                        )
                                    },
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


def to_json(files: int, findings: list[Finding], threshold: float, **metadata) -> str:
    return json.dumps(
        {
            "files": files,
            "threshold": threshold,
            "findings": [asdict(f) | {"level": f.level} for f in findings],
            **metadata,
        },
        indent=1,
    )
