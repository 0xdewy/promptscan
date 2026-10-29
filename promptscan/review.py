"""Compare a PR with its base using one scanner and the base's ignore policy."""

from __future__ import annotations

import difflib
import subprocess
from collections import Counter
from pathlib import Path

from . import scan as S


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def base_text(commit: str, path: str) -> str | None:
    spec = f"{commit}:{path}"
    try:
        if int(git("cat-file", "-s", spec)) > S.MAX_BYTES:
            return None
        data = git("show", spec)
    except subprocess.CalledProcessError:
        return None
    return None if b"\0" in data[:8192] else data.decode("utf-8", errors="replace")


def key(finding):
    return finding.kind, finding.fingerprint or S.fingerprint(
        finding.kind, finding.snippet
    )


def changes(commit: str, root: Path | None = None):
    fields = (
        git(
            "-C",
            str(root or Path.cwd()),
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--name-status",
            "-z",
            "--find-renames",
            "--diff-filter=AMR",
            commit,
            "--",
        )
        .decode("utf-8", errors="surrogateescape")
        .split("\0")
    )
    fields = iter(fields[:-1])
    for status in fields:
        old = next(fields)
        if status.startswith("R"):
            yield next(fields), old
        else:
            yield old, old if status != "A" else None


def scan_changes(base, clf, include=(), exclude=(), paths=(), all_findings=False):
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    scopes = [Path(path).absolute() for path in paths]
    commit = git("rev-parse", "--verify", "--end-of-options", base + "^{commit}")
    commit = commit.decode().strip()
    ignore = base_text(commit, S.IGNORE_FILE) or ""
    patterns = list(exclude) + [
        line.strip()
        for line in ignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    skipped, findings, existing = [], [], 0
    scanned = 0
    for name, old_name in changes(commit, root):
        path = root / name
        if scopes and not any(
            path == scope or scope in path.parents for scope in scopes
        ):
            skipped.append({"path": name, "reason": "outside_requested_paths"})
            continue
        reason = (
            "symlink"
            if path.is_symlink()
            else "generated_directory"
            if any(part in S.SKIP_DIRS for part in Path(name).parts[:-1])
            else "excluded_by_base_policy"
            if S._ignored(name, patterns)
            else "unsupported_type"
            if not S.is_text_file(path, include)
            else None
        )
        if reason:
            skipped.append({"path": name, "reason": reason})
            continue
        head = S.read_text(path)
        if head is None or not head.strip():
            skipped.append({"path": name, "reason": "binary_oversized_or_empty"})
            continue
        before = base_text(commit, old_name) if old_name else ""
        if before is None:
            # A previously unscannable blob is not a clean baseline.
            before = ""
        scanned += 1
        old_findings = S.scan_text(before, clf, name) if before.strip() else []
        known = Counter(key(f) for f in old_findings)
        head_findings = S.scan_text(head, clf, name)
        if all_findings:
            findings.extend(head_findings)
            continue
        new_injection = False
        for finding in head_findings:
            if known[key(finding)]:
                known[key(finding)] -= 1
                existing += 1
            else:
                findings.append(finding)
                new_injection |= finding.kind == "injection"

        # A clear whole-file verdict must not be overturned by a fragment that
        # lost its fixture context. Supplement only a known finding shadowing edits.
        if new_injection or not any(f.kind == "injection" for f in old_findings):
            continue
        if not any(f.kind == "injection" for f in head_findings):
            continue

        old_lines, lines = before.splitlines(keepends=True), head.splitlines(True)
        edits = [
            (a, b)
            for tag, _, _, a, b in difflib.SequenceMatcher(
                None, old_lines, lines, autojunk=False
            ).get_opcodes()
            if tag != "equal" and a != b
        ]
        if not edits:
            continue
        # A new lower-scoring payload must not be hidden by a known, stronger
        # finding elsewhere in the file. Mask unchanged evidence in a complete
        # document view, keeping its fixture/syntax context and source line numbers.
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        masked = list(head)
        old_keys = Counter(key(f) for f in old_findings if f.kind == "injection")
        for start, paragraph in S._spans(head):
            ident = ("injection", S.fingerprint("injection", paragraph))
            end = start + len(paragraph)
            edited = any(start < offsets[b] and end > offsets[a] for a, b in edits)
            if not edited and old_keys[ident]:
                old_keys[ident] -= 1
                masked[start:end] = ["\n" if c == "\n" else " " for c in paragraph]
        masked = "".join(masked)
        while masked.strip():
            extra = S.scan_text(masked, clf, name)
            edited_findings = [
                f
                for f in extra
                if any(first + 1 <= f.line <= last for first, last in edits)
            ]
            findings.extend(edited_findings)
            if any(f.kind == "injection" for f in edited_findings):
                break
            remaining = next((f for f in extra if f.kind == "injection"), None)
            if remaining is None:
                break
            previous = masked
            for start, paragraph in S._spans(masked):
                first = S._line_of(masked, start)
                if first <= remaining.line <= first + paragraph.count("\n"):
                    end = start + len(paragraph)
                    if any(start < offsets[b] and end > offsets[a] for a, b in edits):
                        findings.append(remaining)
                    else:
                        blank = "".join("\n" if c == "\n" else " " for c in paragraph)
                        masked = masked[:start] + blank + masked[end:]
                    break
            if masked == previous:
                break
    unique = {(f.path, f.line, *key(f)): f for f in findings}
    return (
        scanned,
        list(unique.values()),
        {
            "base_commit": commit,
            "existing_findings": existing,
            "skipped": skipped,
            "ignore_policy": "base branch",
        },
    )
