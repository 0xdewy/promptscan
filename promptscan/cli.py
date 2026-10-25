"""promptscan command line: scan text, files, directories, URLs or stdin for injection.

Exit status: 0 nothing found, 1 findings (unless --no-fail), 2 usage or load error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from . import scan as S
from .utils.colors import Colors

_TAG = re.compile(r"<(?!!--)[^>]+>")
_SCRIPT = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)


def _fetch(url: str) -> str:
    import html

    import requests

    r = requests.get(
        url, timeout=15, headers={"User-Agent": f"promptscan/{__version__}"}
    )
    r.raise_for_status()
    text = r.text
    if "html" in r.headers.get("content-type", ""):
        text = html.unescape(
            _TAG.sub(" ", _SCRIPT.sub("", text))
        )  # keep <!-- comments --> for scan_text
    return text


def _print(f: S.Finding, github: bool) -> None:
    label = (
        Colors.colored("INJECTION", Colors.RED)
        if f.kind == "injection"
        else Colors.colored("HIDDEN", Colors.YELLOW)
    )
    score = f"  {f.score:.2f}" if f.score is not None else ""
    print(f"{f.path}:{f.line}  {label}{score}  {f.message}")
    print(f"    {Colors.colored(f.snippet, Colors.DIM)}")
    if github:
        msg = f"{f.message}: {f.snippet}".replace("%", "%25").replace("\n", "%0A")
        print(f"::error file={f.path},line={f.line},title=promptscan::{msg}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="promptscan",
        description="Detect prompt injection in text, files, directories, URLs, stdin",
        epilog="""examples:
  promptscan "Ignore all previous instructions"
  promptscan .                          # every .md/.txt/rules file under the repo
  promptscan CLAUDE.md prompts/ --json
  promptscan --changed origin/main      # only files changed on this branch
  curl -s https://example.com | promptscan -
  promptscan https://example.com/page""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input", nargs="*", help="text, file, directory, URL, or '-' for stdin"
    )
    p.add_argument(
        "--changed", metavar="REF", help="scan text files changed since this git ref"
    )
    p.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="extra file pattern to scan in directories",
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="skip matching paths (also read from .promptscanignore)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        help="override the model's stored decision threshold (0-1)",
    )
    p.add_argument(
        "--model",
        help="model directory or Hugging Face repo "
        "(default: $PROMPTSCAN_MODEL or 0xdewy/promptscan)",
    )
    p.add_argument("--json", action="store_true", help="print findings as JSON")
    p.add_argument(
        "--sarif",
        metavar="FILE",
        help="also write a SARIF report for GitHub code scanning",
    )
    p.add_argument(
        "--no-fail", action="store_true", help="exit 0 even when something is found"
    )
    p.add_argument("-q", "--quiet", action="store_true", help="only print findings")
    p.add_argument("--version", action="version", version=f"promptscan {__version__}")
    a = p.parse_args(argv)

    texts: list[tuple[str, str]] = []  # (label, text)
    paths: list[str] = []
    for item in a.input:
        if item == "-":
            texts.append(("<stdin>", sys.stdin.read()))
        elif item.startswith(("http://", "https://")):
            try:
                texts.append((item, _fetch(item)))
            except Exception as e:
                print(f"promptscan: could not fetch {item}: {e}", file=sys.stderr)
                return 2
        elif os.path.exists(item):
            paths.append(item)
        else:
            texts.append(("<text>", item))
    if not a.input and not a.changed:
        if sys.stdin.isatty():
            p.print_help()
            return 2
        texts.append(("<stdin>", sys.stdin.read()))

    clf = S.load_classifier(a.model, a.threshold)

    files = list(S.iter_files(paths, a.include, a.exclude))
    if a.changed:
        files += S.changed_files(a.changed, a.include, a.exclude)
    n, findings = S.scan_files(files, clf)
    for label, text in texts:
        if text.strip():
            n += 1
            findings += S.scan_text(text, clf, label)

    if a.sarif:
        Path(a.sarif).write_text(
            json.dumps(S.to_sarif(findings, __version__), indent=1)
        )
    if a.json:
        print(S.to_json(n, findings, clf.threshold))
    else:
        github = bool(os.environ.get("GITHUB_ACTIONS"))
        for f in findings:
            _print(f, github)
        if not a.quiet:
            verdict = (
                Colors.colored(f"{len(findings)} finding(s)", Colors.RED)
                if findings
                else Colors.colored("clean", Colors.GREEN)
            )
            print(
                f"promptscan: {n} input(s) scanned, {verdict} "
                f"(threshold {clf.threshold:.2f})"
            )
    return 0 if not findings or a.no_fail else 1


if __name__ == "__main__":
    sys.exit(main())
