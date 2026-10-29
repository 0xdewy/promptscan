"""promptscan command line: scan text, files, directories, URLs or stdin for injection.

Exit status: 0 no blockers, 1 blocking findings (unless --no-fail), 2 error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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
    def display(value):
        # Repository content must not emit terminal controls or workflow commands.
        return json.dumps(value, ensure_ascii=True)[1:-1].replace("::", r"\:\:")

    label = (
        Colors.colored("INJECTION", Colors.RED)
        if f.kind == "injection"
        else Colors.colored("HIDDEN", Colors.YELLOW)
    )
    score = f"  {f.score:.2f}" if f.score is not None else ""
    print(f"{display(f.path)}:{f.line}  {label}{score}  {display(f.message)}")
    print(f"    {Colors.colored(display(f.snippet), Colors.DIM)}")
    if github:

        def escape(value, property=False):
            value = "".join(
                f"\\u{ord(char):04x}"
                if (ord(char) < 32 and char not in "\r\n") or ord(char) == 127
                else char
                for char in value
            )
            value = value.replace("%", "%25").replace("\r", "%0D")
            value = value.replace("\n", "%0A")
            return value.replace(",", "%2C").replace(":", "%3A") if property else value

        msg = escape(f"{f.message}: {f.snippet}")
        print(
            f"::{f.level} file={escape(f.path, True)},"
            f"line={f.line},title=promptscan::{msg}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="promptscan",
        description="Detect prompt injection in text, files, directories, URLs, stdin",
        epilog="""examples:
  promptscan "Ignore all previous instructions"
  promptscan .                          # prompt, code and config files
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
        "--changed", metavar="REF", help="report new findings against this git ref"
    )
    p.add_argument(
        "--all-findings",
        action="store_true",
        help="with --changed, also report existing findings in modified files",
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
    p.add_argument(
        "--model-revision", help="revision for an explicit Hub model override"
    )
    p.add_argument(
        "--fail-on",
        choices=("injection", "all"),
        default="injection",
        help="blocking findings; Unicode hygiene is a warning by default",
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
    if a.threshold is not None and not 0 < a.threshold <= 1:
        p.error("--threshold must be in (0, 1]")
    if a.all_findings and not a.changed:
        p.error("--all-findings requires --changed")

    texts: list[tuple[str, str]] = []  # (label, text)
    paths: list[str] = []
    for item in a.input:
        if a.changed:
            if item == "-" or item.startswith(("http://", "https://")):
                p.error("--changed accepts repository paths, not URLs or stdin")
            paths.append(item)
            continue
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

    skipped = []
    files = (
        [] if a.changed else list(S.iter_files(paths, a.include, a.exclude, skipped))
    )
    review = {}
    try:
        clf = S.load_classifier(a.model, a.threshold, revision=a.model_revision)
        if a.changed:
            from .review import scan_changes

            n, findings, review = scan_changes(
                a.changed, clf, a.include, a.exclude, paths, a.all_findings
            )
            skipped.extend(review.pop("skipped"))
        else:
            files = list(dict.fromkeys(files))
            n, findings = S.scan_files(files, clf, skipped)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"promptscan: scan failed: {error}", file=sys.stderr)
        return 2
    for label, text in texts:
        if text.strip():
            n += 1
            findings += S.scan_text(text, clf, label)

    blocking = sum(f.kind == "injection" or a.fail_on == "all" for f in findings)
    metadata = {
        "coverage": {"scanned_inputs": n, "skipped": skipped},
        "review": review,
        "blocking_findings": blocking,
        "release": {
            "version": __version__,
            "scanner_policy": S.SCANNER_POLICY_VERSION,
            "model": getattr(clf, "identity", None),
            "threshold": clf.threshold,
        },
    }

    if a.sarif:
        Path(a.sarif).write_text(
            json.dumps(S.to_sarif(findings, __version__, metadata["release"]), indent=1)
        )
    if a.json:
        output = S.to_json(n, findings, clf.threshold, **metadata)
        # Escaping these characters preserves JSON values while preventing
        # untrusted strings from becoming runner commands in stdout.
        print(
            output.replace("::", r"\u003a\u003a")
            if os.environ.get("GITHUB_ACTIONS")
            else output
        )
    else:
        github = bool(os.environ.get("GITHUB_ACTIONS"))
        for f in findings:
            _print(f, github)
        if not a.quiet:
            if not n:
                verdict = Colors.colored("no matching inputs", Colors.YELLOW)
            elif findings:
                verdict = f"{blocking} blocking, {len(findings) - blocking} warning(s)"
            else:
                verdict = Colors.colored("clean", Colors.GREEN)
            print(
                f"promptscan: {n} input(s) scanned, {verdict} "
                f"(threshold {clf.threshold:g}; {len(skipped)} skipped)"
            )
    return 1 if blocking and not a.no_fail else 0


if __name__ == "__main__":
    sys.exit(main())
