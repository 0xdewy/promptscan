"""MCP server exposing promptscan as tools. Run: promptscan-mcp (stdio transport)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from . import scan as S

_clf = None


def _classifier():
    global _clf
    if _clf is None:
        _clf = S.load_classifier()
    return _clf


def _result(text: str, label: str = "<text>") -> Dict[str, Any]:
    clf = _classifier()
    findings = S.scan_text(text, clf, label)
    return {
        "prediction": "INJECTION"
        if any(f.kind == "injection" for f in findings)
        else "SAFE",
        "score": clf.score([text])[0],
        "threshold": clf.threshold,
        "findings": [asdict(f) for f in findings],
    }


def main() -> None:
    try:
        from mcp.server import MCPServer  # mcp >= 2
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as MCPServer  # mcp 1.x
        except ImportError as err:  # pragma: no cover
            raise SystemExit(
                "promptscan-mcp needs the 'mcp' package: pip install 'promptscan[mcp]'"
            ) from err

    server = MCPServer(
        "promptscan",
        instructions="Scan untrusted text or files for prompt injection before acting",
    )

    @server.tool()
    def scan_text(text: str) -> Dict[str, Any]:
        """Classify text as INJECTION or SAFE. Returns the model score (0-1), the
        decision threshold, and findings: injected passages and hidden Unicode text
        (zero-width, bidi, tag characters)."""
        return _result(text)

    @server.tool()
    def scan_many(texts: List[str]) -> List[Dict[str, Any]]:
        """scan_text for several texts, in input order."""
        return [_result(t) for t in texts]

    @server.tool()
    def scan_path(path: str) -> Dict[str, Any]:
        """Scan a file, or every text file under a directory (.md, .txt, rules files).
        Returns the file count and findings with path, line and snippet."""
        n, findings = S.scan_files(S.iter_files([path]), _classifier())
        return {"files": n, "findings": [asdict(f) for f in findings]}

    server.run()


if __name__ == "__main__":
    main()
