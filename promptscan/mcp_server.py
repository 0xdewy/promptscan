"""MCP server exposing promptscan as a tool. Run: promptscan-mcp (stdio transport)."""
from __future__ import annotations

from typing import Any, Dict, List

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        from .unified_detector import UnifiedDetector

        _detector = UnifiedDetector(model_type="ensemble")
    return _detector


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as err:  # pragma: no cover
        raise SystemExit("promptscan-mcp needs the 'mcp' package: pip install 'promptscan[mcp]'") from err

    server = FastMCP("promptscan", instructions="Scan text for prompt injection before passing it to an LLM.")

    @server.tool()
    def scan_text(text: str) -> Dict[str, Any]:
        """Classify text as INJECTION or SAFE using promptscan's CNN+LSTM+Transformer ensemble.

        Returns prediction, confidence (0-1) and each model's vote.
        """
        r = _get_detector().predict(text)
        return {
            "prediction": r["prediction"],
            "confidence": r.get("confidence"),
            "individual_predictions": r.get("individual_predictions", {}),
        }

    @server.tool()
    def scan_many(texts: List[str]) -> List[Dict[str, Any]]:
        """Classify several texts at once. Same fields as scan_text, in input order."""
        out = []
        for r in _get_detector().predict_batch(texts):
            out.append({"prediction": r.get("prediction"), "confidence": r.get("confidence"),
                        "individual_predictions": r.get("individual_predictions", {})})
        return out

    server.run()


if __name__ == "__main__":
    main()
