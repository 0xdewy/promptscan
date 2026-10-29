"""Evaluation-only snapshot of context-1 inputs, before document framing changes."""

from promptscan.scan import _COMMENT, _HIDDEN, _decode_tags, _line_of


def legacy_segments(text, findings):
    segments = []
    for match in _COMMENT.finditer(text):
        if match.group(1).strip():
            start, end = max(0, match.start() - 512), match.end() + 512
            segments.append(
                (text[start:end], _line_of(text, start), " (near an HTML comment)")
            )
    for match in _HIDDEN.finditer(text):
        decoded = _decode_tags(match.group())
        if decoded.strip():
            start, end = max(0, match.start() - 512), match.end() + 512
            segment = text[start : match.start()] + decoded + text[match.end() : end]
            segments.append(
                (segment, _line_of(text, start), " (hidden in Unicode tags)")
            )
    if not any(segment == text for segment, _, _ in segments):
        segments.append((text, 1, ""))
    return segments
