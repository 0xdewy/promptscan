#!/usr/bin/env python3
"""
Markdown parsing utilities for Safe Prompts.
Provides functions to extract plain text from markdown files for prompt injection analysis.
"""

import re
from typing import Optional


def parse_markdown_to_text(markdown_content: str, use_library: bool = True) -> str:
    """
    Convert markdown content to plain text for analysis.

    This function extracts readable text from markdown while preserving
    important content like code blocks (which may contain prompts).

    Args:
        markdown_content: Raw markdown text
        use_library: Whether to try using the markdown library if available

    Returns:
        Plain text suitable for prompt injection analysis
    """
    # Try to use markdown library if available and requested
    if use_library:
        try:
            return _parse_with_markdown_library(markdown_content)
        except (ImportError, AttributeError, Exception):
            # Fall back to regex parsing
            pass

    # Use regex-based parsing as fallback
    return _parse_with_regex(markdown_content)


def _parse_with_markdown_library(markdown_content: str) -> str:
    """
    Parse markdown using the markdown library for better accuracy.

    This is the preferred method when the library is available.
    """
    import markdown

    # Simple approach: convert to HTML and strip tags
    html = markdown.markdown(markdown_content)

    # Strip HTML tags but preserve text
    text = re.sub(r"<[^>]+>", "", html).strip()

    # Also handle common HTML entities
    import html

    text = html.unescape(text)

    return text


def _parse_with_regex(markdown_content: str) -> str:
    """
    Parse markdown using regex patterns.

    This is a fallback method when the markdown library is not available.
    It handles common markdown syntax but may miss some edge cases.
    """
    text = markdown_content

    # Remove ATX headers (#, ##, etc.) but keep the text
    text = re.sub(r"^#+\s+(.*?)\s*#*$", r"\1", text, flags=re.MULTILINE)

    # Remove Setext headers (underlined with === or ---)
    text = re.sub(r"^(.+)\n=+$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"^(.+)\n-+$", r"\1", text, flags=re.MULTILINE)

    # Remove emphasis but keep text
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)

    # Italic: *text* or _text_
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)

    # Strikethrough: ~~text~~
    text = re.sub(r"~~(.*?)~~", r"\1", text)

    # Links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Images: ![alt](url) -> alt (or empty if no alt)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    # Inline code: `code` -> code (preserve as-is, important for prompts)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Block quotes: > text -> text
    text = re.sub(r"^>\s+(.*)$", r"\1", text, flags=re.MULTILINE)

    # Lists: * item, - item, + item, 1. item -> item
    text = re.sub(r"^[\*\-\+]\s+(.*)$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+(.*)$", r"\1", text, flags=re.MULTILINE)

    # Horizontal rules: ---, ***, ___ -> remove
    text = re.sub(r"^[\-\*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Handle code blocks specially - preserve content
    # Triple backtick code blocks
    def preserve_code_block(match):
        # Extract the code content and return it as-is
        code_content = match.group(1)
        return f"\n{code_content}\n"

    # Match code blocks with language specification
    text = re.sub(r"```\w*\n(.*?)\n```", preserve_code_block, text, flags=re.DOTALL)
    # Match code blocks without language specification
    text = re.sub(r"```\n(.*?)\n```", preserve_code_block, text, flags=re.DOTALL)

    # Indented code blocks (4 spaces or 1 tab)
    # We'll preserve these by not modifying lines that start with 4+ spaces or a tab
    # But we need to dedent them for readability
    lines = text.split("\n")
    processed_lines = []
    in_code_block = False

    for line in lines:
        # Check if line starts a code block (4+ spaces or tab)
        if re.match(r"^(?:\s{4,}|\t)", line):
            if not in_code_block:
                in_code_block = True
            # Remove the indentation but keep the line
            dedented = re.sub(r"^\s{4,}|\t", "", line, count=1)
            processed_lines.append(dedented)
        else:
            if in_code_block:
                in_code_block = False
            processed_lines.append(line)

    text = "\n".join(processed_lines)

    # Clean up extra whitespace
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)

    return text.strip()


def is_markdown_file(filename: str) -> bool:
    """
    Check if a file is likely a markdown file based on extension.

    Args:
        filename: The filename to check

    Returns:
        True if the file has a markdown extension
    """
    markdown_extensions = {
        ".md",
        ".markdown",
        ".mdown",
        ".mkd",
        ".mkdn",
        ".mdwn",
        ".mdt",
        ".mdtext",
    }
    return any(filename.lower().endswith(ext) for ext in markdown_extensions)


def read_and_parse_file(filepath: str, use_library: bool = True) -> str:
    """
    Read a file and parse it if it's markdown.

    Args:
        filepath: Path to the file
        use_library: Whether to use markdown library if available

    Returns:
        Parsed text content
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if is_markdown_file(filepath):
        return parse_markdown_to_text(content, use_library)

    return content.strip()


def get_file_type_display(filename: str) -> str:
    """
    Get a display string for the file type.

    Args:
        filename: The filename

    Returns:
        Display string like "Text", "Markdown", etc.
    """
    if is_markdown_file(filename):
        return "Markdown"
    elif filename.lower().endswith(".txt"):
        return "Text"
    else:
        # Try to infer from extension
        ext = filename[filename.rfind(".") :].lower() if "." in filename else ""
        if ext:
            return f"{ext[1:].upper()} file"
        return "File"


# Test the parser
if __name__ == "__main__":
    # Simple test
    test_markdown = """# Test Document
    
This is a **bold** statement and this is *italic*.

Here's some `inline code` and a [link](https://example.com).

```python
print("Hello, World!")
```

1. First item
2. Second item

> This is a block quote.

Another paragraph.
"""

    print("Original markdown:")
    print(test_markdown)
    print("\n" + "=" * 50 + "\n")

    print("Parsed text:")
    parsed = parse_markdown_to_text(test_markdown, use_library=False)
    print(parsed)

    print("\n" + "=" * 50 + "\n")
    print(f"Is 'README.md' a markdown file? {is_markdown_file('README.md')}")
    print(f"Is 'notes.txt' a markdown file? {is_markdown_file('notes.txt')}")
    print(f"Display for 'README.md': {get_file_type_display('README.md')}")
    print(f"Display for 'notes.txt': {get_file_type_display('notes.txt')}")
