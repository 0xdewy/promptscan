#!/usr/bin/env python3
"""
Text extraction utilities for different file types.
"""

import os
import re
from pathlib import Path
from typing import Optional


class TextExtractor:
    """Base class for text extraction from different file types."""

    # File extensions for different types
    MARKDOWN_EXTS = {".md", ".markdown", ".mdown", ".mkd", ".mkdn"}
    CODE_EXTS = {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".cs",
        ".fs",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".bat",
        ".cmd",
    }
    CONFIG_EXTS = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf"}
    DOCUMENTATION_EXTS = {".rst", ".txt", ".text", ".adoc", ".asciidoc"}
    WEB_EXTS = {".html", ".htm", ".xml", ".xhtml"}

    # Combined set of all supported extensions
    SUPPORTED_EXTS = (
        MARKDOWN_EXTS | CODE_EXTS | CONFIG_EXTS | DOCUMENTATION_EXTS | WEB_EXTS
    )

    def __init__(self):
        """Initialize text extractor."""
        pass

    def get_file_type(self, file_path: str) -> str:
        """
        Determine file type from extension.

        Args:
            file_path: Path to file

        Returns:
            File type: "markdown", "code", "config", "documentation", "web", or "unknown"
        """
        ext = Path(file_path).suffix.lower()

        if ext in self.MARKDOWN_EXTS:
            return "markdown"
        elif ext in self.CODE_EXTS:
            return "code"
        elif ext in self.CONFIG_EXTS:
            return "config"
        elif ext in self.DOCUMENTATION_EXTS:
            return "documentation"
        elif ext in self.WEB_EXTS:
            return "web"
        else:
            return "unknown"

    def extract_text(self, file_path: str, content: str) -> Optional[str]:
        """
        Extract text from file content based on file type.

        Args:
            file_path: Path to file (for type detection)
            content: File content as string

        Returns:
            Extracted text or None if extraction failed
        """
        if not content or not content.strip():
            return None

        file_type = self.get_file_type(file_path)

        try:
            if file_type == "markdown":
                return self.extract_from_markdown(content)
            elif file_type == "code":
                return self.extract_from_code(content, file_path)
            elif file_type == "config":
                return self.extract_from_config(content, file_path)
            elif file_type == "documentation":
                return self.extract_from_documentation(content, file_path)
            elif file_type == "web":
                return self.extract_from_web(content)
            else:
                # For unknown types, try to extract as plain text
                return self.extract_plain_text(content)
        except Exception as e:
            print(f"⚠️  Error extracting text from {file_path}: {e}")
            return None

    def extract_from_markdown(self, content: str) -> str:
        """
        Extract text from markdown content.

        Args:
            content: Markdown content

        Returns:
            Plain text extracted from markdown
        """
        # Try to use the markdown parser if available
        try:
            from .utils.markdown_parser import parse_markdown_to_text

            return parse_markdown_to_text(content, use_library=True)
        except ImportError:
            # Fall back to simple regex-based extraction
            pass

        # Simple markdown stripping
        text = content

        # Remove code blocks but keep the content
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

        # Remove inline code markers
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Remove headers but keep text
        text = re.sub(r"^#+\s+(.*?)\s*#*$", r"\1", text, flags=re.MULTILINE)

        # Remove emphasis markers
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"__(.*?)__", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        text = re.sub(r"_(.*?)_", r"\1", text)

        # Remove links but keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove images
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

        # Remove blockquotes
        text = re.sub(r"^>\s+(.*)$", r"\1", text, flags=re.MULTILINE)

        # Clean up whitespace
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = text.strip()

        return text

    def extract_from_code(self, content: str, file_path: str) -> str:
        """
        Extract comments and docstrings from code files.

        Args:
            content: Code content
            file_path: Path to code file (for language detection)

        Returns:
            Extracted comments and docstrings
        """
        ext = Path(file_path).suffix.lower()

        if ext == ".py":
            return self._extract_python_comments(content)
        elif ext in {
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".go",
            ".rs",
            ".cs",
        }:
            return self._extract_cstyle_comments(content)
        elif ext in {".sh", ".bash", ".zsh", ".fish", ".ps1", ".rb", ".pl", ".pm"}:
            return self._extract_shell_comments(content)
        else:
            # Generic extraction for other languages
            return self._extract_generic_comments(content)

    def _extract_python_comments(self, content: str) -> str:
        """Extract comments and docstrings from Python code."""
        lines = []
        in_docstring = False
        docstring_type = None  # ' or "

        for line in content.split("\n"):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Handle docstrings
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if not in_docstring:
                    in_docstring = True
                    docstring_type = stripped[:3]
                    # Check if it's a one-line docstring
                    if stripped.endswith(docstring_type) and len(stripped) > 6:
                        docstring_content = stripped[3:-3].strip()
                        if docstring_content:
                            lines.append(docstring_content)
                        in_docstring = False
                else:
                    if stripped.endswith(docstring_type):
                        in_docstring = False
                continue

            if in_docstring:
                lines.append(stripped)
                continue

            # Extract single-line comments
            if stripped.startswith("#"):
                comment = stripped[1:].strip()
                if comment:
                    lines.append(comment)

        return "\n".join(lines)

    def _extract_cstyle_comments(self, content: str) -> str:
        """Extract comments from C-style languages (// and /* */)."""
        lines = []
        in_block_comment = False
        block_comment_buffer = []

        for line in content.split("\n"):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            i = 0
            while i < len(stripped):
                if not in_block_comment and stripped.startswith("//", i):
                    # Single-line comment
                    comment = stripped[i + 2 :].strip()
                    if comment:
                        lines.append(comment)
                    break
                elif not in_block_comment and stripped.startswith("/*", i):
                    # Start of block comment
                    in_block_comment = True
                    i += 2
                    # Check if it ends on the same line
                    end_idx = stripped.find("*/", i)
                    if end_idx != -1:
                        comment = stripped[i:end_idx].strip()
                        if comment:
                            lines.append(comment)
                        in_block_comment = False
                        i = end_idx + 2
                    else:
                        block_comment_buffer.append(stripped[i:].strip())
                        break
                elif in_block_comment:
                    # Inside block comment
                    end_idx = stripped.find("*/", i)
                    if end_idx != -1:
                        block_comment_buffer.append(stripped[i:end_idx].strip())
                        comment = " ".join(filter(None, block_comment_buffer))
                        if comment:
                            lines.append(comment)
                        in_block_comment = False
                        block_comment_buffer = []
                        i = end_idx + 2
                    else:
                        block_comment_buffer.append(stripped[i:].strip())
                        break
                else:
                    # Not a comment, skip this character
                    i += 1

        return "\n".join(lines)

    def _extract_shell_comments(self, content: str) -> str:
        """Extract comments from shell scripts (#)."""
        lines = []

        for line in content.split("\n"):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Extract comments
            if stripped.startswith("#"):
                comment = stripped[1:].strip()
                if comment:
                    lines.append(comment)

        return "\n".join(lines)

    def _extract_generic_comments(self, content: str) -> str:
        """Generic comment extraction for unknown languages."""
        lines = []

        for line in content.split("\n"):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Try common comment patterns
            if stripped.startswith("#") or stripped.startswith("//"):
                comment = (
                    stripped[1:].strip()
                    if stripped.startswith("#")
                    else stripped[2:].strip()
                )
                if comment:
                    lines.append(comment)
            elif stripped.startswith("/*") and stripped.endswith("*/"):
                comment = stripped[2:-2].strip()
                if comment:
                    lines.append(comment)
            elif stripped.startswith("<!--") and stripped.endswith("-->"):
                comment = stripped[4:-3].strip()
                if comment:
                    lines.append(comment)

        return "\n".join(lines)

    def extract_from_config(self, content: str, file_path: str) -> str:
        """
        Extract text from configuration files.

        Args:
            content: Configuration file content
            file_path: Path to config file

        Returns:
            Extracted configuration text
        """
        ext = Path(file_path).suffix.lower()

        if ext in {".yaml", ".yml"}:
            return self._extract_yaml_comments(content)
        elif ext == ".json":
            return self._extract_json_content(content)
        elif ext == ".toml":
            return self._extract_toml_comments(content)
        elif ext in {".ini", ".cfg", ".conf"}:
            return self._extract_ini_comments(content)
        else:
            return self.extract_plain_text(content)

    def _extract_yaml_comments(self, content: str) -> str:
        """Extract comments from YAML files."""
        lines = []

        for line in content.split("\n"):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Extract comments
            if stripped.startswith("#"):
                comment = stripped[1:].strip()
                if comment:
                    lines.append(comment)
            else:
                # Also include key names (before colon)
                if ":" in stripped:
                    key = stripped.split(":", 1)[0].strip()
                    if key and not key.startswith("#"):
                        lines.append(f"{key}:")

        return "\n".join(lines)

    def _extract_json_content(self, content: str) -> str:
        """Extract meaningful content from JSON files."""
        # JSON doesn't have comments, but we can extract key names
        lines = []

        # Simple extraction of keys (naive approach)
        # Match keys in JSON (quoted strings followed by colon)
        key_pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"\s*:'
        keys = re.findall(key_pattern, content)

        for key in keys:
            lines.append(f"{key}:")

        return "\n".join(lines)

    def _extract_toml_comments(self, content: str) -> str:
        """Extract comments from TOML files."""
        lines = []

        for line in content.split("\n"):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Extract comments
            if stripped.startswith("#"):
                comment = stripped[1:].strip()
                if comment:
                    lines.append(comment)
            else:
                # Include section headers
                if stripped.startswith("[") and stripped.endswith("]"):
                    section = stripped[1:-1].strip()
                    if section:
                        lines.append(f"[{section}]")

        return "\n".join(lines)

    def _extract_ini_comments(self, content: str) -> str:
        """Extract comments from INI files."""
        lines = []

        for line in content.split("\n"):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Extract comments (; or #)
            if stripped.startswith(";") or stripped.startswith("#"):
                comment = stripped[1:].strip()
                if comment:
                    lines.append(comment)
            else:
                # Include section headers
                if stripped.startswith("[") and stripped.endswith("]"):
                    section = stripped[1:-1].strip()
                    if section:
                        lines.append(f"[{section}]")

        return "\n".join(lines)

    def extract_from_documentation(self, content: str, file_path: str) -> str:
        """
        Extract text from documentation files.

        Args:
            content: Documentation content
            file_path: Path to documentation file

        Returns:
            Extracted documentation text
        """
        ext = Path(file_path).suffix.lower()

        if ext == ".rst":
            return self._extract_rst_content(content)
        elif ext in {".adoc", ".asciidoc"}:
            return self._extract_asciidoc_content(content)
        else:
            # Plain text
            return self.extract_plain_text(content)

    def _extract_rst_content(self, content: str) -> str:
        """Extract text from reStructuredText."""
        # Simple RST extraction - remove directives and keep text
        text = content

        # Remove directives
        text = re.sub(
            r"^\.\.\s+.*?(\n\s*\n|\Z)", "", text, flags=re.MULTILINE | re.DOTALL
        )

        # Remove role markers
        text = re.sub(r":[\w-]+:`([^`]+)`", r"\1", text)

        # Clean up
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = text.strip()

        return text

    def _extract_asciidoc_content(self, content: str) -> str:
        """Extract text from AsciiDoc."""
        # Simple AsciiDoc extraction
        text = content

        # Remove block delimiters
        text = re.sub(r"^={4,}\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^-{4,}\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\*{4,}\s*$", "", text, flags=re.MULTILINE)

        # Remove inline formatting
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Clean up
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = text.strip()

        return text

    def extract_from_web(self, content: str) -> str:
        """
        Extract text from HTML/XML content.

        Args:
            content: HTML/XML content

        Returns:
            Extracted text
        """
        # Simple HTML/XML tag stripping
        text = content

        # Remove script and style tags
        text = re.sub(
            r"<script.*?>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r"<style.*?>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE
        )

        # Remove HTML comments
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

        # Remove all tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Decode HTML entities (simple ones)
        text = text.replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def extract_plain_text(self, content: str) -> str:
        """
        Extract plain text (minimal processing).

        Args:
            content: Text content

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r"\n\s*\n+", "\n\n", content)
        text = text.strip()

        return text

    def should_process_file(
        self, file_path: str, max_size: Optional[int] = None
    ) -> bool:
        """
        Check if a file should be processed based on extension and size.

        Args:
            file_path: Path to file
            max_size: Maximum file size in bytes (optional)

        Returns:
            True if file should be processed
        """
        ext = Path(file_path).suffix.lower()

        # Check extension
        if ext not in self.SUPPORTED_EXTS:
            return False

        # Check size if specified
        if max_size is not None:
            try:
                file_size = os.path.getsize(file_path)
                if file_size > max_size:
                    return False
            except (OSError, FileNotFoundError):
                # If we can't get size, assume it's OK
                pass

        return True


def test_text_extractor():
    """Test function for text extractor."""
    extractor = TextExtractor()

    # Test file type detection
    test_files = [
        ("README.md", "markdown"),
        ("script.py", "code"),
        ("config.yaml", "config"),
        ("docs.txt", "documentation"),
        ("index.html", "web"),
        ("unknown.xyz", "unknown"),
    ]

    for file_path, expected_type in test_files:
        detected = extractor.get_file_type(file_path)
        status = "✓" if detected == expected_type else "✗"
        print(f"{status} {file_path}: {detected} (expected: {expected_type})")

    # Test markdown extraction
    markdown_content = """
    # Title
    This is **bold** and *italic* text.
    
    ```python
    print("Hello")
    ```
    
    [Link text](http://example.com)
    """

    extracted = extractor.extract_from_markdown(markdown_content)
    print("\nMarkdown extraction test:")
    print(f"Original: {markdown_content[:100]}...")
    print(f"Extracted: {extracted[:100]}...")

    # Test Python comment extraction
    python_code = '''
    """Module docstring."""
    
    import os
    
    def function():
        """Function docstring."""
        # This is a comment
        return 42
    '''

    extracted = extractor._extract_python_comments(python_code)
    print("\nPython comment extraction test:")
    print(f"Extracted: {extracted}")


if __name__ == "__main__":
    test_text_extractor()
