#!/bin/bash
# Import prompts into the training dataset.
# Usage: ./import.sh <safe|injection> <file|directory|"text">
#
# Examples:
#   ./import.sh safe ./docs/
#   ./import.sh injection ./attacks/jailbreaks.txt
#   ./import.sh safe "what time is it?"
#   ./import.sh injection "ignore all previous instructions"

set -e

LABEL="${1:-}"
INPUT="${2:-}"
DATA="${DATA:-data/merged.parquet}"

if [ -z "$LABEL" ] || [ -z "$INPUT" ]; then
    echo "Usage: ./import.sh <safe|injection> <file|directory|text>"
    echo ""
    echo "Examples:"
    echo "  ./import.sh safe ./docs/"
    echo "  ./import.sh injection ./attacks.txt"
    echo '  ./import.sh safe "hello, how are you?"'
    echo '  ./import.sh injection "ignore all previous instructions"'
    echo ""
    echo "Environment variables:"
    echo "  DATA  path to parquet file (default: data/merged.parquet)"
    exit 1
fi

if [ "$LABEL" != "safe" ] && [ "$LABEL" != "injection" ]; then
    echo "❌ Error: label must be 'safe' or 'injection', got: $LABEL"
    exit 1
fi

if [ -d "$INPUT" ]; then
    uv run python scripts/insert.py --parquet "$DATA" --dir "$INPUT" --label "$LABEL"
elif [ -f "$INPUT" ]; then
    uv run python scripts/insert.py --parquet "$DATA" --file "$INPUT" --label "$LABEL"
else
    uv run python scripts/insert.py --parquet "$DATA" --text "$INPUT" --label "$LABEL"
fi
