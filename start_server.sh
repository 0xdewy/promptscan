#!/bin/bash

# Simple script to start Prompt Detective production server

cd "$(dirname "$0")"

echo "Starting Prompt Detective Production Server..."
echo "=========================================="

# Check if dependencies are installed
if ! uv run python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    uv pip install fastapi uvicorn pydantic python-multipart
fi

# Start the server
echo ""
echo "Starting server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

cd website/api
uv run python main.py