#!/bin/bash

# Prompt Detective - Simple Free Version
# Starts the FastAPI server without x402 payment

set -e

# Change to website directory
cd "$(dirname "$0")"

echo "Starting Prompt Detective (Free Version)..."
echo "=========================================="

# Check if Python dependencies are installed
echo "Checking dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing simple dependencies..."
    cd ..
    if ! uv pip install -r website/requirements_simple.txt; then
        echo "Failed to install dependencies. Trying alternative method..."
        pip install fastapi uvicorn pydantic python-multipart
    fi
    cd website
fi

# Start the server
echo ""
echo "Starting FastAPI server on http://localhost:8000"
echo "  - Frontend: http://localhost:8000/"
echo "  - API Docs: http://localhost:8000/api/docs"
echo "  - Health: http://localhost:8000/api/v1/health"
echo ""
echo "Service is COMPLETELY FREE - No payment required!"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the server
cd ..
echo "Starting server from project root..."
uv run uvicorn website.api.main:app --host 0.0.0.0 --port 8000 --reload