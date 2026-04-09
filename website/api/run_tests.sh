#!/bin/bash
# Script to start server and run tests

set -e

cd /home/user/code/prompt-scan/website/api

echo "Starting promptscan server..."
echo "============================================================"

# Start server in background
../../.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

echo "Server started with PID: $SERVER_PID"
echo "Waiting for server to initialize..."

# Wait for server to be ready
sleep 8

echo "============================================================"
echo "Running tests..."
echo "============================================================"

# Run the file upload test
echo "Running file upload test..."
../../.venv/bin/python test_file_upload.py

echo ""
echo "============================================================"
echo "Tests completed."
echo "Server is still running on http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo "============================================================"

# Wait for user to press Ctrl+C
wait $SERVER_PID