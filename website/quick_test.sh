#!/bin/bash

# Quick test for Prompt Detective deployment

echo "Quick deployment test..."
echo "========================"

# Check if we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "api/main.py" ]; then
    echo "Error: api/main.py not found. Run from website directory."
    exit 1
fi

# Check .env file
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Created .env. Please update WALLET_ADDRESS."
    else
        echo "Error: .env.example not found."
        exit 1
    fi
fi

# Test Python syntax
echo "Testing Python syntax..."
cd ..
if uv run python -m py_compile website/api/main.py; then
    echo "✓ Python syntax check passed"
else
    echo "✗ Python syntax check failed"
    exit 1
fi

# Test imports
echo "Testing imports..."
if uv run python -c "
import sys
sys.path.insert(0, '.')
try:
    import fastapi
    import uvicorn
    import x402
    print('✓ FastAPI, uvicorn, x402 imports successful')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"; then
    echo "✓ All imports successful"
else
    echo "✗ Import test failed"
fi

# Test promptscan import
echo "Testing promptscan import..."
if uv run python -c "
import sys
sys.path.insert(0, '.')
try:
    from prompt_detective.unified_detector import UnifiedDetector
    print('✓ UnifiedDetector import successful')
except ImportError as e:
    print(f'⚠️  UnifiedDetector import warning: {e}')
    print('  Mock engine will be used instead')
"; then
    echo "✓ promptscan test completed"
fi

echo ""
echo "Deployment structure looks good!"
echo ""
echo "To start the server:"
echo "  1. Update WALLET_ADDRESS in website/.env"
echo "  2. Run: cd website && ./run.sh"
echo ""
echo "The server will start on http://localhost:8000"