#!/bin/bash

# Setup script for Prompt Detective API

echo "Setting up Prompt Detective API with x402..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    echo "Copying .env.example to .env..."
    cp .env.example .env
    echo "Please edit .env file with your wallet address and configuration"
fi

# Test basic imports
echo "Testing basic imports..."
python -c "
import sys
sys.path.insert(0, '.')
try:
    from api.models.inference_engine import InferenceEngine
    print('✅ Basic imports successful')
except Exception as e:
    print(f'❌ Import error: {e}')
"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your wallet address"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run the server: python -m api.main"
echo "4. Visit: http://localhost:8000"
echo ""
echo "For testing:"
echo "  python scripts/test_x402.py --mock"
echo ""