#!/bin/bash
# Deployment script for Prompt Detective on Hetzner CPU

set -e

echo "=========================================="
echo "Prompt Detective - Hetzner Deployment"
echo "=========================================="

# Check if we're on Hetzner
if [[ "$(hostname)" == *"hetzner"* ]] || [[ "$(hostname -f)" == *"hetzner"* ]]; then
    echo "✓ Detected Hetzner environment"
else
    echo "⚠️  Not on Hetzner - this script is optimized for Hetzner CPU servers"
fi

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
echo "Installing Python and system dependencies..."
sudo apt-get install -y python3 python3-pip python3-venv git curl wget

# Clone or update repository
if [ -d "safe_prompts" ]; then
    echo "Updating existing repository..."
    cd safe_prompts
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/yourusername/safe_prompts.git
    cd safe_prompts
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r website/requirements_simple_production.txt

# Install PyTorch for CPU (optimized for Hetzner)
echo "Installing PyTorch for CPU..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install promptscan package
echo "Installing promptscan package..."
pip install -e .

# Download model checkpoints (if not already present)
echo "Checking for model checkpoints..."
if [ ! -f "prompt_detective/models/checkpoints/cnn_best.pt" ]; then
    echo "Downloading model checkpoints..."
    # Add download commands here or ensure they're in the repository
    echo "⚠️  Model checkpoints not found. Please ensure they are in prompt_detective/models/checkpoints/"
fi

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/promptscan.service > /dev/null << EOF
[Unit]
Description=Prompt Detective API Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/python -m website.api.main
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
echo "Starting Prompt Detective service..."
sudo systemctl daemon-reload
sudo systemctl enable promptscan
sudo systemctl start promptscan

# Wait for service to start
echo "Waiting for service to start..."
sleep 5

# Check service status
echo "Checking service status..."
if sudo systemctl is-active --quiet promptscan; then
    echo "✅ Prompt Detective service is running"
    
    # Test the API
    echo "Testing API endpoints..."
    curl -s http://localhost:8000/api/v1/health | grep -q "healthy" && echo "✅ Health check passed" || echo "⚠️  Health check failed"
    curl -s http://localhost:8000/api/v1/info | grep -q "version" && echo "✅ Info endpoint working" || echo "⚠️  Info endpoint failed"
    
    echo ""
    echo "=========================================="
    echo "DEPLOYMENT COMPLETE!"
    echo "=========================================="
    echo ""
    echo "Prompt Detective is now running at:"
    echo "  • Web UI: http://$(hostname -I | awk '{print $1}'):8000"
    echo "  • API: http://$(hostname -I | awk '{print $1}'):8000/api/v1/predict"
    echo "  • API Docs: http://$(hostname -I | awk '{print $1}'):8000/api/docs"
    echo ""
    echo "Service management commands:"
    echo "  sudo systemctl status promptscan"
    echo "  sudo systemctl restart promptscan"
    echo "  sudo journalctl -u promptscan -f"
    echo ""
    echo "To test the service:"
    echo "  curl -X POST http://localhost:8000/api/v1/predict \\"
    echo "    -H \"Content-Type: application/json\" \\"
    echo "    -d '{\"prompt\": \"Test prompt\"}'"
    echo ""
else
    echo "❌ Service failed to start. Check logs with:"
    echo "   sudo journalctl -u promptscan -f"
    exit 1
fi