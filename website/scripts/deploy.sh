#!/bin/bash

# Prompt Detective Deployment Script
# For Hetzner CPX31 server deployment

set -e  # Exit on error

echo "🚀 Prompt Detective Deployment Script"
echo "====================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Please run as root or with sudo"
    exit 1
fi

# Configuration
APP_USER="promptdetective"
APP_DIR="/opt/promptscan"
VENV_DIR="$APP_DIR/venv"
LOG_DIR="/var/log/promptscan"
CONFIG_DIR="/etc/promptscan"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Update system
print_status "Updating system packages..."
apt-get update
apt-get upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    gcc \
    g++ \
    curl \
    git \
    nginx \
    certbot \
    python3-certbot-nginx \
    supervisor \
    ufw

# Create application user
if ! id "$APP_USER" &>/dev/null; then
    print_status "Creating application user: $APP_USER..."
    useradd -m -s /bin/bash "$APP_USER"
    usermod -aG sudo "$APP_USER"
else
    print_warning "User $APP_USER already exists"
fi

# Create directories
print_status "Creating directories..."
mkdir -p "$APP_DIR" "$LOG_DIR" "$CONFIG_DIR"
chown -R "$APP_USER:$APP_USER" "$APP_DIR" "$LOG_DIR"
chmod 755 "$APP_DIR" "$LOG_DIR"

# Set up Python virtual environment
print_status "Setting up Python virtual environment..."
sudo -u "$APP_USER" python3.11 -m venv "$VENV_DIR"
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install --upgrade pip

# Copy application files (assuming they're in current directory)
print_status "Copying application files..."
if [ -d "./safe_prompts_deployment" ]; then
    cp -r ./safe_prompts_deployment/* "$APP_DIR/"
else
    print_error "Application files not found in ./safe_prompts_deployment"
    exit 1
fi

chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# Install Python dependencies
print_status "Installing Python dependencies..."
cd "$APP_DIR"
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install -r requirements.txt

# Create environment file
print_status "Creating environment configuration..."
cat > "$CONFIG_DIR/.env" << EOF
# Prompt Detective API Configuration
WALLET_ADDRESS=REPLACE_WITH_YOUR_WALLET_ADDRESS
FACILITATOR_URL=https://api.cdp.coinbase.com/platform/v2/x402
NETWORK=eip155:8453
PRICE=\$0.01
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
LOG_FORMAT=json
CORS_ORIGINS=*
ENVIRONMENT=production
EOF

chown "$APP_USER:$APP_USER" "$CONFIG_DIR/.env"
chmod 600 "$CONFIG_DIR/.env"

# Create symlink to environment file
ln -sf "$CONFIG_DIR/.env" "$APP_DIR/.env"

# Set up Supervisor
print_status "Configuring Supervisor..."
cat > /etc/supervisor/conf.d/promptscan.conf << EOF
[program:promptscan]
command=$VENV_DIR/bin/gunicorn -c $APP_DIR/config/gunicorn.conf.py api.main:app
directory=$APP_DIR
user=$APP_USER
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=$LOG_DIR/error.log
stdout_logfile=$LOG_DIR/access.log
environment=PYTHONPATH="$APP_DIR",PATH="$VENV_DIR/bin:%(ENV_PATH)s"
EOF

# Set up Nginx
print_status "Configuring Nginx..."
cat > /etc/nginx/sites-available/promptscan << EOF
server {
    listen 80;
    server_name _;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss application/atom+xml image/svg+xml;
    
    # Client settings
    client_max_body_size 10M;
    
    # API endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support if needed
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Static files
    location /static/ {
        alias $APP_DIR/api/frontend/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Frontend
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

ln -sf /etc/nginx/sites-available/promptscan /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Configure firewall
print_status "Configuring firewall..."
ufw allow 22/tcp  # SSH
ufw allow 80/tcp  # HTTP
ufw allow 443/tcp # HTTPS
ufw --force enable

# Start services
print_status "Starting services..."
systemctl restart supervisor
systemctl restart nginx
systemctl enable supervisor
systemctl enable nginx

# Verify services are running
print_status "Verifying services..."
sleep 3

if systemctl is-active --quiet supervisor; then
    print_status "Supervisor is running"
else
    print_error "Supervisor failed to start"
    exit 1
fi

if systemctl is-active --quiet nginx; then
    print_status "Nginx is running"
else
    print_error "Nginx failed to start"
    exit 1
fi

# Check application health
print_status "Checking application health..."
sleep 5
if curl -f http://localhost/api/v1/health > /dev/null 2>&1; then
    print_status "Application is healthy"
else
    print_warning "Application health check failed (might need more time to start)"
fi

# Print deployment summary
echo ""
echo "====================================="
echo "✅ Deployment Complete!"
echo "====================================="
echo ""
echo "Application Information:"
echo "  • URL: http://$(curl -s ifconfig.me)"
echo "  • API: http://$(curl -s ifconfig.me)/api/v1/predict"
echo "  • Docs: http://$(curl -s ifconfig.me)/api/docs"
echo "  • Health: http://$(curl -s ifconfig.me)/api/v1/health"
echo ""
echo "Next Steps:"
echo "  1. Edit wallet address in: $CONFIG_DIR/.env"
echo "  2. Set up SSL certificate:"
echo "     certbot --nginx -d your-domain.com"
echo "  3. Monitor logs:"
echo "     tail -f $LOG_DIR/access.log"
echo "     tail -f $LOG_DIR/error.log"
echo "  4. Check supervisor status:"
echo "     supervisorctl status"
echo ""
echo "Troubleshooting:"
echo "  • Check logs: journalctl -u supervisor"
echo "  • Restart app: supervisorctl restart promptscan"
echo "  • Reload nginx: nginx -s reload"
echo ""
echo "Remember to:"
echo "  • Replace WALLET_ADDRESS in $CONFIG_DIR/.env"
echo "  • Set up DNS for your domain"
echo "  • Configure SSL/TLS certificates"
echo "  • Set up monitoring and alerts"
echo ""

print_status "Deployment script completed successfully!"