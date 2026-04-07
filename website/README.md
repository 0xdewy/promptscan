# Prompt Detective API with x402 Payments

A production-ready FastAPI server for prompt injection detection with x402 payment protocol integration. Accepts payments via Base network (USDC) at $0.01 per prediction.

## Features

- **x402 Payment Protocol**: Pay-per-use API with HTTP 402 status code
- **Ensemble AI Models**: CNN + LSTM + Transformer consensus voting
- **Simple Frontend**: Free web interface for testing
- **Production Ready**: Docker, Gunicorn, Nginx configuration
- **Hetzner Optimized**: Configured for CPX31 (8 vCPU, 16GB RAM)
- **Mainnet Ready**: Base network with USDC payments

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Free)                       │
│  Simple HTML/JS interface with wallet connection         │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                    FastAPI Server                        │
│  - /api/v1/predict (x402-gated)                         │
│  - /api/v1/health (free)                                │
│  - /api/v1/info (free)                                  │
│  - / (frontend)                                         │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                 PyTorch Inference                        │
│  - CNN model (pattern detection)                        │
│  - LSTM model (sequential understanding)                │
│  - Transformer model (contextual accuracy)              │
│  - Ensemble voting with confidence scores               │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- EVM wallet with USDC on Base network (for receiving payments)
- Coinbase Developer Platform account (for CDP facilitator)

### 2. Installation

```bash
# Clone and setup
cd safe_prompts_deployment

# Install dependencies
pip install -r requirements.txt

# Copy environment file and configure
cp .env.example .env
# Edit .env with your wallet address
```

### 3. Configuration

Edit `.env` file:

```bash
# REQUIRED: Your wallet address for receiving payments
WALLET_ADDRESS=0xYourWalletAddressHere

# x402 Configuration
FACILITATOR_URL=https://api.cdp.coinbase.com/platform/v2/x402
NETWORK=eip155:8453  # Base mainnet
PRICE=$0.01

# Server Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
```

### 4. Run Locally

```bash
# Development mode
python -m api.main

# Production mode with Gunicorn
gunicorn -c config/gunicorn.conf.py api.main:app
```

Visit: http://localhost:8000

## API Usage

### Free Endpoints

```bash
# Health check
curl http://localhost:8000/api/v1/health

# API information
curl http://localhost:8000/api/v1/info

# Frontend
curl http://localhost:8000/
```

### Paid Endpoint (x402)

```bash
# This will return 402 Payment Required
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test prompt"}'

# Response: 402 Payment Required with payment instructions
```

### Using x402 Clients

**Python:**
```python
from x402 import x402ClientSync
from x402.http.clients import x402_requests
from x402.mechanisms.evm import EthAccountSigner
from eth_account import Account

# Setup client with wallet
account = Account.from_key("your_private_key")
signer = EthAccountSigner(account)

client = x402ClientSync()
# Register EVM scheme...

with x402_requests(client) as session:
    response = session.post(
        "http://localhost:8000/api/v1/predict",
        json={"prompt": "Your prompt here"}
    )
    print(response.json())
```

**CLI:**
```bash
# Using x402 CLI (install via npm)
x402 pay http://localhost:8000/api/v1/predict \
  --prompt "Your prompt here" \
  --network base
```

## Deployment

### Hetzner CPX31 Deployment

```bash
# Run deployment script (as root)
chmod +x scripts/deploy.sh
sudo ./scripts/deploy.sh

# Follow post-deployment steps:
# 1. Update wallet address in /etc/prompt-detective/.env
# 2. Set up SSL certificates
# 3. Configure DNS
```

### Docker Deployment

```bash
# Build and run
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose logs -f
```

## Testing

```bash
# Test mock mode (no payment required)
python scripts/test_x402.py --mock

# Test payment flow (requires running server)
python scripts/test_x402.py --payment

# Run all tests
pytest tests/
```

## Pricing & Revenue

- **Price per prediction**: $0.01
- **Payment method**: USDC on Base network via x402
- **Facilitator**: Coinbase Developer Platform (CDP)
- **No platform fees**: x402 protocol has 0% fees

**Revenue Example:**
- 100 predictions/day = $1.00/day
- 3,000 predictions/month = $30.00/month
- Server cost (Hetzner CPX31): €14/month (~$15)
- Break-even: ~1,500 predictions/month

## Model Details

The ensemble combines three models:

| Model | Parameters | Strength | Inference Time |
|-------|------------|----------|----------------|
| **CNN** | 2.7M | Pattern detection | ~10ms |
| **LSTM** | 3.3M | Sequential understanding | ~15ms |
| **LSTM** | 3.3M | Sequential understanding | ~15ms |
| **Transformer** | 67M | Contextual accuracy | ~25ms |
| **Ensemble** | 73M | Combined robustness | ~50ms |

**Voting Strategy:** Majority voting with confidence weighting

## x402 Integration Details

- **Protocol**: x402 v2 with CAIP-2 network identifiers
- **Network**: Base mainnet (`eip155:8453`)
- **Token**: USDC (EIP-3009 for gasless transfers)
- **Facilitator**: CDP (`api.cdp.coinbase.com`)
- **Price Format**: `$0.01` (must include $ prefix)
- **Headers**: `PAYMENT-REQUIRED`, `PAYMENT-SIGNATURE`

## Security

- **CORS**: Configurable origins
- **Input validation**: Max 10,000 characters
- **Rate limiting**: Implemented at Nginx level
- **HTTPS**: Required for production
- **Wallet security**: Private keys never leave client

## Monitoring

```bash
# Application logs
tail -f /var/log/prompt-detective/access.log
tail -f /var/log/prompt-detective/error.log

# Supervisor status
supervisorctl status prompt-detective

# System metrics
htop  # CPU/Memory
df -h # Disk usage
```

## Troubleshooting

### Common Issues

1. **402 Payment Required but no payment header**
   - Check facilitator URL in `.env`
   - Verify network identifier format

2. **Models not loading**
   - Check `prompt_detective` package is accessible
   - Verify model checkpoint files exist

3. **Wallet connection issues**
   - Ensure wallet has USDC on Base network
   - Check x402 client configuration

4. **High memory usage**
   - Reduce Gunicorn workers in `config/gunicorn.conf.py`
   - Consider ONNX optimization for Phase 2

### Logs Location

- Application: `/var/log/prompt-detective/`
- Supervisor: `journalctl -u supervisor`
- Nginx: `/var/log/nginx/`

## Development

### Project Structure

```
safe_prompts_deployment/
├── api/
│   ├── main.py              # FastAPI application
│   ├── models/              # Inference engine
│   ├── frontend/static/     # Web interface
│   └── utils/              # Utilities
├── config/                  # Configuration files
├── docker/                  # Docker files
├── scripts/                 # Deployment scripts
├── tests/                   # Test files
├── requirements.txt         # Dependencies
└── README.md               # This file
```

### Adding New Features

1. **New payment networks**: Add additional `PaymentOption` in `main.py`
2. **Model optimization**: Implement ONNX conversion in Phase 2
3. **Advanced features**: Add rate limiting, analytics, etc.

## License

MIT License

## Support

- **x402 Documentation**: https://docs.x402.org
- **CDP Facilitator**: https://docs.cdp.coinbase.com/x402
- **Base Network**: https://base.org
- **Issue Tracker**: GitHub Issues

## Acknowledgments

- x402 Foundation for the payment protocol
- Coinbase Developer Platform for facilitator service
- Base network for low-cost transactions
- Original prompt_detective authors for the AI models