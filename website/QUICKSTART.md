# Prompt Detective Deployment - Quick Start

## Overview
FastAPI server with x402 payment gating for prompt injection detection. The API is x402-gated ($0.01 per prediction) while the frontend is free to use.

## Quick Start

1. **Update wallet address:**
   ```bash
   cd website
   # Edit .env file and update WALLET_ADDRESS with your Base mainnet wallet
   nano .env  # or use your favorite editor
   ```

2. **Start the server:**
   ```bash
   ./run.sh
   ```

3. **Access the application:**
   - Frontend: http://localhost:8000/
   - API Docs: http://localhost:8000/api/docs
   - Health check: http://localhost:8000/api/v1/health

## Features

- **x402 Payment Integration**: API calls require $0.01 payment via x402 protocol
- **Ensemble Model**: Uses CNN, LSTM, and Transformer models for detection
- **Free Frontend**: Web interface is free to use
- **Base Mainnet**: Uses Base mainnet for payments (network: `eip155:8453`)
- **CDP Facilitator**: Uses Coinbase Developer Platform facilitator

## Testing

Run the quick test to verify setup:
```bash
./quick_test.sh
```

## Production Deployment

For production deployment to Hetzner CPX31, see:
- `scripts/deploy.sh` - Deployment script
- `config/gunicorn.conf.py` - Production server config
- `docker/` - Docker configuration

## API Endpoints

- `POST /api/v1/predict` - x402-protected prediction endpoint ($0.01)
- `GET /api/v1/health` - Free health check
- `GET /api/v1/info` - Free API information
- `GET /api/v1/stats` - Free model statistics
- `GET /` - Free web interface

## Environment Variables

Create `.env` file from `.env.example`:
```bash
WALLET_ADDRESS=0xYourWalletAddressHere  # REQUIRED: Your Base mainnet wallet
NETWORK=eip155:8453                      # Base mainnet
PRICE=$0.01                              # Price per prediction
FACILITATOR_URL=https://api.cdp.coinbase.com/platform/v2/x402
```

## Troubleshooting

1. **Import errors**: Run `uv sync --extra deployment` from project root
2. **PyTorch issues**: The system will fall back to mock engine if PyTorch fails
3. **Port conflicts**: Change port in `run.sh` (line 57)
4. **Wallet address**: Must be a valid Base mainnet address

## Next Steps

1. Update `.env` with your wallet address
2. Test with `./run.sh`
3. Deploy to production using `scripts/deploy.sh`