# Prompt Detective - Production Deployment

Production-ready prompt injection detection service using real ensemble models (CNN, LSTM, Transformer). Optimized for CPU deployment on Hetzner servers.

## Features

- **Real AI Models**: Uses actual trained CNN, LSTM, and Transformer models from the `prompt-detective` package
- **Ensemble Detection**: Combines predictions from 3 different model architectures
- **Minimal UI**: Clean, professional interface focused on functionality
- **Production Ready**: Error handling, health checks, and monitoring
- **CPU Optimized**: Configured for efficient CPU inference on Hetzner servers
- **Free Service**: No payment or authentication required

## Quick Start

### 1. Install Dependencies

```bash
cd /home/user/code/safe_prompts
./website/run_production.sh
```

The script will:
- Check Python version (3.8+ required)
- Install all dependencies (PyTorch CPU, Transformers, FastAPI, etc.)
- Verify model files exist
- Configure environment for optimal CPU performance
- Start the production server

### 2. Access the Service

- **Web Interface**: http://localhost:8000/
- **API Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/api/v1/health
- **API Info**: http://localhost:8000/api/v1/info

### 3. Test the System

```bash
cd website
python test_production.py
```

## API Endpoints

### `POST /api/v1/predict`
Analyze a prompt for injection attempts.

**Request:**
```json
{
  "prompt": "Your prompt text here"
}
```

**Response:**
```json
{
  "prompt": "Original prompt",
  "ensemble_prediction": "SAFE|INJECTION",
  "ensemble_confidence": 0.95,
  "individual_predictions": [
    {
      "model": "CNN",
      "prediction": "SAFE",
      "confidence": 0.92,
      "model_idx": 0
    },
    {
      "model": "LSTM",
      "prediction": "INJECTION",
      "confidence": 0.87,
      "model_idx": 1
    },
    {
      "model": "Transformer",
      "prediction": "SAFE",
      "confidence": 0.96,
      "model_idx": 2
    }
  ],
  "inference_time_ms": 125.5,
  "votes": {
    "injection": 1,
    "safe": 2
  },
  "model_source": "real_ensemble"
}
```

### `GET /api/v1/health`
Health check endpoint.

### `GET /api/v1/info`
API information and status.

### `GET /api/v1/stats`
Model statistics and system info.

## Model Architecture

### CNN (Convolutional Neural Network)
- **Purpose**: Pattern detection for local injection signatures
- **Strength**: Fast inference, good for known patterns
- **Checkpoint**: `cnn_best.pt`

### LSTM (Long Short-Term Memory)
- **Purpose**: Sequential understanding of prompt structure
- **Strength**: Understands context and sequence
- **Checkpoint**: `lstm_best.pt`

### Transformer (DistilBERT fine-tuned)
- **Purpose**: Contextual analysis using transformer architecture
- **Strength**: Deep semantic understanding
- **Checkpoint**: `transformer_best.pt`

### Ensemble Strategy
- **Voting**: Majority vote from 3 models
- **Confidence**: Weighted by individual model confidence
- **Fallback**: Graceful degradation if models fail

## Deployment on Hetzner

### Server Requirements
- **CPU**: 2+ cores recommended (optimized for CPU inference)
- **RAM**: 4GB+ (models load into memory)
- **Storage**: 2GB+ for models and dependencies
- **Python**: 3.8 or higher

### Performance Optimization
- **Thread Configuration**: Uses all available CPU cores
- **Memory Management**: PyTorch memory optimization settings
- **Model Caching**: Models loaded once at startup
- **Batch Processing**: Single prompt optimization

### Monitoring
- Health endpoint for status checks
- Inference time tracking
- Model loading status
- Error rate monitoring

## Project Structure

```
website/
├── api/
│   ├── main.py                    # Production FastAPI server
│   └── frontend/static/
│       ├── index.html             # Minimal production UI
│       ├── style.css              # Professional CSS
│       └── script.js              # Production JavaScript
├── requirements_production.txt    # CPU-optimized dependencies
├── run_production.sh             # Production startup script
├── test_production.py            # Production test suite
└── README_PRODUCTION.md          # This file
```

## Model Files

Models are loaded from:
```
prompt_detective/models/checkpoints/
├── cnn_best.pt
├── lstm_best.pt
└── transformer_best.pt
```

## Troubleshooting

### Common Issues

1. **Models fail to load**
   ```
   Error: Model files not found
   ```
   **Solution**: Ensure model files exist in `prompt_detective/models/checkpoints/`

2. **Slow inference**
   ```
   Inference time > 5 seconds
   ```
   **Solution**: Check CPU usage, ensure sufficient resources

3. **Memory errors**
   ```
   CUDA out of memory / CPU memory error
   ```
   **Solution**: The system is CPU-only, check available RAM

4. **Import errors**
   ```
   ModuleNotFoundError: No module named 'torch'
   ```
   **Solution**: Run `./run_production.sh` to install dependencies

### Logs and Monitoring

- Check server logs for initialization messages
- Monitor `/api/v1/health` endpoint
- Check system resource usage (CPU, RAM)
- Review error responses from API

## Development

### Updating Models
1. Place new model checkpoints in `prompt_detective/models/checkpoints/`
2. Restart the server
3. Models are automatically loaded on startup

### Customizing UI
- Edit `website/api/frontend/static/index.html`
- Modify `website/api/frontend/static/style.css`
- Update `website/api/frontend/static/script.js`

### Adding Features
- Extend `website/api/main.py` with new endpoints
- Update frontend JavaScript to call new endpoints
- Test thoroughly with `test_production.py`

## License

Production deployment for prompt injection detection. Models trained on proprietary datasets.