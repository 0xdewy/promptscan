"""
FastAPI application with x402 payment middleware for Prompt Detective API.
"""

import os
import sys
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# x402 imports
from x402.http import FacilitatorConfig, HTTPFacilitatorClient, PaymentOption
from x402.http.middleware.fastapi import PaymentMiddlewareASGI
from x402.http.types import RouteConfig
from x402.mechanisms.evm.exact import ExactEvmServerScheme
from x402.schemas import Network
from x402.server import x402ResourceServer

# Import from installed promptscan package
try:
    from prompt_detective.unified_detector import UnifiedDetector

    print("Using UnifiedDetector from promptscan package")
    inference_engine = UnifiedDetector(model_type="ensemble", device="cpu")
except ImportError as e:
    print(f"Failed to import UnifiedDetector: {e}")
    print("Falling back to simple mock engine")
    # Fallback to simple engine
    from api.models.inference_engine_simple import SimpleInferenceEngine

    inference_engine = SimpleInferenceEngine()

app = FastAPI(
    title="Prompt Detective API",
    version="1.0.0",
    description="Prompt injection detection using ensemble of CNN, LSTM, and Transformer models",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
EVENT_WALLET_ADDRESS = os.getenv(
    "WALLET_ADDRESS", "0xYourWalletAddressHere"
)  # You'll provide this
EVENT_NETWORK: Network = os.getenv("NETWORK", "eip155:8453")  # Base mainnet
FACILITATOR_URL = os.getenv(
    "FACILITATOR_URL", "https://api.cdp.coinbase.com/platform/v2/x402"
)
PRICE = os.getenv("PRICE", "$0.01")

# inference_engine is already initialized in the import section

# Create facilitator client (CDP for mainnet)
facilitator = HTTPFacilitatorClient(FacilitatorConfig(url=FACILITATOR_URL))

# Create resource server and register EVM scheme
server = x402ResourceServer(facilitator)
server.register(EVENT_NETWORK, ExactEvmServerScheme())

# Define x402-protected routes
routes: Dict[str, RouteConfig] = {
    "POST /api/v1/predict": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=EVENT_WALLET_ADDRESS,
                price=PRICE,
                network=EVENT_NETWORK,
            ),
        ],
        mime_type="application/json",
        description="Analyze prompt for injection attacks using CNN+LSTM+Transformer ensemble",
        extensions={
            "bazaar": {
                "discoverable": True,
                "category": "ai-safety",
                "tags": ["prompt-injection", "security", "nlp", "ai-security"],
            },
        },
    ),
}

# Add x402 payment middleware
app.add_middleware(PaymentMiddlewareASGI, routes=routes, server=server)

# Mount frontend (free access)
import os

static_dir = os.path.join(os.path.dirname(__file__), "frontend", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def frontend():
    """Serve the free frontend interface"""
    try:
        index_path = os.path.join(static_dir, "index.html")
        with open(index_path) as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Prompt Detective API</h1><p>Frontend coming soon</p>"
        )


@app.post("/api/v1/predict")
async def predict(prompt: str) -> Dict[str, Any]:
    """
    x402-protected prediction endpoint.
    Requires payment via x402 protocol.

    Returns individual model predictions and ensemble consensus.
    """
    try:
        # Validate input
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        if len(prompt) > 10000:
            raise HTTPException(
                status_code=400, detail="Prompt too long (max 10000 characters)"
            )

        # Run inference
        result = inference_engine.predict(prompt.strip())

        # Format individual predictions
        individual_formatted = []
        if "individual_predictions" in result:
            for i, pred in enumerate(result["individual_predictions"]):
                model_type = pred.get("model_type", f"model_{i}")
                individual_formatted.append(
                    {
                        "model": model_type,
                        "prediction": pred.get("prediction", "UNKNOWN"),
                        "confidence": pred.get("confidence", 0.0),
                        "model_idx": pred.get("model_idx", i),
                    }
                )
        else:
            # Fallback if individual predictions not available
            individual_formatted = [
                {
                    "model": "cnn",
                    "prediction": "UNKNOWN",
                    "confidence": 0.0,
                    "model_idx": 0,
                },
                {
                    "model": "lstm",
                    "prediction": "UNKNOWN",
                    "confidence": 0.0,
                    "model_idx": 1,
                },
                {
                    "model": "transformer",
                    "prediction": "UNKNOWN",
                    "confidence": 0.0,
                    "model_idx": 2,
                },
            ]

        return {
            "prompt": prompt,
            "individual_predictions": individual_formatted,
            "ensemble_prediction": result.get("prediction", "UNKNOWN"),
            "ensemble_confidence": result.get("confidence", 0.0),
            "voting_strategy": result.get("strategy", "unknown"),
            "model_details": {
                "cnn": {
                    "name": "Convolutional Neural Network",
                    "description": "Pattern detection for local injection patterns",
                    "strength": "Fast inference, good for known patterns",
                },
                "lstm": {
                    "name": "Bidirectional LSTM",
                    "description": "Sequential understanding of prompt structure",
                    "strength": "Understands context and sequence",
                },
                "transformer": {
                    "name": "DistilBERT fine-tuned",
                    "description": "Contextual understanding using transformer architecture",
                    "strength": "Highest accuracy, understands nuance",
                },
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/api/v1/health")
async def health():
    """Health check endpoint (free access)"""
    # Get model info from UnifiedDetector
    try:
        model_info = inference_engine.get_info()
        models_loaded = model_info.get("models", ["ensemble"])
    except:
        models_loaded = ["ensemble"]

    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "x402_enabled": True,
        "facilitator": FACILITATOR_URL,
        "network": EVENT_NETWORK,
        "price": PRICE,
        "wallet_address": EVENT_WALLET_ADDRESS[:10] + "..."
        if EVENT_WALLET_ADDRESS
        else "not_set",
    }


@app.get("/api/v1/info")
async def info():
    """API information (free access)"""
    return {
        "name": "Prompt Detective API",
        "version": "1.0.0",
        "description": "Prompt injection detection using ensemble of CNN, LSTM, and Transformer models",
        "pricing": PRICE + " per prediction",
        "supported_networks": [EVENT_NETWORK],
        "payment_method": "x402 protocol (HTTP 402 Payment Required)",
        "endpoints": {
            "POST /api/v1/predict": {
                "description": "x402-protected prediction endpoint",
                "price": PRICE,
                "authentication": "x402 payment signature",
            },
            "GET /api/v1/health": "Free health check",
            "GET /api/v1/info": "Free API information",
            "GET /": "Free web interface",
        },
        "models": {
            "total": 3,
            "types": ["CNN", "LSTM", "Transformer"],
            "ensemble": "Majority voting with confidence weighting",
        },
    }


@app.get("/api/v1/stats")
async def stats():
    """API statistics (free access)"""
    # Get model info from UnifiedDetector
    try:
        model_info = inference_engine.get_info()
        if model_info.get("type") == "ensemble":
            models_loaded = model_info.get("models", ["cnn", "lstm", "transformer"])
            model_types = models_loaded
        else:
            models_loaded = [model_info.get("type", "single")]
            model_types = models_loaded
    except:
        models_loaded = ["ensemble"]
        model_types = ["cnn", "lstm", "transformer"]

    return {
        "models_loaded": models_loaded,
        "model_types": model_types,
        "device": str(inference_engine.device),
        "detector_type": getattr(inference_engine, "model_type", "ensemble"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
