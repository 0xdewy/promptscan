"""
Production FastAPI server for Prompt Detective
Uses real ensemble models (CNN, LSTM, Transformer) from promptscan package
Optimized for CPU deployment on Hetzner
"""

import os
import time
import traceback
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Prompt Detective API",
    version="1.0.0",
    description="Production prompt injection detection using ensemble of CNN, LSTM, and Transformer models",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic model for request body
class PromptRequest(BaseModel):
    prompt: str


class ProductionInferenceEngine:
    """Production inference engine using real trained models."""

    def __init__(self):
        self.models_loaded = {}
        self.device = "cpu"
        self.detector = None
        self.initialized = False
        self.init_error = None

        try:
            print("Initializing ProductionInferenceEngine...")
            print("Loading real ensemble models (CNN, LSTM, Transformer)...")

            # Import and initialize UnifiedDetector
            from prompt_detective.unified_detector import UnifiedDetector

            # Initialize with ensemble mode
            self.detector = UnifiedDetector(model_type="ensemble", device="cpu")

            # Test with a simple prompt to ensure models work
            test_result = self.detector.predict("Test initialization")
            print(f"✓ Models loaded successfully")
            print(f"  Test prediction: {test_result.get('prediction', 'N/A')}")
            print(f"  Test confidence: {test_result.get('confidence', 'N/A')}")

            self.models_loaded = {"cnn": True, "lstm": True, "transformer": True}
            self.initialized = True

        except ImportError as e:
            self.init_error = f"Failed to import models: {e}"
            print(f"✗ {self.init_error}")
            traceback.print_exc()
        except Exception as e:
            self.init_error = f"Failed to initialize models: {e}"
            print(f"✗ {self.init_error}")
            traceback.print_exc()

    def predict(self, prompt: str) -> Dict[str, Any]:
        """Predict if prompt contains injection using real ensemble models."""
        start_time = time.time()

        if not self.initialized or self.detector is None:
            raise RuntimeError(f"Inference engine not initialized: {self.init_error}")

        try:
            # Use the real UnifiedDetector
            result = self.detector.predict(prompt)

            # Format the result for our API
            inference_time = round((time.time() - start_time) * 1000, 2)  # ms

            # Extract individual predictions if available
            individual_predictions = []
            if "individual_predictions" in result:
                for i, pred in enumerate(result["individual_predictions"]):
                    model_type = pred.get("model_type", f"model_{i}")
                    if i == 0:
                        model_name = "CNN"
                    elif i == 1:
                        model_name = "LSTM"
                    elif i == 2:
                        model_name = "Transformer"
                    else:
                        model_name = model_type

                    individual_predictions.append(
                        {
                            "model": model_name,
                            "prediction": pred.get("prediction", "UNKNOWN"),
                            "confidence": round(pred.get("confidence", 0.0), 3),
                            "model_idx": i,
                        }
                    )
            else:
                # Fallback if individual predictions not in result
                individual_predictions = [
                    {
                        "model": "CNN",
                        "prediction": "UNKNOWN",
                        "confidence": 0.0,
                        "model_idx": 0,
                    },
                    {
                        "model": "LSTM",
                        "prediction": "UNKNOWN",
                        "confidence": 0.0,
                        "model_idx": 1,
                    },
                    {
                        "model": "Transformer",
                        "prediction": "UNKNOWN",
                        "confidence": 0.0,
                        "model_idx": 2,
                    },
                ]

            # Count votes
            injection_votes = sum(
                1 for p in individual_predictions if p["prediction"] == "INJECTION"
            )
            safe_votes = len(individual_predictions) - injection_votes

            return {
                "individual_predictions": individual_predictions,
                "ensemble_prediction": result.get("prediction", "UNKNOWN"),
                "ensemble_confidence": round(result.get("confidence", 0.0), 3),
                "inference_time_ms": inference_time,
                "votes": {"injection": injection_votes, "safe": safe_votes},
                "model_source": "real_ensemble",
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Prediction failed: {e}")

    def get_info(self) -> Dict[str, Any]:
        """Get engine information."""
        if self.initialized:
            return {
                "type": "production_ensemble",
                "models_loaded": list(self.models_loaded.keys()),
                "device": self.device,
                "initialized": self.initialized,
                "description": "Real CNN, LSTM, and Transformer ensemble models",
            }
        else:
            return {
                "type": "production_ensemble",
                "models_loaded": [],
                "device": self.device,
                "initialized": self.initialized,
                "error": self.init_error,
                "description": "Failed to initialize real models",
            }


# Initialize inference engine
print("\n" + "=" * 60)
print("PROMPT DETECTIVE - PRODUCTION SERVER")
print("=" * 60)
inference_engine = ProductionInferenceEngine()
print("=" * 60 + "\n")

# Mount frontend
import os

static_dir = os.path.join(os.path.dirname(__file__), "frontend", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def frontend():
    """Serve the frontend interface"""
    try:
        index_path = os.path.join(static_dir, "index.html")
        with open(index_path) as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Prompt Detective</h1><p>Server is running</p>")


@app.post("/api/v1/predict")
async def predict(request: PromptRequest) -> Dict[str, Any]:
    """
    Production prediction endpoint using real ensemble models.

    Returns individual model predictions and ensemble consensus.
    """
    try:
        prompt_text = request.prompt

        # Validate input
        if not prompt_text or not prompt_text.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        if len(prompt_text) > 10000:
            raise HTTPException(
                status_code=400, detail="Prompt too long (max 10000 characters)"
            )

        # Run inference with real models
        result = inference_engine.predict(prompt_text.strip())

        return {
            "prompt": prompt_text,
            "individual_predictions": result["individual_predictions"],
            "ensemble_prediction": result["ensemble_prediction"],
            "ensemble_confidence": result["ensemble_confidence"],
            "inference_time_ms": result["inference_time_ms"],
            "votes": result["votes"],
            "model_source": result.get("model_source", "unknown"),
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
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Model inference error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/api/v1/health")
async def health():
    """Health check endpoint"""
    engine_info = inference_engine.get_info()

    response = {
        "status": "healthy" if engine_info["initialized"] else "degraded",
        "service": "Prompt Detective API",
        "version": "1.0.0",
        "models_loaded": engine_info["models_loaded"],
        "model_type": engine_info["type"],
        "device": engine_info["device"],
        "initialized": engine_info["initialized"],
    }

    if not engine_info["initialized"] and "error" in engine_info:
        response["error"] = engine_info["error"]
        response["status"] = "unhealthy"

    return response


@app.get("/api/v1/info")
async def info():
    """API information"""
    engine_info = inference_engine.get_info()

    return {
        "name": "Prompt Detective",
        "version": "1.0.0",
        "description": "Production prompt injection detection using ensemble of CNN, LSTM, and Transformer models",
        "status": "production",
        "deployment": "Hetzner CPU",
        "model_status": "loaded" if engine_info["initialized"] else "failed",
        "models": {
            "total": 3,
            "types": ["CNN", "LSTM", "Transformer"],
            "ensemble": "Majority voting with confidence",
        },
        "endpoints": {
            "POST /api/v1/predict": "Analyze prompt for injection",
            "GET /api/v1/health": "System health check",
            "GET /api/v1/info": "API information",
            "GET /": "Web interface",
        },
    }


@app.get("/api/v1/stats")
async def stats():
    """API statistics"""
    engine_info = inference_engine.get_info()
    return {
        "models_loaded": engine_info["models_loaded"],
        "model_types": engine_info["models_loaded"],
        "device": engine_info["device"],
        "model_type": engine_info["type"],
        "initialized": engine_info["initialized"],
        "description": engine_info["description"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
