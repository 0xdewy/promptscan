"""
Production FastAPI server for promptscan
Uses real ensemble models (CNN, LSTM, Transformer)
Optimized for CPU deployment
"""

import os
import time
import traceback
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="promptscan API",
    version="1.0.0",
    description="Prompt injection detection",
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


# Pydantic model for feedback request
class FeedbackRequest(BaseModel):
    prompt: str
    predicted_label: str
    user_label: str
    ensemble_confidence: float
    individual_predictions: List[Dict[str, Any]]
    model_type: str = "ensemble"
    voting_strategy: str = "majority"
    source: str = "web_interface"


# Pydantic models for file upload
class FilePredictionResult(BaseModel):
    """Result for a single file prediction."""

    filename: str
    content_preview: str
    prediction: Dict[str, Any]
    error: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    """Request for batch prediction."""

    prompts: List[str]
    sources: Optional[List[str]] = None


class BatchPredictionResult(BaseModel):
    """Result for batch prediction."""

    results: List[FilePredictionResult]
    summary: Dict[str, Any]
    total_files: int
    processed_files: int
    failed_files: int


# File processing utilities
class FileProcessor:
    """Utilities for processing uploaded files."""

    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md",
        ".json",
        ".csv",
        ".yaml",
        ".yml",
        ".py",
        ".js",
        ".html",
    }
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_TOTAL_SIZE = 100 * 1024 * 1024  # 100MB

    @staticmethod
    def is_supported_file(filename: str) -> bool:
        """Check if file extension is supported."""
        ext = Path(filename).suffix.lower()
        return ext in FileProcessor.SUPPORTED_EXTENSIONS

    @staticmethod
    def read_file_content(file: UploadFile) -> Tuple[str, Optional[str]]:
        """
        Read content from uploaded file based on file type.

        Returns:
            Tuple of (content, error_message)
        """
        try:
            # Check file size
            file.file.seek(0, 2)  # Seek to end
            file_size = file.file.tell()
            file.file.seek(0)  # Reset to beginning

            if file_size > FileProcessor.MAX_FILE_SIZE:
                return (
                    "",
                    f"File too large: {file_size} bytes (max {FileProcessor.MAX_FILE_SIZE})",
                )

            # Read based on file type
            ext = Path(file.filename).suffix.lower()

            if ext in {".txt", ".md", ".py", ".js", ".html"}:
                # Text files - read as UTF-8
                content_bytes = file.file.read()
                try:
                    content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    # Try other encodings
                    try:
                        content = content_bytes.decode("latin-1")
                    except:
                        return "", "Could not decode file as text"

            elif ext == ".json":
                # JSON files - parse and extract text
                content_bytes = file.file.read()
                try:
                    json_content = json.loads(content_bytes.decode("utf-8"))
                    # Try to extract text from common JSON structures
                    content = FileProcessor._extract_text_from_json(json_content)
                except json.JSONDecodeError:
                    return "", "Invalid JSON format"
                except UnicodeDecodeError:
                    return "", "Could not decode JSON file"

            elif ext in {".csv", ".yaml", ".yml"}:
                # For now, read as text - we'll add proper parsing later
                content_bytes = file.file.read()
                try:
                    content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    return "", "Could not decode file as text"

            else:
                return "", f"Unsupported file type: {ext}"

            # Limit content length for processing
            if len(content) > 100000:  # 100k characters
                content = content[:100000] + "\n...[truncated]"

            return content, None

        except Exception as e:
            return "", f"Error reading file: {str(e)}"

    @staticmethod
    def _extract_text_from_json(data: Any) -> str:
        """Extract text content from JSON structure."""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            # Try common keys that might contain text
            text_keys = ["text", "content", "prompt", "message", "description"]
            for key in text_keys:
                if key in data and isinstance(data[key], str):
                    return data[key]

            # Recursively extract from values
            texts = []
            for value in data.values():
                if isinstance(value, str):
                    texts.append(value)
                elif isinstance(value, (dict, list)):
                    extracted = FileProcessor._extract_text_from_json(value)
                    if extracted:
                        texts.append(extracted)
            return "\n".join(texts)
        elif isinstance(data, list):
            texts = []
            for item in data:
                extracted = FileProcessor._extract_text_from_json(item)
                if extracted:
                    texts.append(extracted)
            return "\n".join(texts)
        else:
            return str(data)

    @staticmethod
    def process_files(files: List[UploadFile]) -> List[Dict[str, Any]]:
        """Process multiple uploaded files."""
        results = []
        total_size = 0

        for file in files:
            # Check file type
            if not FileProcessor.is_supported_file(file.filename):
                results.append(
                    {
                        "filename": file.filename,
                        "content": "",
                        "error": f"Unsupported file type: {Path(file.filename).suffix}",
                    }
                )
                continue

            # Read file content
            content, error = FileProcessor.read_file_content(file)

            if error:
                results.append(
                    {"filename": file.filename, "content": "", "error": error}
                )
            else:
                results.append(
                    {"filename": file.filename, "content": content, "error": None}
                )

        return results


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
            from promptscan.unified_detector import UnifiedDetector

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

    def predict_batch(
        self, prompts: List[str], sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Predict for multiple prompts in batch."""
        start_time = time.time()

        if not self.initialized or self.detector is None:
            raise RuntimeError(f"Inference engine not initialized: {self.init_error}")

        try:
            # Use the real UnifiedDetector's batch prediction if available
            if hasattr(self.detector, "predict_batch"):
                batch_results = self.detector.predict_batch(prompts)
            else:
                # Fallback to sequential prediction
                batch_results = []
                for prompt in prompts:
                    result = self.detector.predict(prompt)
                    batch_results.append(result)

            # Format results
            inference_time = round((time.time() - start_time) * 1000, 2)  # ms

            # Process individual results
            individual_results = []
            injection_count = 0
            safe_count = 0

            for i, result in enumerate(batch_results):
                source = sources[i] if sources and i < len(sources) else f"prompt_{i}"

                # Extract individual predictions if available
                individual_predictions = []
                if "individual_predictions" in result:
                    for j, pred in enumerate(result["individual_predictions"]):
                        if j == 0:
                            model_name = "CNN"
                        elif j == 1:
                            model_name = "LSTM"
                        elif j == 2:
                            model_name = "Transformer"
                        else:
                            model_name = f"Model {j}"

                        individual_predictions.append(
                            {
                                "model": model_name,
                                "prediction": pred.get("prediction", "UNKNOWN"),
                                "confidence": round(pred.get("confidence", 0.0), 3),
                                "model_idx": j,
                            }
                        )

                # Count votes
                prediction = result.get("prediction", "UNKNOWN")
                if prediction == "INJECTION":
                    injection_count += 1
                elif prediction == "SAFE":
                    safe_count += 1

                individual_results.append(
                    {
                        "source": source,
                        "prompt_preview": result.get("prompt", prompts[i])[:100]
                        + ("..." if len(prompts[i]) > 100 else ""),
                        "prediction": prediction,
                        "confidence": round(result.get("confidence", 0.0), 3),
                        "individual_predictions": individual_predictions,
                        "inference_time": result.get("inference_time_ms", 0),
                    }
                )

            return {
                "results": individual_results,
                "summary": {
                    "total": len(prompts),
                    "injections": injection_count,
                    "safe": safe_count,
                    "injection_percentage": round(
                        injection_count / len(prompts) * 100, 1
                    )
                    if prompts
                    else 0,
                    "safe_percentage": round(safe_count / len(prompts) * 100, 1)
                    if prompts
                    else 0,
                },
                "total_inference_time_ms": inference_time,
                "avg_inference_time_ms": round(inference_time / len(prompts), 2)
                if prompts
                else 0,
            }

        except Exception as e:
            raise RuntimeError(f"Batch prediction failed: {str(e)}")


# Initialize inference engine
print("\n" + "=" * 60)
print("PROMPTSCAN - PRODUCTION SERVER")
print("=" * 60)
inference_engine = ProductionInferenceEngine()

# Initialize feedback store
try:
    from promptscan.feedback_store import ParquetFeedbackStore

    feedback_store = ParquetFeedbackStore()
    print("✓ Feedback store initialized")
except Exception as e:
    print(f"✗ Failed to initialize feedback store: {e}")
    feedback_store = None

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
        return HTMLResponse(content="<h1>promptscan</h1><p>Server is running</p>")


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


@app.post("/api/v1/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload and analyze multiple files for prompt injection detection.

    Supports: .txt, .md, .json, .csv, .yaml, .yml, .py, .js, .html
    Max file size: 10MB per file, 100MB total
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        # Check total number of files
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="Too many files (max 50)")

        # Process files
        processor = FileProcessor()
        file_contents = processor.process_files(files)

        # Extract prompts from file contents
        prompts = []
        sources = []
        errors = []

        for file_data in file_contents:
            if file_data["error"]:
                errors.append(
                    {"filename": file_data["filename"], "error": file_data["error"]}
                )
            elif file_data["content"]:
                prompts.append(file_data["content"])
                sources.append(file_data["filename"])

        if not prompts:
            if errors:
                return {
                    "success": False,
                    "errors": errors,
                    "message": "All files failed to process",
                }
            else:
                raise HTTPException(
                    status_code=400, detail="No valid content found in files"
                )

        # Run batch prediction
        batch_result = inference_engine.predict_batch(prompts, sources)

        # Add file metadata to results
        for i, result in enumerate(batch_result["results"]):
            if i < len(file_contents):
                file_data = file_contents[i]
                result["filename"] = file_data["filename"]
                result["content_preview"] = file_data["content"][:200] + (
                    "..." if len(file_data["content"]) > 200 else ""
                )
                if file_data["error"]:
                    result["error"] = file_data["error"]

        # Add processing summary
        batch_result["file_processing"] = {
            "total_files": len(files),
            "successful_files": len(prompts),
            "failed_files": len(errors),
            "errors": errors,
        }

        return batch_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")


@app.post("/api/v1/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction endpoint for multiple prompts.

    Accepts a list of prompts and optional source identifiers.
    """
    try:
        if not request.prompts:
            raise HTTPException(status_code=400, detail="No prompts provided")

        if len(request.prompts) > 100:
            raise HTTPException(status_code=400, detail="Too many prompts (max 100)")

        # Run batch prediction
        result = inference_engine.predict_batch(request.prompts, request.sources)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/api/v1/health")
async def health():
    """Health check endpoint"""
    engine_info = inference_engine.get_info()

    response = {
        "status": "healthy" if engine_info["initialized"] else "degraded",
        "service": "promptscan API",
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
        "name": "promptscan",
        "version": "1.0.0",
        "description": "Prompt injection detection",
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


@app.post("/api/v1/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Accept user feedback on model predictions.

    Expected JSON:
    {
        "prompt": "text analyzed",
        "predicted_label": "SAFE" or "INJECTION",
        "user_label": "SAFE" or "INJECTION",
        "ensemble_confidence": 0.95,
        "individual_predictions": [...],
        "model_type": "ensemble",
        "voting_strategy": "majority",
        "source": "web_interface"
    }
    """
    try:
        # Validate input
        if not feedback.prompt or not feedback.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        if feedback.predicted_label not in ["SAFE", "INJECTION"]:
            raise HTTPException(
                status_code=400, detail="predicted_label must be 'SAFE' or 'INJECTION'"
            )

        if feedback.user_label not in ["SAFE", "INJECTION"]:
            raise HTTPException(
                status_code=400, detail="user_label must be 'SAFE' or 'INJECTION'"
            )

        if not 0 <= feedback.ensemble_confidence <= 1:
            raise HTTPException(
                status_code=400, detail="ensemble_confidence must be between 0 and 1"
            )

        # Check if feedback store is available
        if feedback_store is None:
            raise HTTPException(
                status_code=503, detail="Feedback system is not available"
            )

        # Add feedback to store
        feedback_id = feedback_store.add_feedback(
            text=feedback.prompt.strip(),
            predicted_label=feedback.predicted_label,
            user_label=feedback.user_label,
            ensemble_confidence=feedback.ensemble_confidence,
            individual_predictions=feedback.individual_predictions,
            model_type=feedback.model_type,
            voting_strategy=feedback.voting_strategy,
            source=feedback.source,
        )

        # Get feedback statistics
        stats = feedback_store.get_statistics()

        return {
            "status": "success",
            "message": "Feedback submitted successfully",
            "feedback_id": feedback_id,
            "statistics": stats,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to submit feedback: {str(e)}"
        )


@app.get("/api/v1/feedback/stats")
async def feedback_stats():
    """Get feedback statistics"""
    if feedback_store is None:
        raise HTTPException(status_code=503, detail="Feedback system is not available")

    stats = feedback_store.get_statistics()
    return {
        "status": "success",
        "statistics": stats,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
