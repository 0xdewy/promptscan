"""
Basic tests for Prompt Detective API
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_imports():
    """Test that basic imports work."""
    try:
        from api.models.inference_engine import InferenceEngine

        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_inference_engine_creation():
    """Test that inference engine can be created."""
    from api.models.inference_engine import InferenceEngine

    engine = InferenceEngine()
    assert engine is not None
    assert hasattr(engine, "predict")
    assert hasattr(engine, "models_loaded")

    # Check that models_loaded is a dictionary
    assert isinstance(engine.models_loaded, dict)


def test_mock_prediction():
    """Test mock prediction functionality."""
    from api.models.inference_engine import InferenceEngine

    engine = InferenceEngine()

    # Test with a simple prompt
    result = engine.predict("Test prompt")

    # Check result structure
    assert isinstance(result, dict)
    assert "individual" in result
    assert "ensemble" in result
    assert "inference_time" in result

    # Check individual predictions
    individual = result["individual"]
    assert "cnn" in individual
    assert "lstm" in individual
    assert "transformer" in individual

    # Check each model has prediction and confidence
    for model in ["cnn", "lstm", "transformer"]:
        assert "prediction" in individual[model]
        assert "confidence" in individual[model]
        assert isinstance(individual[model]["prediction"], str)
        assert isinstance(individual[model]["confidence"], (int, float))

    # Check ensemble result
    ensemble = result["ensemble"]
    assert "prediction" in ensemble
    assert "confidence" in ensemble

    # Check inference time
    assert isinstance(result["inference_time"], (int, float))
    assert result["inference_time"] > 0


def test_memory_info():
    """Test memory information retrieval."""
    from api.models.inference_engine import InferenceEngine

    engine = InferenceEngine()
    memory_info = engine.get_memory_info()

    assert isinstance(memory_info, dict)
    assert "rss_mb" in memory_info
    assert "vms_mb" in memory_info
    assert "percent" in memory_info
    assert "available_mb" in memory_info
    assert "total_mb" in memory_info

    # Check values are positive numbers
    for key in ["rss_mb", "vms_mb", "available_mb", "total_mb"]:
        assert isinstance(memory_info[key], (int, float))
        assert memory_info[key] > 0

    # Percent should be between 0 and 100
    assert 0 <= memory_info["percent"] <= 100


def test_error_handling():
    """Test error handling in predictions."""
    from api.models.inference_engine import InferenceEngine

    engine = InferenceEngine()

    # Test with empty prompt
    result = engine.predict("")
    assert result is not None

    # Test with very long prompt (should still work in mock mode)
    long_prompt = "x" * 1000
    result = engine.predict(long_prompt)
    assert result is not None


if __name__ == "__main__":
    # Run tests directly
    test_imports()
    print("✅ Imports test passed")

    test_inference_engine_creation()
    print("✅ Inference engine creation test passed")

    test_mock_prediction()
    print("✅ Mock prediction test passed")

    test_memory_info()
    print("✅ Memory info test passed")

    test_error_handling()
    print("✅ Error handling test passed")

    print("\n✅ All basic tests passed!")
