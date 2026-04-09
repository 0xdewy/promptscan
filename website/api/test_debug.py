#!/usr/bin/env python3
"""Debug the main.py import."""

import sys
import traceback

# Add the project root to path
sys.path.insert(0, "../../")

print("Attempting to import main.py...")
try:
    import main

    print("✅ Import successful")

    # Check if app exists
    if hasattr(main, "app"):
        print("✅ FastAPI app exists")

        # Try to access the app
        print(f"App title: {main.app.title}")
        print(f"App version: {main.app.version}")

        # Check if inference_engine exists
        if hasattr(main, "inference_engine"):
            print("✅ Inference engine exists")
            if main.inference_engine.initialized:
                print("✅ Inference engine initialized")
            else:
                print("❌ Inference engine not initialized")
                if hasattr(main.inference_engine, "init_error"):
                    print(f"   Error: {main.inference_engine.init_error}")
        else:
            print("❌ Inference engine not found")

        # Check if feedback_store exists
        if hasattr(main, "feedback_store"):
            if main.feedback_store is not None:
                print("✅ Feedback store initialized")
            else:
                print("❌ Feedback store is None")
        else:
            print("❌ Feedback store not found")

    else:
        print("❌ No FastAPI app found")

except Exception as e:
    print(f"❌ Import failed: {e}")
    traceback.print_exc()
