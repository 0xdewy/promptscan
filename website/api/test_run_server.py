#!/usr/bin/env python3
"""Run the server and capture output."""

import subprocess
import time
import sys
import os


def run_server():
    """Run the server and print its output."""
    print("Starting server...")

    # Change to the API directory
    os.chdir("/home/user/code/prompt-scan/website/api")

    # Run the server
    proc = subprocess.Popen(
        [
            "../../.venv/bin/python",
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    print(f"Server PID: {proc.pid}")

    # Read output line by line
    try:
        for line in iter(proc.stdout.readline, ""):
            print(f"SERVER: {line}", end="")
            sys.stdout.flush()

            # Check for specific messages
            if "Uvicorn running" in line:
                print("\n✅ Server is running!")
                # Keep server running for a bit
                time.sleep(10)
                break
            elif "error" in line.lower() or "exception" in line.lower():
                print(f"\n❌ Error detected: {line}")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("\nStopping server...")
        proc.terminate()
        proc.wait()
        print("Server stopped")


if __name__ == "__main__":
    run_server()
