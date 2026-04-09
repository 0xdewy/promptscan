#!/usr/bin/env python3
"""Simple FastAPI test to check if it works."""

from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print("Starting simple FastAPI test server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
