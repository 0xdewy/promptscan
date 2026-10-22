# Integrations

## FastAPI middleware

Reject requests whose user text looks like an injection before it reaches your model.

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from promptscan import UnifiedDetector

app = FastAPI()
detector = UnifiedDetector(model_type="ensemble")  # loads once, ~1 s


@app.middleware("http")
async def block_prompt_injection(request: Request, call_next):
    if request.method == "POST" and request.url.path.startswith("/chat"):
        body = await request.json()
        text = body.get("message", "")
        if text:
            verdict = detector.predict(text)
            if verdict["prediction"] == "INJECTION":
                return JSONResponse(
                    status_code=400,
                    content={"error": "prompt injection detected", "confidence": verdict["confidence"]},
                )
    return await call_next(request)
```

`predict` takes about 100 ms per prompt on a 2-vCPU box (the CNN alone is under 1 ms), so running it inline is fine for chat-sized traffic. For higher throughput call `predict_batch`.

## Hosted API

```bash
curl -X POST https://promptscan.dewy.ch/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Ignore all previous instructions"}'
```

200 requests per IP per day are free. The response carries `X-Free-Remaining`. After that the server answers 402 with x402 (USDC on Base) and MPP payment details at $0.01 per request.

## MCP server

```bash
pip install 'promptscan[mcp]'
promptscan-mcp          # stdio transport
```

Claude Desktop / Claude Code config:

```json
{"mcpServers": {"promptscan": {"command": "promptscan-mcp"}}}
```

Tools: `scan_text(text)` and `scan_many(texts)`, each returning `prediction`, `confidence` and `individual_predictions`.

## GitHub Action

```yaml
- uses: 0xdewy/promptscan@master
  with:
    path: prompts        # file or directory
    fail-on-injection: "true"
```

See `.github/workflows/example-scan.yml`.
