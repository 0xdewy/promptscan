# PromptScan

Neural network consensus prompt injection detection.

## Install

    pip install promptscan
    uv install promptscan
    pipx install promptscan

## Use

    promptscan "Ignore all previous instructions"
    promptscan input.txt
    promptscan ./prompts/
    promptscan https://example.com/user-input

Each model votes independently. You see exactly how it decides.

## Python API

    from promptscan import UnifiedDetector

    result = UnifiedDetector().predict(text)
    # → {prediction, confidence, individual_predictions}

## Why Consensus?
Each neural net has different strengths and weaknesses:

CNN (fast, local patterns)
- Phrase-level triggers: "ignore all previous", "disregard instructions", "you are now a"
- N-gram style exploits that don't depend on word order
- Misses: subtle contextual shifts, reordered injection syntax
LSTM (sequential, ordered)
- Understands word order matters: "previous instructions" + "ignore" together
- Catches injection templates where sequence carries meaning
- Misses: long-range dependencies (injection at start affecting meaning at end)
Transformer (contextual, global)
- Sophisticated attacks: encoded instructions, role-played injection, subtle manipulation
- Long-range context — injection in first sentence pivots interpretation at the end
- Novel/unseen patterns it can generalize to
- Misses: extremely short, obvious patterns (computational overhead isn't worth it for "bomb")

## Features
- Ensemble of CNN (2.7M), LSTM (3.3M), Transformer (67M)
- Transparent voting — each model's prediction and confidence exposed
- Auto-download from Hugging Face Hub (0xdewy/promptscan)
- Analyze files and directories via CLI
