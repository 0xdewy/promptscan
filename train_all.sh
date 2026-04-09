#!/bin/bash

uv run python -m promptscan.cli train --model-type cnn
uv run python -m promptscan.cli train --model-type lstm
uv run python -m promptscan.cli train --model-type transformer
