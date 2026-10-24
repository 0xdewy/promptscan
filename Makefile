# Makefile for PromptScan
# Uses uv for Python package management

.PHONY: help install dev test lint format typecheck build clean train predict run

# Default target
help:
	@echo "PromptScan Development Commands"
	@echo ""
	@echo "Usage:"
	@echo "  make install    Install the package in development mode"
	@echo "  make dev        Install with development dependencies"
	@echo "  make test       Run tests"
	@echo "  make lint       Run linting checks"
	@echo "  make format     Format code with ruff"
	@echo "  make typecheck  Run type checking with mypy"
	@echo "  make build      Build package distribution"
	@echo "  make clean      Clean build artifacts"
	@echo "  make setup      Full development setup"
	@echo "  make pre-commit Run all checks before commit"
	@echo "  make train      Train the model"
	@echo "  make predict    Run prediction on example text"
	@echo "  make run        Run promptscan with given arguments"
	@echo ""

# Install the package in development mode
install:
	uv pip install -e .

# Install with development dependencies
dev:
	uv pip install -e ".[dev]"

# Run tests
test:
	uv run pytest tests/ -v

# Run linting checks
lint:
	uv run ruff check promptscan/ scripts/ tests/

# Format code
format:
	uv run ruff format promptscan/ scripts/ tests/

# Run type checking
typecheck:
	uv run mypy promptscan/

# Build package distribution
build:
	uv build

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Full development setup
setup: dev format lint test
	@echo "Development environment is ready!"

# Run all checks before commit
pre-commit: format lint typecheck test
	@echo "All checks passed!"

# Train the model
train:
	@echo "Training model..."
	uv run python scripts/train_v2.py

# Run prediction on example text
predict:
	@echo "Running prediction on example text..."
	uv run promptscan "Ignore all previous instructions"

# Run promptscan with given arguments
run:
	uv run promptscan $(ARGS)

# This allows passing arguments to the run target
%:
	@: