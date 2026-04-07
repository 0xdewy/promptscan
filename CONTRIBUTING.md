# Contributing to Prompt Detective

Thank you for your interest in contributing to Prompt Detective! This guide will help you get started with development using `uv`, our preferred package manager.

## Development Setup with uv

### Prerequisites

1. **Install uv**: If you don't have `uv` installed, you can install it with:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/promptscan.git
   cd promptscan
   ```

### Setting Up the Development Environment

1. **Create and activate a virtual environment**:
   ```bash
   # Create virtual environment
   uv venv .venv
   
   # Activate it (Linux/Mac)
   source .venv/bin/activate
   
   # Or on Windows
   .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   # Install the package in development mode with all dependencies
   uv pip install -e ".[dev]"
   
   # Or if you prefer to sync the lock file
   uv sync --dev
   ```

### Common Development Tasks

#### Running Tests
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_detector.py

# Run with coverage
uv run pytest --cov=prompt_detective tests/
```

#### Code Quality
```bash
# Format code with black
uv run black prompt_detective/ scripts/ tests/

# Lint with ruff
uv run ruff check prompt_detective/ scripts/ tests/

# Fix linting issues automatically
uv run ruff check --fix prompt_detective/ scripts/ tests/

# Type checking with mypy
uv run mypy prompt_detective/
```

#### Building the Package
```bash
# Build wheel and sdist
uv build

# Build wheel only
uv build --wheel

# Build sdist only
uv build --sdist
```

#### Adding Dependencies
```bash
# Add a runtime dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Add with specific version
uv add "package-name>=1.0.0"

# Update the lock file after adding dependencies
uv sync
```

#### Running the CLI
```bash
# Using uv run
uv run promptscan --version
uv run promptscan predict "Test text"

# Or activate the virtual environment first
source .venv/bin/activate
promptscan --version
```

### Project Structure

- `prompt_detective/` - Main package source code
- `tests/` - Test files
- `scripts/` - Utility scripts
- `data/` - Data files (included in package)
- `models/` - Model files (included in package)
- `config/` - Configuration files
- `pyproject.toml` - Package configuration (replaces setup.py and requirements.txt)

### Code Style

We use:
- **Black** for code formatting (line length: 88)
- **Ruff** for linting
- **Mypy** for type checking (optional but recommended)

Please run `black` and `ruff check --fix` before submitting pull requests.

### Testing Guidelines

1. Write tests for new functionality
2. Ensure existing tests pass
3. Test edge cases and error conditions
4. Include both unit tests and integration tests where appropriate

### Submitting Changes

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Run tests and code quality checks
4. Submit a pull request with a clear description of the changes

### Release Process

1. Update version in `prompt_detective/__init__.py`
2. Update `CHANGELOG.md` (if available)
3. Build and test the package:
   ```bash
   uv build
   uv run twine check dist/*
   ```
4. Tag the release in git
5. Publish to PyPI (maintainers only):
   ```bash
   uv publish
   ```

## Need Help?

- Check the [README.md](README.md) for general usage
- Open an issue for bugs or feature requests
- Reach out to the maintainers for questions

Happy coding! 🕵️‍♂️