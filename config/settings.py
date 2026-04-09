"""
Configuration settings for safe prompts project.
"""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TESTS_DIR = BASE_DIR / "tests"

# File paths
MODEL_PATH = MODELS_DIR / "best_model.pt"

# Parquet data paths
# NOTE: Static split files are deprecated and have been consolidated
# Training uses prompts.parquet with dynamic splits
PROMPTS_PARQUET_PATH = DATA_DIR / "prompts.parquet"  # Consolidated dataset
FEEDBACK_PARQUET_PATH = (
    DATA_DIR / "unverified_user_submissions.parquet"
)  # User feedback

# Test fixtures
TEST_FIXTURES_DIR = TESTS_DIR / "fixtures"
TEST_SAFE_PATH = TEST_FIXTURES_DIR / "test_safe.txt"
TEST_INJECTION_PATH = TEST_FIXTURES_DIR / "test_injection.txt"
TEST_URL_PATH = TEST_FIXTURES_DIR / "url_test.txt"

# External data
EXTERNAL_DATA_DIR = DATA_DIR / "external"
BENIGN_EN_PATH = EXTERNAL_DATA_DIR / "benign_en.csv"
INJECTION_EN_PATH = EXTERNAL_DATA_DIR / "injection_en.csv"
BENIGN_ES_PATH = EXTERNAL_DATA_DIR / "benign_es.csv"
INJECTION_ES_PATH = EXTERNAL_DATA_DIR / "injection_es.csv"

# Model parameters
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 64
NUM_FILTERS = 50
NUM_CLASSES = 2

# Training parameters
BATCH_SIZE = 16  # Reduced from 32 to prevent OOM errors
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
PATIENCE = 3

# Ensure directories exist
for directory in [
    DATA_DIR,
    MODELS_DIR,
    TESTS_DIR,
    TEST_FIXTURES_DIR,
    EXTERNAL_DATA_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
