import logging
from pathlib import Path

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "trained_models" / "processed_data_combined.csv"
TFIDF_PATH = ROOT / "trained_models" / "tfidf_vectorizer.pkl"
SCALER_PATH = ROOT / "trained_models" / "scaler.pkl"
DEFAULT_LIMIT = 20

# Logging configuration
LOG_LEVEL = "INFO"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("shopmind")
