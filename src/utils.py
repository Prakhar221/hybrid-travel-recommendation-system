import os
import logging
from pathlib import Path
import joblib
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TravelRecommender")

# Project Directory Structure
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
FEEDBACK_DB_PATH = DATA_DIR / "user_feedback.csv"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Return project root directory path."""
    return ROOT_DIR


def save_model(model, filename: str) -> Path:
    """Save trained model to models directory using joblib."""
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    logger.info(f"Model saved successfully to {filepath}")
    return filepath


def load_model(filename: str):
    """Load model artifact from models directory."""
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        logger.warning(f"Model file {filepath} not found.")
        return None
    model = joblib.load(filepath)
    logger.info(f"Loaded model from {filepath}")
    return model


def log_user_feedback(user_id: str, business_id: str, feedback_type: str, rating: float = None):
    """Store user feedback (like, dislike, rating) locally."""
    feedback_entry = pd.DataFrame([{
        "user_id": user_id,
        "business_id": business_id,
        "feedback_type": feedback_type,
        "rating": rating,
        "timestamp": pd.Timestamp.now().isoformat()
    }])
    
    if FEEDBACK_DB_PATH.exists():
        feedback_entry.to_csv(FEEDBACK_DB_PATH, mode="a", header=False, index=False)
    else:
        feedback_entry.to_csv(FEEDBACK_DB_PATH, mode="w", header=True, index=False)
    logger.info(f"Feedback logged for user {user_id} on item {business_id}")
