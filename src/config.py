# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DB_CONFIG = {
    "host": os.environ.get("MINNEO_DB_HOST", "localhost"),
    "port": int(os.environ.get("MINNEO_DB_PORT", "5432")),
    "database": os.environ.get("MINNEO_DB_NAME", "minneo_db"),
    "user": os.environ.get("MINNEO_DB_USER", "postgres"),
    "password": os.environ.get("MINNEO_DB_PASSWORD", "")
}

MODEL_CONFIG = {
    "embedding_dim": 64,
    "hidden_dim": 128,
    "tree_conv_layers": 3,
    "fc_layers": 3,
    "learning_rate": 0.001,
    "batch_size": 32
}

SEARCH_CONFIG = {
    "time_cutoff_ms": 250,
    "max_plans_explored": 1000
}