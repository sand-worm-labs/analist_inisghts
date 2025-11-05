"""
Configuration management for Dune Analytics query collection.
Loads environment variables and ensures required directories exist.
"""
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
DUNE_API_KEYS = [key.strip() for key in os.getenv("DUNE_API_KEYS", "").split(",") if key.strip()]
DATA_PATH = os.getenv("DATA_PATH", "./data")
PROGRAM_CURSOR = os.getenv("PROGRAM_CURSOR", "./cursor")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"


Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(PROGRAM_CURSOR).mkdir(parents=True, exist_ok=True)

def validate_config():
    """Validate that required configuration is present."""
    if not DUNE_API_KEYS or DUNE_API_KEYS == ['']:
        raise ValueError("DUNE_API_KEYS must be set in .env file")
    return True

def show_config():
    """Display current configuration (for debugging)."""
    print(f"DUNE_API_KEYS: {len(DUNE_API_KEYS)} keys loaded")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"PROGRAM_CURSOR: {PROGRAM_CURSOR}")
    print(f"DEBUG: {DEBUG}")

if __name__ == "__main__":
    try:
        validate_config()
        show_config()
    except ValueError as e:
        print(f"Configuration error: {e}")