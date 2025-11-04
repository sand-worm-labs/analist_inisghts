from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

# Access variables with defaults
DUNE_API_KEYS = os.getenv("DUNE_API_KEYS", "").split(",")  # list of keys
DATA_PATH = os.getenv("DATA_PATH", "./data")
PROGRAM_CURSOR = os.getenv("PROGRAM_CURSOR", "./cursor")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Ensure directories exist
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(PROGRAM_CURSOR).mkdir(parents=True, exist_ok=True)

def show_config():
    print(f"DUNE_API_KEYS={'SET' if DUNE_API_KEYS else 'Not set'}")
    print(f"DATA_PATH={DATA_PATH}")
    print(f"PROGRAM_CURSOR={PROGRAM_CURSOR if 'PROGRAM_CURSOR' in globals() else 'Not set'}")
    print(f"DEBUG={DEBUG}")


if __name__ == "__main__":
    show_config()
