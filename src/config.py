"""
Configuration management for Dune Analytics query collection.
Loads environment variables, ensures required directories exist,
and logs configuration info to both console and file.
"""
from dotenv import load_dotenv
import os
from pathlib import Path
import logging
from src.utils import ensure_dirs, setup_logger

load_dotenv()

class Config:
    """Holds and validates Dune Analytics configuration."""
    
    def __init__(self):
        # Load environment variables
        self.DUNE_API_KEYS = [key.strip() for key in os.getenv("DUNE_API_KEYS", "").split(",") if key.strip()]
        self.DATA_PATH = os.getenv("DATA_PATH", "./data")
        self.LOGS_PATH = os.getenv("LOGS_PATH", "./logs")
        self.PROGRAM_CURSOR = os.getenv("PROGRAM_CURSOR", "./cursor")
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"

        # Ensure directories exist
        ensure_dirs(self.DATA_PATH, self.LOGS_PATH, self.PROGRAM_CURSOR)

        # Setup logger
        log_file_path = Path(self.LOGS_PATH) / "config.log"
        self.logger = setup_logger(log_file=log_file_path, debug=self.DEBUG)

        # Validate configuration
        self.validate()

    def validate(self):
        """Ensure required configuration is present."""
        if not self.DUNE_API_KEYS:
            raise ValueError("DUNE_API_KEYS must be set in .env file and cannot be empty")
        self.logger.debug("Configuration validated successfully.")

    def show(self):
        """Log the current configuration."""
        self.logger.info("Configuration Summary:")
        self.logger.info("DUNE_API_KEYS: %d keys loaded", len(self.DUNE_API_KEYS))
        self.logger.info("DATA_PATH: %s", self.DATA_PATH)
        self.logger.info("LOGS_PATH: %s", self.LOGS_PATH)
        self.logger.info("PROGRAM_CURSOR: %s", self.PROGRAM_CURSOR)
        self.logger.info("DEBUG: %s", self.DEBUG)
        self.logger.info("Logs are being saved to: %s", Path(self.LOGS_PATH) / "config.log")


# === Main execution ===
if __name__ == "__main__":
    try:
        config = Config()
        config.show()
    except ValueError as e:
        print(f"[ERROR] Configuration error: {e}")  # fallback if logger fails
