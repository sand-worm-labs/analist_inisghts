"""
Logging utilities for consistent log formatting across modules.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "app_logger", 
    log_file: Optional[Path] = None, 
    debug: bool = False
) -> logging.Logger:
    """
    Setup a logger that logs to console and optionally to a file.
    
    Args:
        name: Logger name (use __name__ for module-specific loggers)
        log_file: Optional path to log file
        debug: If True, set level to DEBUG; otherwise INFO
        
    Returns:
        Configured logger instance
        
    Example:
        logger = setup_logger("my_module", Path("./logs/app.log"), debug=True)
        logger.info("Application started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger