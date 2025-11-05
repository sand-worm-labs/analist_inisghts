"""
Shared utility functions for data processing and analysis.
"""

import re
from pathlib import Path
from typing import Set, List, Tuple
import pyarrow.parquet as pq
import logging


def setup_logger(name: str = "config_logger", log_file: Path = None, debug: bool = False) -> logging.Logger:
    """Setup a logger that logs to console and optionally to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

    return logger


def ensure_dirs(*paths: str):
    """Ensure that all provided directories exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase, collapsing whitespace,
    and replacing literals with placeholders.
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)               # Collapse whitespace
    text = re.sub(r"'[^']*'", "'<val>'", text)    # Replace string literals
    text = re.sub(r"\b0x[a-f0-9]+\b", "<address>", text)  # Replace hex addresses
    return text.strip()


def get_saved_ids(data_dir: Path, logger: logging.Logger = None) -> Set[int]:
    """
    Scan all Parquet files in a directory and extract saved query IDs.
    """
    saved_ids = set()
    for parquet_file in data_dir.glob("*.parquet"):
        try:
            table = pq.read_table(parquet_file)
            if "query_id" in table.schema.names:
                saved_ids.update(table.column("query_id").to_pylist())
        except Exception as e:
            if logger:
                logger.warning("Failed to read %s: %s", parquet_file, e)
            else:
                print(f"[WARN] Failed to read {parquet_file}: {e}")
    return saved_ids


def find_missing_ids(start_id: int, end_id: int, saved_ids: Set[int], logger: logging.Logger = None) -> List[int]:
    """
    Find missing IDs in a range and return them sorted.
    """
    expected_ids = set(range(start_id, end_id + 1))
    missing_ids = sorted(expected_ids - saved_ids)

    if logger:
        logger.info("Found %d missing IDs in range %d-%d", len(missing_ids), start_id, end_id)
    else:
        print(f"[INFO] Found {len(missing_ids)} missing IDs in range {start_id}-{end_id}")

    return missing_ids


def group_consecutive_ids(ids: List[int]) -> List[Tuple[int, int]]:
    """
    Group consecutive IDs into ranges for efficient batch processing.
    """
    if not ids:
        return []

    ranges = []
    start = end = ids[0]

    for current in ids[1:]:
        if current == end + 1:
            end = current
        else:
            ranges.append((start, end))
            start = end = current

    ranges.append((start, end))
    return ranges
