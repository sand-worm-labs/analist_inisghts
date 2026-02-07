"""
Utility modules for data processing and analysis.

Provides backward-compatible imports from the original utils.py
"""

from src.utils.io import (
    ensure_dirs,
    get_saved_ids,
    find_missing_ids,
    group_consecutive_ids,
    get_query_objects,
)

from src.utils.logging import setup_logger

from src.utils.normalize import (
    clean_sql,
    normalize_sql,
    normalize_text,
    compute_hash,
)

__all__ = [
    # io
    "ensure_dirs",
    "get_saved_ids",
    "find_missing_ids",
    "group_consecutive_ids",
    "get_query_objects",
    # logging
    "setup_logger",
    # normalize
    "clean_sql",
    "normalize_sql",
    "normalize_text",
    "compute_hash",
]