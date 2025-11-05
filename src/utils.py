"""
Shared utility functions for data processing and analysis.
"""
import re
import pyarrow.parquet as pq
from pathlib import Path
from typing import Set, List


def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase, collapsing whitespace,
    and replacing literals with placeholders.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    text = re.sub(r"'[^']*'", "'<val>'", text)  # Replace string literals
    text = re.sub(r"\b0x[a-f0-9]+\b", "<address>", text)  # Replace hex addresses
    return text.strip()


def get_saved_ids(data_dir: Path) -> Set[int]:
    """
    Scan all parquet files in a directory and extract saved query IDs.
    
    Args:
        data_dir: Directory containing parquet files
        
    Returns:
        Set of all saved query IDs
    """
    saved_ids = set()
    
    for parquet_file in data_dir.glob("*.parquet"):
        try:
            table = pq.read_table(parquet_file)
            if "query_id" in table.schema.names:
                saved_ids.update(table.column("query_id").to_pylist())
        except Exception as e:
            print(f"[WARN] Failed to read {parquet_file}: {e}")
    
    return saved_ids


def find_missing_ids(start_id: int, end_id: int, saved_ids: Set[int]) -> List[int]:
    """
    Find missing IDs in a range.
    
    Args:
        start_id: Start of ID range (inclusive)
        end_id: End of ID range (inclusive)
        saved_ids: Set of IDs that have been saved
        
    Returns:
        Sorted list of missing IDs
    """
    expected_ids = set(range(start_id, end_id + 1))
    missing_ids = sorted(expected_ids - saved_ids)
    print(f"[INFO] Found {len(missing_ids)} missing IDs in range {start_id}-{end_id}")
    return missing_ids


def group_consecutive_ids(ids: List[int]) -> List[tuple]:
    """
    Group consecutive IDs into ranges for efficient batch processing.
    
    Args:
        ids: Sorted list of IDs
        
    Returns:
        List of (start, end) tuples representing consecutive ranges
    """
    if not ids:
        return []
    
    ranges = []
    range_start = ids[0]
    range_end = ids[0]
    
    for i in range(1, len(ids)):
        if ids[i] == range_end + 1:
            range_end = ids[i]
        else:
            ranges.append((range_start, range_end))
            range_start = ids[i]
            range_end = ids[i]
    
    ranges.append((range_start, range_end))
    return ranges