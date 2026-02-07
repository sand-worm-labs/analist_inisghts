"""
File I/O utilities for parquet handling and directory management.
"""

from pathlib import Path
from typing import Set, List, Tuple, Optional
import logging

import pyarrow.parquet as pq

from src.utils.normalize import normalize_text


def ensure_dirs(*paths: str) -> None:
    """
    Ensure that all provided directories exist.
    
    Args:
        *paths: Variable number of directory paths to create
        
    Example:
        ensure_dirs("./data", "./logs", "./output")
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_saved_ids(
    data_dir: Path, 
    logger: Optional[logging.Logger] = None
) -> Set[int]:
    """
    Scan all Parquet files in a directory and extract saved query IDs.
    
    Args:
        data_dir: Directory containing parquet files
        logger: Optional logger for warnings
        
    Returns:
        Set of query IDs found in parquet files
    """
    saved_ids = set()
    
    for parquet_file in data_dir.glob("*.parquet"):
        try:
            table = pq.read_table(parquet_file)
            if "query_id" in table.schema.names:
                saved_ids.update(table.column("query_id").to_pylist())
        except Exception as e:
            msg = f"Failed to read {parquet_file}: {e}"
            if logger:
                logger.warning(msg)
            else:
                print(f"[WARN] {msg}")
                
    return saved_ids


def find_missing_ids(
    start_id: int, 
    end_id: int, 
    saved_ids: Set[int], 
    logger: Optional[logging.Logger] = None
) -> List[int]:
    """
    Find missing IDs in a range and return them sorted.
    
    Args:
        start_id: Start of ID range (inclusive)
        end_id: End of ID range (inclusive)
        saved_ids: Set of already saved IDs
        logger: Optional logger for info messages
        
    Returns:
        Sorted list of missing IDs
    """
    expected_ids = set(range(start_id, end_id + 1))
    missing_ids = sorted(expected_ids - saved_ids)

    msg = f"Found {len(missing_ids)} missing IDs in range {start_id}-{end_id}"
    if logger:
        logger.info(msg)
    else:
        print(f"[INFO] {msg}")

    return missing_ids


def group_consecutive_ids(ids: List[int]) -> List[Tuple[int, int]]:
    """
    Group consecutive IDs into ranges for efficient batch processing.
    
    Args:
        ids: List of IDs to group
        
    Returns:
        List of (start, end) tuples representing consecutive ranges
        
    Example:
        >>> group_consecutive_ids([1, 2, 3, 7, 8, 15])
        [(1, 3), (7, 8), (15, 15)]
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


def get_query_objects(
    data_dir: Path, 
    limit: Optional[int] = None
) -> List[dict]:
    """
    Load query objects from parquet files with normalized text fields.

    Args:
        data_dir: Directory containing parquet files
        limit: Maximum number of queries to load (None for all)

    Returns:
        List of query dictionaries with fields:
        - query_id: int
        - name: str (normalized)
        - description: str (normalized)
        - tags: List[str] (lowercase)
        - owner: str
        - query_sql: str (raw)
    """
    queries = []
    print(f"[INFO] Loading queries from {data_dir}...")

    for parquet_file in sorted(data_dir.glob("*.parquet")):
        if limit and len(queries) >= limit:
            break

        try:
            table = pq.read_table(parquet_file).to_pydict()

            ids = table.get("query_id", [])
            names = table.get("name", [])
            owners = table.get("owner", [])
            query_sqls = table.get("query_sql", [])
            descriptions = table.get("description", [])
            tags_list = table.get("tags", [])

            for i in range(len(ids)):
                if limit and len(queries) >= limit:
                    break

                queries.append({
                    "query_id": ids[i],
                    "name": normalize_text(names[i]) if i < len(names) and names[i] else "",
                    "description": normalize_text(descriptions[i]) if i < len(descriptions) and descriptions[i] else "",
                    "tags": [t.lower().strip() for t in tags_list[i]] if i < len(tags_list) and tags_list[i] else [],
                    "owner": owners[i] if i < len(owners) else "",
                    "query_sql": query_sqls[i] if i < len(query_sqls) else ""
                })

        except Exception as e:
            print(f"[WARN] Failed to read {parquet_file}: {e}")

    print(f"[INFO] Loaded {len(queries)} queries")
    return queries