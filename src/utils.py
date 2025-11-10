"""
Shared utility functions for data processing and analysis.
"""

import re
from pathlib import Path
from typing import Set, List, Tuple
import pyarrow.parquet as pq
import logging
import hashlib


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


def get_query_objects(data_dir: Path, limit: int = None):
    """
    Load query objects from parquet files with normalized text fields.

    Args:
        data_dir: Directory containing parquet files
        limit: Maximum number of queries to load (None for all)

    Returns:
        List of query dictionaries with normalized text
    """
    queries = []
    print(f"[INFO] Loading queries from {data_dir}...")

    # Use sorted to ensure consistent ordering
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

                qid = ids[i]
                name = names[i] if i < len(names) else ""
                desc = descriptions[i] if i < len(descriptions) else ""
                owner = owners[i] if i < len(owners) else ""
                sql = query_sqls[i] if i < len(query_sqls) else ""
                tags = tags_list[i] if i < len(tags_list) else []

                queries.append({
                    "query_id": qid,
                    "name": normalize_text(name) if name else "",
                    "description": normalize_text(desc) if desc else "",
                    "tags": [t.lower().strip() for t in tags] if tags else [],
                    "owner": owner,
                    "query_sql": sql
                })

        except Exception as e:
            print(f"[WARN] Failed to read {parquet_file}: {e}")

    print(f"[INFO] Loaded {len(queries)} queries")
    return queries
    
def clean_sql(query_sql: str) -> str:
    """
    Remove SQL comments from a query string.
    Handles both -- single-line comments and /* */ multi-line comments.
    """
    # Remove multi-line comments /* ... */
    no_block_comments = re.sub(r'/\*.*?\*/', '', query_sql, flags=re.DOTALL)
    # Remove single-line comments -- ... (till end of line)
    no_line_comments = re.sub(r'--.*?$', '', no_block_comments, flags=re.MULTILINE)
    # Optional: strip extra whitespace
    cleaned_sql = no_line_comments.strip()
    return cleaned_sql

def normalize_sql(sql: str) -> str:
    """Remove SQL comments, normalize whitespace and casing."""
    if not sql:
        return ""
    sql = clean_sql(sql)  # shared utility (handles base cleaning)

    # Remove single-line and block comments
    sql = re.sub(r'--.*?(\r?\n|$)', ' ', sql)
    sql = re.sub(r'#.*?(\r?\n|$)', ' ', sql)
    sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)

    # Normalize whitespace and lowercase
    sql = sql.lower()
    sql = re.sub(r'\s+', ' ', sql)
    return sql.strip()


def normalize_text(text: str) -> str:
    """Normalize general text (semantic fields)."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def compute_hash(value: str) -> str:
    """Compute stable SHA1 hash for fast comparison."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()
