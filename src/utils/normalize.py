"""
Shared utility functions for Sandworm query processing.

Contains:
- SQL cleaning and normalization
- Text preparation for embeddings
- File I/O helpers
- Hashing utilities
- Query object loaders
"""

import re
import hashlib
import logging
from pathlib import Path
from typing import Set, List, Tuple, Dict, Optional, Any

# Optional: pyarrow for parquet support
try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


# =============================================================================
# SQL Cleaning and Normalization
# =============================================================================

def clean_sql(sql: str) -> str:
    """
    Remove SQL comments from a query string.
    Handles both -- single-line and /* */ multi-line comments.
    
    Args:
        sql: Raw SQL string
        
    Returns:
        SQL with comments removed
    """
    if not sql:
        return ""
    
    # Remove multi-line comments /* ... */
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    
    # Remove single-line comments -- ... (till end of line)
    sql = re.sub(r'--.*?$', '', sql, flags=re.MULTILINE)
    
    # Also handle # comments (some dialects)
    sql = re.sub(r'#.*?$', '', sql, flags=re.MULTILINE)
    
    return sql.strip()


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison/deduplication.
    
    - Removes comments
    - Normalizes whitespace
    - Lowercases
    
    Args:
        sql: Raw SQL string
        
    Returns:
        Normalized SQL string
    """
    if not sql:
        return ""
    
    sql = clean_sql(sql)
    sql = sql.lower()
    sql = re.sub(r'\s+', ' ', sql)
    
    return sql.strip()


def normalize_sql_for_embedding(sql: str) -> str:
    """
    Normalize SQL for embedding generation.
    
    - Removes comments
    - Normalizes whitespace
    - Preserves keywords and structure
    - Replaces literals with placeholders
    
    Args:
        sql: Raw SQL string
        
    Returns:
        Normalized SQL ready for embedding
    """
    if not sql:
        return ""
    
    sql = clean_sql(sql)
    
    # Normalize string literals -> <STR>
    sql = re.sub(r"'[^']*'", "<STR>", sql)
    
    # Normalize numeric literals -> <NUM>
    # Be careful not to match table names like erc20
    sql = re.sub(r'\b\d+\.?\d*\b', '<NUM>', sql)
    
    # Normalize hex addresses -> <ADDR>
    sql = re.sub(r'0x[a-fA-F0-9]+', '<ADDR>', sql)
    
    # Normalize parameter placeholders {{param}} -> <PARAM>
    sql = re.sub(r'\{\{[^}]+\}\}', '<PARAM>', sql)
    
    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql)
    
    # Lowercase for consistency
    sql = sql.lower().strip()
    
    return sql


def normalize_sql_for_signature(sql: str) -> str:
    """
    Normalize SQL for signature/fingerprinting.
    
    More aggressive normalization than embedding:
    - All literals replaced
    - IN lists collapsed
    - Column lists collapsed
    
    Args:
        sql: Raw SQL string
        
    Returns:
        Normalized SQL for signature generation
    """
    if not sql:
        return ""
    
    sql = normalize_sql_for_embedding(sql)
    
    # Collapse IN lists with many items
    sql = re.sub(r'\bin\s*\([^)]{100,}\)', 'IN (?)', sql)
    
    # Collapse long SELECT column lists
    sql = re.sub(r'select\s+[^f]{200,}?\s+from', 'SELECT ... FROM', sql, flags=re.IGNORECASE)
    
    return sql


# =============================================================================
# Text Preparation for Embeddings
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize general text (names, descriptions, tags).
    
    Args:
        text: Raw text string
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    return text


def prepare_semantic_text(query: Dict[str, Any]) -> str:
    """
    Prepare semantic text for embedding.
    Combines name, description, owner, and tags.
    
    Args:
        query: Query dictionary
        
    Returns:
        Combined text string for semantic embedding
    """
    parts = []
    
    if query.get('name'):
        parts.append(query['name'])
    
    if query.get('description'):
        parts.append(query['description'])
    
    if query.get('owner'):
        parts.append(f"by {query['owner']}")
    
    tags = query.get('tags', [])
    if tags:
        parts.append(' '.join(tags))
    
    return ' '.join(parts)


def prepare_sql_text(query: Dict[str, Any]) -> str:
    """
    Prepare SQL text for embedding.
    
    Args:
        query: Query dictionary with query_sql field
        
    Returns:
        Normalized SQL string for embedding
    """
    sql = query.get('query_sql', '')
    return normalize_sql_for_embedding(sql)


# =============================================================================
# Hashing Utilities
# =============================================================================

def compute_hash(value: str, algorithm: str = 'sha1') -> str:
    """
    Compute hash of a string for fast comparison.
    
    Args:
        value: String to hash
        algorithm: Hash algorithm ('sha1', 'md5', 'sha256')
        
    Returns:
        Hex digest of hash
    """
    if algorithm == 'md5':
        return hashlib.md5(value.encode('utf-8')).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(value.encode('utf-8')).hexdigest()
    else:  # default sha1
        return hashlib.sha1(value.encode('utf-8')).hexdigest()


def compute_sql_hash(sql: str) -> str:
    """
    Compute hash of normalized SQL for deduplication.
    
    Args:
        sql: Raw SQL string
        
    Returns:
        MD5 hash of normalized SQL
    """
    normalized = normalize_sql(sql)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


# =============================================================================
# File I/O Helpers
# =============================================================================

def ensure_dirs(*paths: str):
    """
    Ensure that all provided directories exist.
    
    Args:
        *paths: Directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_saved_ids(data_dir: Path, logger: Optional[logging.Logger] = None) -> Set[int]:
    """
    Scan all Parquet files in a directory and extract saved query IDs.
    
    Args:
        data_dir: Directory containing parquet files
        logger: Optional logger for warnings
        
    Returns:
        Set of query IDs found
    """
    if not HAS_PYARROW:
        if logger:
            logger.warning("pyarrow not installed, cannot read parquet files")
        return set()
    
    saved_ids = set()
    data_dir = Path(data_dir)
    
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


def find_missing_ids(
    start_id: int,
    end_id: int,
    saved_ids: Set[int],
    logger: Optional[logging.Logger] = None
) -> List[int]:
    """
    Find missing IDs in a range and return them sorted.
    
    Args:
        start_id: Start of ID range
        end_id: End of ID range (inclusive)
        saved_ids: Set of IDs already saved
        logger: Optional logger
        
    Returns:
        Sorted list of missing IDs
    """
    expected_ids = set(range(start_id, end_id + 1))
    missing_ids = sorted(expected_ids - saved_ids)
    
    if logger:
        logger.info("Found %d missing IDs in range %d-%d", len(missing_ids), start_id, end_id)
    
    return missing_ids


def group_consecutive_ids(ids: List[int]) -> List[Tuple[int, int]]:
    """
    Group consecutive IDs into ranges for efficient batch processing.
    
    Args:
        ids: List of IDs (will be sorted)
        
    Returns:
        List of (start, end) tuples representing consecutive ranges
    """
    if not ids:
        return []
    
    ids = sorted(ids)
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
    limit: Optional[int] = None,
    normalize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load query objects from parquet files.
    
    Args:
        data_dir: Directory containing parquet files
        limit: Maximum number of queries to load (None for all)
        normalize: Whether to normalize text fields
        
    Returns:
        List of query dictionaries
    """
    if not HAS_PYARROW:
        print("[ERROR] pyarrow not installed, cannot read parquet files")
        return []
    
    queries = []
    data_dir = Path(data_dir)
    
    print(f"[INFO] Loading queries from {data_dir}...")
    
    # Use sorted for consistent ordering
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
                
                query = {
                    "query_id": qid,
                    "name": normalize_text(name) if normalize and name else (name or ""),
                    "description": normalize_text(desc) if normalize and desc else (desc or ""),
                    "tags": [t.lower().strip() for t in tags] if tags else [],
                    "owner": owner or "",
                    "query_sql": sql or "",
                }
                
                queries.append(query)
        
        except Exception as e:
            print(f"[WARN] Failed to read {parquet_file}: {e}")
    
    print(f"[INFO] Loaded {len(queries)} queries")
    return queries    