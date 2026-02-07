"""
Text and SQL normalization utilities.

Used for:
- Cleaning SQL queries (removing comments)
- Normalizing text for comparison/hashing
- Computing stable hashes for deduplication
"""

import re
import hashlib


def clean_sql(sql: str) -> str:
    """
    Remove SQL comments from a query string.
    
    Handles:
    - Single-line comments: -- comment
    - Multi-line comments: /* comment */
    
    Args:
        sql: Raw SQL query string
        
    Returns:
        SQL with comments removed
        
    Example:
        >>> clean_sql("SELECT * -- get all\\nFROM table")
        'SELECT * \\nFROM table'
    """
    if not sql:
        return ""
    
    # Remove multi-line comments /* ... */
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    
    # Remove single-line comments -- ... (till end of line)
    sql = re.sub(r'--.*?$', '', sql, flags=re.MULTILINE)
    
    return sql.strip()


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison and hashing.
    
    Performs:
    - Comment removal
    - Lowercase conversion
    - Whitespace normalization
    
    Args:
        sql: Raw SQL query string
        
    Returns:
        Normalized SQL string
        
    Example:
        >>> normalize_sql("SELECT  *  FROM  table -- comment")
        'select * from table'
    """
    if not sql:
        return ""
    
    # Remove comments
    sql = clean_sql(sql)
    
    # Also handle # comments (MySQL style)
    sql = re.sub(r'#.*?(\r?\n|$)', ' ', sql)
    
    # Normalize whitespace and lowercase
    sql = sql.lower()
    sql = re.sub(r'\s+', ' ', sql)
    
    return sql.strip()


def normalize_text(text: str) -> str:
    """
    Normalize general text for comparison.
    
    Performs:
    - Lowercase conversion
    - Whitespace normalization
    - Strip leading/trailing whitespace
    
    Args:
        text: Raw text string
        
    Returns:
        Normalized text string
        
    Example:
        >>> normalize_text("  Hello   World  ")
        'hello world'
    """
    if not text:
        return ""
    
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    return text


def compute_hash(value: str) -> str:
    """
    Compute stable SHA1 hash for fast comparison.
    
    Args:
        value: String to hash
        
    Returns:
        40-character hexadecimal hash string
        
    Example:
        >>> compute_hash("hello")
        'aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d'
    """
    return hashlib.sha1(value.encode("utf-8")).hexdigest()