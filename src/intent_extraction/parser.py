"""
SQL comment extraction utilities.

Extracts:
- Header comments (before WITH/SELECT)
- CTE-level comments (before each CTE)
- Inline comments within statements
- Column annotations
"""

import re
from typing import Dict, List, Optional, Tuple


def extract_comments(sql: str) -> Dict[str, any]:
    """
    Extract all comments from SQL query.
    
    Args:
        sql: Raw SQL string
        
    Returns:
        Dictionary with:
        - header_comments: Comments before main query
        - cte_comments: Dict mapping CTE names to their comments
        - inline_comments: List of inline comments
        - all_comments: List of all comments found
    """
    if not sql:
        return {
            "header_comments": [],
            "cte_comments": {},
            "inline_comments": [],
            "all_comments": [],
        }
    
    result = {
        "header_comments": [],
        "cte_comments": {},
        "inline_comments": [],
        "all_comments": [],
    }
    
    # Extract all comments
    all_comments = extract_all_comments(sql)
    result["all_comments"] = all_comments
    
    # Extract header comments (before WITH or SELECT)
    result["header_comments"] = extract_header_comments(sql)
    
    # Extract CTE-level comments
    result["cte_comments"] = extract_cte_comments(sql)
    
    # Inline comments are those not captured above
    header_set = set(result["header_comments"])
    cte_set = set()
    for comments in result["cte_comments"].values():
        if isinstance(comments, dict):
            if comments.get("before"):
                cte_set.add(comments["before"])
            cte_set.update(comments.get("inline", []))
        elif isinstance(comments, str):
            cte_set.add(comments)
    
    result["inline_comments"] = [
        c for c in all_comments 
        if c not in header_set and c not in cte_set
    ]
    
    return result


def extract_all_comments(sql: str) -> List[str]:
    """
    Extract all comments from SQL (both single-line and block).
    
    Args:
        sql: Raw SQL string
        
    Returns:
        List of comment strings (without -- or /* */ markers)
    """
    comments = []
    
    # Single-line comments: -- comment
    single_line = re.findall(r'--\s*(.+?)(?:\r?\n|$)', sql)
    comments.extend([c.strip() for c in single_line if c.strip()])
    
    # Block comments: /* comment */
    block = re.findall(r'/\*\s*(.*?)\s*\*/', sql, re.DOTALL)
    comments.extend([c.strip() for c in block if c.strip()])
    
    return comments


def extract_header_comments(sql: str) -> List[str]:
    """
    Extract comments that appear before the main query (WITH or SELECT).
    
    Args:
        sql: Raw SQL string
        
    Returns:
        List of header comment strings
    """
    comments = []
    
    # Find where the main query starts
    # Look for WITH or SELECT (case insensitive)
    match = re.search(r'\b(WITH|SELECT)\b', sql, re.IGNORECASE)
    
    if match:
        header = sql[:match.start()]
        
        # Extract comments from header
        single_line = re.findall(r'--\s*(.+?)(?:\r?\n|$)', header)
        comments.extend([c.strip() for c in single_line if c.strip()])
        
        block = re.findall(r'/\*\s*(.*?)\s*\*/', header, re.DOTALL)
        comments.extend([c.strip() for c in block if c.strip()])
    
    return comments


def extract_cte_comments(sql: str) -> Dict[str, Dict[str, any]]:
    """
    Extract comments associated with each CTE.
    
    Looks for patterns like:
    - -- Comment before CTE
      cte_name AS (
    - /* Block comment */
      cte_name AS (
    - cte_name AS ( -- inline comment
    
    Args:
        sql: Raw SQL string
        
    Returns:
        Dictionary mapping CTE names to their comments:
        {
            "cte_name": {
                "before": "comment before CTE",
                "inline": ["inline comments"]
            }
        }
    """
    cte_comments = {}
    
    # Pattern: comment followed by CTE definition
    # Matches: -- comment\n  cte_name AS (
    # Or: /* comment */\n  cte_name AS (
    
    # Pattern for single-line comment before CTE
    pattern_single = r'--\s*([^\n]+)\n\s*,?\s*(\w+)\s+AS\s*\('
    for match in re.finditer(pattern_single, sql, re.IGNORECASE):
        comment = match.group(1).strip()
        cte_name = match.group(2).lower()
        
        if cte_name not in cte_comments:
            cte_comments[cte_name] = {"before": None, "inline": []}
        cte_comments[cte_name]["before"] = comment
    
    # Pattern for block comment before CTE
    pattern_block = r'/\*\s*(.*?)\s*\*/\s*,?\s*(\w+)\s+AS\s*\('
    for match in re.finditer(pattern_block, sql, re.IGNORECASE | re.DOTALL):
        comment = match.group(1).strip()
        cte_name = match.group(2).lower()
        
        if cte_name not in cte_comments:
            cte_comments[cte_name] = {"before": None, "inline": []}
        
        # Prefer block comment if no single-line comment
        if cte_comments[cte_name]["before"] is None:
            cte_comments[cte_name]["before"] = comment
    
    # Find inline comments within CTE bodies
    # This is harder - we'd need to track CTE boundaries
    # For now, just extract section markers
    
    # Section markers like: -- === STEP 1 === or -- --- Aggregation ---
    section_markers = re.findall(r'--\s*[=\-]{3,}\s*([^=\-\n]+?)\s*[=\-]*\s*(?:\r?\n|$)', sql)
    
    return cte_comments


def extract_column_comments(sql: str) -> Dict[str, str]:
    """
    Extract comments that appear after column definitions.
    
    Pattern: column_name -- comment
    
    Args:
        sql: Raw SQL string
        
    Returns:
        Dictionary mapping column names to their comments
    """
    column_comments = {}
    
    # Pattern: identifier followed by -- comment
    # This is approximate - would need context to be accurate
    pattern = r'(\w+)\s+--\s*([^\n]+)'
    
    for match in re.finditer(pattern, sql):
        col_name = match.group(1).lower()
        comment = match.group(2).strip()
        
        # Skip SQL keywords
        keywords = {'select', 'from', 'where', 'and', 'or', 'join', 'on', 
                   'group', 'order', 'by', 'having', 'limit', 'as', 'with'}
        
        if col_name not in keywords:
            column_comments[col_name] = comment
    
    return column_comments


def get_cte_comment(cte_name: str, cte_comments: Dict) -> Optional[str]:
    """
    Get the comment for a specific CTE.
    
    Args:
        cte_name: Name of the CTE
        cte_comments: Dictionary from extract_cte_comments
        
    Returns:
        Comment string or None
    """
    cte_name_lower = cte_name.lower()
    
    if cte_name_lower in cte_comments:
        info = cte_comments[cte_name_lower]
        if isinstance(info, dict):
            return info.get("before")
        return info
    
    return None