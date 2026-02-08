"""
Signature building for query and CTE embedding.

Signatures are concatenated text strings optimized for
semantic embedding and clustering.
"""

from typing import Dict, Any, List, Optional


def build_query_signature(features: Dict[str, Any]) -> str:
    """
    Build a semantic signature for a query.
    
    Format:
        "{name} | {header_comments} | {tables} | {cte_names} | {cte_comments} | {output_columns} | {patterns}"
    
    Args:
        features: Feature dictionary from extract_query_features
        
    Returns:
        Signature string for embedding
    """
    parts = []
    
    # 1. Query name
    metadata = features.get("metadata", {})
    name = metadata.get("name", "")
    if name:
        parts.append(normalize_for_signature(name))
    
    # 2. Header comments
    header_comments = features.get("header_comments", [])
    if header_comments:
        parts.append(" ".join(normalize_for_signature(c) for c in header_comments))
    
    # 3. Tables (non-CTE, non-query refs)
    tables = features.get("tables", [])
    table_names = [
        t["full_name"] for t in tables 
        if not t.get("is_cte_ref") and not t.get("is_query_ref")
    ]
    if table_names:
        parts.append(" ".join(table_names))
    
    # 4. CTE names
    ctes = features.get("ctes", [])
    cte_names = [cte.get("name", "") for cte in ctes if cte.get("name")]
    if cte_names:
        parts.append(" ".join(cte_names))
    
    # 5. CTE comments
    cte_comments = []
    for cte in ctes:
        comments = cte.get("comments", {})
        if isinstance(comments, dict):
            before = comments.get("before")
            if before:
                cte_comments.append(normalize_for_signature(before))
    if cte_comments:
        parts.append(" ".join(cte_comments))
    
    # 6. Output columns
    final_select = features.get("final_select", {})
    output_cols = final_select.get("columns", {}).get("output", [])
    if output_cols:
        # Filter out * and complex expressions
        clean_cols = [c for c in output_cols if c != "*" and len(c) < 50]
        if clean_cols:
            parts.append(" ".join(clean_cols))
    
    # 7. Detected patterns
    intent = features.get("intent_signals", {})
    patterns = intent.get("detected_patterns", [])
    if patterns:
        parts.append(" ".join(patterns))
    
    # 8. Domain hints
    domain_hints = intent.get("domain_hints", [])
    if domain_hints:
        parts.append(" ".join(domain_hints))
    
    # Join with separator
    signature = " | ".join(p for p in parts if p)
    
    return signature.lower().strip()


def build_cte_signature(cte: Dict[str, Any]) -> str:
    """
    Build a semantic signature for a single CTE.
    
    Format:
        "{cte_name} | {tables} | {output_columns} | {functions} | {comment}"
    
    Args:
        cte: CTE dictionary from extract_ctes
        
    Returns:
        Signature string for embedding
    """
    parts = []
    
    # 1. CTE name
    name = cte.get("name", "")
    if name:
        parts.append(normalize_for_signature(name))
    
    # 2. Tables used
    tables = cte.get("tables", [])
    if tables:
        parts.append(" ".join(tables))
    
    # 3. Output columns
    columns = cte.get("columns", {})
    output_cols = columns.get("output", [])
    if output_cols:
        clean_cols = [c for c in output_cols if c != "*" and len(c) < 50]
        if clean_cols:
            parts.append(" ".join(clean_cols))
    
    # 4. Functions/operations
    operations = cte.get("operations", {})
    functions = []
    functions.extend(operations.get("aggregations", []))
    functions.extend(operations.get("window_functions", []))
    
    if operations.get("unions"):
        functions.append("UNION")
    if operations.get("distinct"):
        functions.append("DISTINCT")
    
    if functions:
        parts.append(" ".join(functions))
    
    # 5. Comment
    comments = cte.get("comments", {})
    if isinstance(comments, dict):
        before = comments.get("before")
        if before:
            parts.append(normalize_for_signature(before))
    
    # 6. Intent signals
    intent_signals = cte.get("intent_signals", [])
    if intent_signals:
        parts.append(" ".join(intent_signals))
    
    # Join with separator
    signature = " | ".join(p for p in parts if p)
    
    return signature.lower().strip()


def normalize_for_signature(text: str) -> str:
    """
    Normalize text for inclusion in signature.
    
    - Lowercase
    - Collapse whitespace
    - Remove special characters
    - Limit length
    """
    import re
    
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Replace underscores with spaces (for readability)
    text = text.replace("_", " ")
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Limit length
    if len(text) > 200:
        text = text[:200]
    
    return text.strip()


def build_batch_signatures(
    features_list: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Build signatures for a batch of queries.
    
    Args:
        features_list: List of feature dictionaries
        
    Returns:
        List of dictionaries with query_id and signature
    """
    results = []
    
    for features in features_list:
        query_id = features.get("query_id")
        signature = build_query_signature(features)
        
        # Also build CTE signatures
        cte_signatures = []
        for cte in features.get("ctes", []):
            cte_sig = build_cte_signature(cte)
            if cte_sig:
                cte_signatures.append({
                    "name": cte.get("name"),
                    "signature": cte_sig,
                })
        
        results.append({
            "query_id": query_id,
            "query_signature": signature,
            "cte_signatures": cte_signatures,
        })
    
    return results