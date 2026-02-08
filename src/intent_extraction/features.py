"""
Feature extraction from SQL queries.

Combines parsing, comment extraction, and pattern detection
to produce a complete feature object for each query.
"""

from typing import Dict, Any, Optional, List

from src.intent_extraction.parser import (
    parse_sql,
    extract_tables,
    extract_ctes,
    extract_final_select,
    calculate_complexity,
)
from src.intent_extraction.comments import (
    extract_comments,
    get_cte_comment,
)
from src.intent_extraction.patterns import (
    detect_intent_patterns,
    detect_cte_intent,
    detect_domain_hints,
    detect_output_type,
)


def extract_query_features(query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract comprehensive features from a query object.
    
    Args:
        query: Dictionary with query_id, name, description, tags, query_sql
        
    Returns:
        Feature dictionary matching the extraction schema
    """
    query_id = query.get("query_id")
    name = query.get("name", "")
    description = query.get("description", "")
    tags = query.get("tags", [])
    owner = query.get("owner", "")
    sql = query.get("query_sql", "")
    
    # Initialize result
    result = {
        "query_id": query_id,
        "metadata": {
            "name": name,
            "description": description,
            "tags": tags,
            "owner": owner,
        },
        "parameters": extract_parameters(sql),
        "header_comments": [],
        "tables": [],
        "ctes": [],
        "final_select": {},
        "complexity": {},
        "intent_signals": {
            "detected_patterns": [],
            "domain_hints": [],
            "output_type": "unknown",
        },
    }
    
    # Parse SQL
    ast = parse_sql(sql)
    
    if ast is None:
        # Return minimal result if parsing fails
        result["parse_error"] = True
        return result
    
    # Extract comments
    comments = extract_comments(sql)
    result["header_comments"] = comments.get("header_comments", [])
    cte_comments = comments.get("cte_comments", {})
    
    # Extract tables
    result["tables"] = extract_tables(ast)
    
    # Extract CTEs
    ctes = extract_ctes(ast)
    
    # Enrich CTEs with comments and intent signals
    for cte in ctes:
        cte_name = cte.get("name", "")
        
        # Add comments
        cte["comments"] = {
            "before": get_cte_comment(cte_name, cte_comments),
            "inline": [],
            "column_annotations": {},
        }
        
        # Detect CTE-level intent
        cte["intent_signals"] = detect_cte_intent(cte)
    
    result["ctes"] = ctes
    
    # Extract final SELECT
    result["final_select"] = extract_final_select(ast)
    
    # Calculate complexity
    result["complexity"] = calculate_complexity(ast, ctes)
    
    # Detect overall intent patterns
    result["intent_signals"] = {
        "detected_patterns": detect_intent_patterns(result),
        "domain_hints": detect_domain_hints(result),
        "output_type": detect_output_type(result),
    }
    
    return result


def extract_parameters(sql: str) -> List[Dict[str, Any]]:
    """
    Extract query parameters ({{param_name}} syntax).
    
    Args:
        sql: Raw SQL string
        
    Returns:
        List of parameter dictionaries
    """
    import re
    
    parameters = []
    seen = set()
    
    # Pattern: {{param_name}} or {{param_name:type}}
    pattern = r'\{\{([^}]+)\}\}'
    
    for match in re.finditer(pattern, sql):
        param_str = match.group(1).strip()
        
        # Parse param_name and optional type
        if ':' in param_str:
            name, param_type = param_str.split(':', 1)
            name = name.strip()
            param_type = param_type.strip()
        else:
            name = param_str
            param_type = "text"
        
        if name not in seen:
            seen.add(name)
            parameters.append({
                "name": name,
                "type": param_type,
                "default_value": None,
            })
    
    return parameters


def extract_query_features_batch(
    queries: List[Dict[str, Any]], 
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    Extract features from multiple queries in parallel.
    
    Args:
        queries: List of query dictionaries
        max_workers: Number of parallel workers
        
    Returns:
        List of feature dictionaries
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_query_features, q): q.get("query_id")
            for q in queries
        }
        
        with tqdm(total=len(futures), desc="Extracting features") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    query_id = futures[future]
                    results.append({
                        "query_id": query_id,
                        "error": str(e),
                    })
                pbar.update(1)
    
    return results