"""
Enhanced intent pattern detection for SQL queries.

Uses multi-signal detection with confidence scoring
to identify CTE and query patterns dynamically.
"""

import re
from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# Pattern Definitions - Comprehensive Blockchain Analytics Patterns
# =============================================================================

@dataclass
class PatternSignal:
    """A single signal that contributes to pattern detection."""
    weight: float  # 0.0 to 1.0
    reason: str


@dataclass 
class PatternMatch:
    """Result of pattern matching with confidence."""
    pattern: str
    confidence: float  # 0.0 to 1.0
    signals: List[PatternSignal] = field(default_factory=list)
    
    def add_signal(self, weight: float, reason: str):
        self.signals.append(PatternSignal(weight, reason))
        # Recalculate confidence (capped at 1.0)
        total = sum(s.weight for s in self.signals)
        self.confidence = min(1.0, total)


# Comprehensive pattern catalog
CTE_PATTERNS = {
    # =========================================================================
    # DATA SCAFFOLDING PATTERNS
    # =========================================================================
    "date_spine": {
        "description": "Generates sequence of dates/times for filling gaps",
        "name_hints": ["date", "day", "time", "calendar", "spine", "series", "sequence", "period"],
        "column_hints": ["dt", "day", "date", "hour", "week", "month", "period", "timestamp"],
        "function_hints": ["SEQUENCE", "UNNEST", "GENERATE_SERIES", "GENERATE_TIMESTAMP_ARRAY"],
        "table_hints": [],
        "operation_hints": {"cross_join": True},
        "anti_hints": {"aggregations": True},  # Date spines rarely aggregate
    },
    
    "block_range": {
        "description": "Generates sequence of blocks",
        "name_hints": ["block", "blocks", "block_range", "block_sequence"],
        "column_hints": ["block_number", "block_num", "block"],
        "function_hints": ["SEQUENCE", "UNNEST"],
    },
    
    # =========================================================================
    # ENTITY DISCOVERY PATTERNS  
    # =========================================================================
    "entity_discovery": {
        "description": "Finds unique entities (wallets, tokens, contracts)",
        "name_hints": ["unique", "distinct", "wallets", "addresses", "entities", "holders", "users", "accounts", "traders"],
        "column_hints": ["address", "wallet", "account", "holder", "user", "trader", "from", "to", "sender", "receiver"],
        "operation_hints": {"distinct": True},
        "table_hints": ["evt_transfer", "transactions", "traces"],
    },
    
    "first_activity": {
        "description": "Finds first occurrence/activity for entities",
        "name_hints": ["first", "earliest", "initial", "genesis", "debut"],
        "column_hints": ["first_", "min_", "earliest_"],
        "function_hints": ["MIN", "FIRST_VALUE"],
        "operation_hints": {"group_by": True},
    },
    
    "last_activity": {
        "description": "Finds most recent activity for entities",
        "name_hints": ["last", "latest", "recent", "current"],
        "column_hints": ["last_", "max_", "latest_", "current_"],
        "function_hints": ["MAX", "LAST_VALUE"],
        "operation_hints": {"group_by": True},
    },
    
    # =========================================================================
    # TOKEN FLOW PATTERNS
    # =========================================================================
    "token_transfers": {
        "description": "Extracts token transfer events",
        "name_hints": ["transfer", "transfers", "sends", "receives", "flow", "movement"],
        "table_hints": ["evt_transfer", "erc20", "spl_transfer", "token_transfers"],
        "column_hints": ["from", "to", "value", "amount", "token"],
    },
    
    "net_flow": {
        "description": "Calculates net token flow (in - out)",
        "name_hints": ["net", "flow", "delta", "change", "movement"],
        "column_hints": ["net_", "delta_", "change_", "inflow", "outflow"],
        "function_hints": ["SUM"],
        "pattern_hints": ["CASE WHEN.*from.*THEN.*ELSE"],  # Net flow calculation
    },
    
    "running_balance": {
        "description": "Calculates cumulative/running balance",
        "name_hints": ["running", "cumulative", "balance", "cumul", "total"],
        "column_hints": ["running_", "cumul_", "balance", "total_balance"],
        "function_hints": ["SUM OVER", "COUNT OVER"],
        "pattern_hints": ["ROWS BETWEEN UNBOUNDED PRECEDING", "ROWS UNBOUNDED PRECEDING"],
    },
    
    # =========================================================================
    # AGGREGATION PATTERNS
    # =========================================================================
    "time_aggregation": {
        "description": "Aggregates data by time period",
        "name_hints": ["daily", "hourly", "weekly", "monthly", "by_day", "by_hour", "per_day"],
        "column_hints": ["day", "hour", "week", "month", "period"],
        "function_hints": ["DATE_TRUNC", "SUM", "COUNT", "AVG"],
        "operation_hints": {"group_by": True, "aggregations": True},
    },
    
    "entity_aggregation": {
        "description": "Aggregates data by entity (wallet, token, etc)",
        "name_hints": ["by_wallet", "by_address", "by_token", "per_user", "user_stats"],
        "column_hints": ["address", "wallet", "token", "user"],
        "operation_hints": {"group_by": True, "aggregations": True},
    },
    
    "counting": {
        "description": "Counts occurrences",
        "name_hints": ["count", "counts", "tally", "frequency"],
        "column_hints": ["count", "cnt", "num_", "n_", "total_count"],
        "function_hints": ["COUNT", "COUNT DISTINCT"],
    },
    
    "summation": {
        "description": "Sums values",
        "name_hints": ["sum", "total", "aggregate"],
        "column_hints": ["sum_", "total_", "amount", "value"],
        "function_hints": ["SUM"],
    },
    
    # =========================================================================
    # WINDOW FUNCTION PATTERNS
    # =========================================================================
    "ranking": {
        "description": "Ranks rows within partitions",
        "name_hints": ["rank", "top", "ranked", "ordered", "leaderboard"],
        "column_hints": ["rank", "rn", "row_num", "position"],
        "function_hints": ["ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE"],
    },
    
    "deduplication": {
        "description": "Removes duplicate rows",
        "name_hints": ["dedup", "dedupe", "unique", "distinct", "latest"],
        "function_hints": ["ROW_NUMBER"],
        "pattern_hints": ["WHERE.*rn.*=.*1", "WHERE.*row_num.*=.*1", "QUALIFY.*=.*1"],
    },
    
    "lag_lead": {
        "description": "Compares with previous/next rows",
        "name_hints": ["prev", "previous", "next", "change", "delta", "diff"],
        "column_hints": ["prev_", "previous_", "next_", "lag_", "lead_", "change_", "diff_"],
        "function_hints": ["LAG", "LEAD"],
    },
    
    "percentile": {
        "description": "Calculates percentiles/quantiles",
        "name_hints": ["percentile", "quantile", "median", "quartile"],
        "column_hints": ["p50", "p95", "p99", "median", "percentile"],
        "function_hints": ["PERCENTILE_CONT", "APPROX_PERCENTILE", "NTILE"],
    },
    
    # =========================================================================
    # FILTERING PATTERNS
    # =========================================================================
    "spam_filter": {
        "description": "Filters out spam/bot activity",
        "name_hints": ["spam", "filter", "clean", "valid", "legitimate", "real", "human"],
        "column_hints": ["is_spam", "is_bot", "is_valid"],
        "pattern_hints": ["NOT IN", "WHERE.*spam", "WHERE.*bot"],
        "table_hints": ["labels", "spam"],
    },
    
    "threshold_filter": {
        "description": "Filters by value threshold",
        "name_hints": ["filter", "minimum", "threshold", "significant", "material"],
        "pattern_hints": ["HAVING.*>", "WHERE.*>=", "WHERE.*amount.*>"],
    },
    
    "time_filter": {
        "description": "Filters by time range",
        "name_hints": ["recent", "period", "range", "window"],
        "pattern_hints": ["block_time.*>=", "block_time.*BETWEEN", "NOW().*-.*INTERVAL"],
    },
    
    "whitelist": {
        "description": "Filters to specific allowed values",
        "name_hints": ["whitelist", "allowed", "include", "target"],
        "pattern_hints": ["IN \\(", "= ANY"],
    },
    
    "blacklist": {
        "description": "Excludes specific values",
        "name_hints": ["blacklist", "exclude", "remove", "ignore"],
        "pattern_hints": ["NOT IN", "!= ANY", "<> ALL"],
    },
    
    # =========================================================================
    # ENRICHMENT PATTERNS
    # =========================================================================
    "price_enrichment": {
        "description": "Joins price data for USD valuation",
        "name_hints": ["price", "prices", "usd", "valued", "enriched"],
        "column_hints": ["price", "price_usd", "usd_amount", "usd_value", "amount_usd"],
        "table_hints": ["prices.usd", "prices", "token_prices"],
        "operation_hints": {"joins": True},
    },
    
    "label_enrichment": {
        "description": "Joins label/identity data",
        "name_hints": ["label", "labels", "identity", "name", "tagged"],
        "column_hints": ["label", "name", "entity_name", "protocol"],
        "table_hints": ["labels", "contracts", "entities", "protocols"],
        "operation_hints": {"joins": True},
    },
    
    "token_metadata": {
        "description": "Joins token metadata (symbol, decimals)",
        "name_hints": ["token", "metadata", "info", "details"],
        "column_hints": ["symbol", "decimals", "token_name"],
        "table_hints": ["tokens", "erc20_tokens", "token_info"],
    },
    
    "ens_resolution": {
        "description": "Resolves ENS names",
        "name_hints": ["ens", "name", "resolved"],
        "column_hints": ["ens_name", "name", "domain"],
        "table_hints": ["ens", "domains"],
    },
    
    # =========================================================================
    # CROSS-CHAIN PATTERNS
    # =========================================================================
    "cross_chain_union": {
        "description": "Combines data from multiple chains",
        "name_hints": ["all_chains", "multi_chain", "combined", "unified", "merged"],
        "column_hints": ["chain", "blockchain", "network"],
        "operation_hints": {"unions": True},
        "pattern_hints": ["UNION ALL"],
    },
    
    "chain_comparison": {
        "description": "Compares metrics across chains",
        "name_hints": ["compare", "comparison", "by_chain", "per_chain"],
        "column_hints": ["ethereum", "solana", "bnb", "polygon", "chain"],
    },
    
    # =========================================================================
    # DEX/TRADING PATTERNS
    # =========================================================================
    "swap_extraction": {
        "description": "Extracts DEX swap events",
        "name_hints": ["swap", "swaps", "trade", "trades", "exchange"],
        "table_hints": ["dex.trades", "trades", "swaps", "uniswap", "sushiswap"],
        "column_hints": ["amount_in", "amount_out", "token_in", "token_out"],
    },
    
    "volume_calculation": {
        "description": "Calculates trading volume",
        "name_hints": ["volume", "vol", "trading_volume"],
        "column_hints": ["volume", "volume_usd", "trading_volume", "amount_usd"],
        "function_hints": ["SUM"],
    },
    
    "liquidity_tracking": {
        "description": "Tracks liquidity adds/removes",
        "name_hints": ["liquidity", "lp", "pool", "mint", "burn"],
        "column_hints": ["liquidity", "lp_amount", "pool_share"],
        "table_hints": ["pool", "liquidity", "mint", "burn"],
    },
    
    "price_impact": {
        "description": "Calculates price impact of trades",
        "name_hints": ["impact", "slippage", "price_change"],
        "column_hints": ["price_impact", "slippage", "execution_price"],
    },
    
    # =========================================================================
    # NFT PATTERNS
    # =========================================================================
    "nft_sales": {
        "description": "Extracts NFT sale events",
        "name_hints": ["sale", "sales", "sold", "purchase", "bought"],
        "table_hints": ["nft.trades", "seaport", "opensea", "blur"],
        "column_hints": ["price", "seller", "buyer", "token_id"],
    },
    
    "nft_transfers": {
        "description": "Tracks NFT transfers/mints",
        "name_hints": ["nft_transfer", "mint", "transfer"],
        "table_hints": ["erc721", "erc1155"],
        "column_hints": ["token_id", "from", "to"],
    },
    
    "floor_price": {
        "description": "Calculates collection floor price",
        "name_hints": ["floor", "minimum", "cheapest"],
        "column_hints": ["floor_price", "min_price"],
        "function_hints": ["MIN"],
    },
    
    # =========================================================================
    # LENDING/DEFI PATTERNS
    # =========================================================================
    "lending_positions": {
        "description": "Tracks lending/borrowing positions",
        "name_hints": ["position", "borrow", "supply", "deposit", "lending"],
        "table_hints": ["aave", "compound", "morpho"],
        "column_hints": ["supplied", "borrowed", "collateral", "debt"],
    },
    
    "liquidation": {
        "description": "Tracks liquidation events",
        "name_hints": ["liquidation", "liquidate", "liquidated"],
        "column_hints": ["liquidated_amount", "debt_repaid", "collateral_seized"],
    },
    
    "health_factor": {
        "description": "Calculates position health",
        "name_hints": ["health", "risk", "ltv", "collateral_ratio"],
        "column_hints": ["health_factor", "ltv", "collateral_ratio"],
    },
    
    # =========================================================================
    # SOCIAL/ENGAGEMENT PATTERNS
    # =========================================================================
    "engagement_metrics": {
        "description": "Calculates engagement (likes, comments, shares)",
        "name_hints": ["engagement", "interaction", "activity", "social"],
        "column_hints": ["likes", "comments", "shares", "reactions", "replies"],
        "table_hints": ["farcaster", "lens", "neynar"],
    },
    
    "follower_graph": {
        "description": "Analyzes follower relationships",
        "name_hints": ["follower", "following", "graph", "social_graph"],
        "column_hints": ["follower", "following", "followers_count"],
    },
    
    # =========================================================================
    # UTILITY PATTERNS
    # =========================================================================
    "normalization": {
        "description": "Normalizes data (decimals, case, etc)",
        "name_hints": ["normal", "format", "clean", "standardize"],
        "column_hints": ["normalized", "adjusted", "scaled"],
        "pattern_hints": ["/ 1e18", "/ POWER(10", "LOWER(", "UPPER("],
    },
    
    "null_handling": {
        "description": "Handles null values",
        "name_hints": ["coalesce", "fill", "default"],
        "function_hints": ["COALESCE", "NULLIF", "IFNULL", "NVL"],
    },
    
    "type_casting": {
        "description": "Casts data types",
        "name_hints": ["cast", "convert"],
        "function_hints": ["CAST", "TRY_CAST", "CONVERT"],
        "pattern_hints": ["::"],
    },
    
    "array_operations": {
        "description": "Works with arrays",
        "name_hints": ["array", "list", "unnest", "explode"],
        "function_hints": ["UNNEST", "ARRAY_AGG", "CARDINALITY", "ARRAY_JOIN"],
    },
}


# =============================================================================
# Enhanced Detection Functions
# =============================================================================

def detect_cte_intent(cte: Dict[str, Any]) -> List[str]:
    """
    Detect intent signals for a single CTE using multi-signal matching.
    
    Returns patterns with confidence > 0.3
    
    Args:
        cte: CTE dictionary with name, tables, columns, operations, comments
        
    Returns:
        List of detected pattern names sorted by confidence
    """
    matches = []
    
    # Extract all signals from CTE
    cte_name = cte.get("name", "").lower()
    tables = [t.lower() for t in cte.get("tables", [])]
    cte_refs = [r.lower() for r in cte.get("cte_refs", [])]
    
    columns = cte.get("columns", {})
    output_cols = [c.lower() for c in columns.get("output", [])]
    input_cols = [c.lower() for c in columns.get("input", [])]
    all_cols = output_cols + input_cols
    
    operations = cte.get("operations", {})
    aggregations = [a.upper() for a in operations.get("aggregations", [])]
    window_funcs = [w.upper() for w in operations.get("window_functions", [])]
    all_funcs = aggregations + window_funcs
    
    has_group_by = bool(operations.get("group_by"))
    has_distinct = operations.get("distinct", False)
    has_unions = operations.get("unions", False)
    has_joins = bool(operations.get("joins"))
    
    comment = cte.get("comments", {})
    if isinstance(comment, dict):
        comment_text = (comment.get("before") or "") + " " + " ".join(comment.get("inline", []))
    else:
        comment_text = str(comment) if comment else ""
    comment_text = comment_text.lower()
    
    # Check each pattern
    for pattern_name, pattern_def in CTE_PATTERNS.items():
        match = PatternMatch(pattern=pattern_name, confidence=0.0)
        
        # Check name hints (high weight)
        name_hints = pattern_def.get("name_hints", [])
        for hint in name_hints:
            if hint in cte_name:
                match.add_signal(0.4, f"name contains '{hint}'")
                break  # Only count once per category
        
        # Check column hints
        column_hints = pattern_def.get("column_hints", [])
        col_matches = 0
        for hint in column_hints:
            if any(hint in col for col in all_cols):
                col_matches += 1
        if col_matches > 0:
            weight = min(0.3, col_matches * 0.1)
            match.add_signal(weight, f"{col_matches} column hint matches")
        
        # Check function hints
        function_hints = pattern_def.get("function_hints", [])
        func_matches = 0
        for hint in function_hints:
            if any(hint.upper() in f for f in all_funcs):
                func_matches += 1
        if func_matches > 0:
            weight = min(0.3, func_matches * 0.15)
            match.add_signal(weight, f"{func_matches} function hint matches")
        
        # Check table hints
        table_hints = pattern_def.get("table_hints", [])
        for hint in table_hints:
            if any(hint.lower() in t for t in tables):
                match.add_signal(0.25, f"table matches '{hint}'")
                break
        
        # Check operation hints
        op_hints = pattern_def.get("operation_hints", {})
        if op_hints.get("distinct") and has_distinct:
            match.add_signal(0.2, "uses DISTINCT")
        if op_hints.get("group_by") and has_group_by:
            match.add_signal(0.15, "uses GROUP BY")
        if op_hints.get("aggregations") and aggregations:
            match.add_signal(0.15, "uses aggregations")
        if op_hints.get("unions") and has_unions:
            match.add_signal(0.3, "uses UNION")
        if op_hints.get("joins") and has_joins:
            match.add_signal(0.1, "uses JOIN")
        
        # Check anti-hints (reduce confidence)
        anti_hints = pattern_def.get("anti_hints", {})
        if anti_hints.get("aggregations") and aggregations:
            match.add_signal(-0.2, "has aggregations (unexpected)")
        
        # Check comment for hints
        all_hints = name_hints + column_hints + function_hints
        for hint in all_hints[:5]:  # Check first 5
            if hint.lower() in comment_text:
                match.add_signal(0.2, f"comment mentions '{hint}'")
                break
        
        # Only include if confidence is meaningful
        if match.confidence >= 0.3:
            matches.append(match)
    
    # Sort by confidence and return pattern names
    matches.sort(key=lambda m: m.confidence, reverse=True)
    
    # Return top matches (max 5, confidence > 0.3)
    return [m.pattern for m in matches[:5] if m.confidence >= 0.3]


def detect_cte_intent_detailed(cte: Dict[str, Any]) -> List[PatternMatch]:
    """
    Like detect_cte_intent but returns full PatternMatch objects
    with confidence scores and reasoning.
    
    Useful for debugging and understanding classifications.
    """
    matches = []
    
    # [Same logic as detect_cte_intent, but return full matches]
    cte_name = cte.get("name", "").lower()
    tables = [t.lower() for t in cte.get("tables", [])]
    
    columns = cte.get("columns", {})
    output_cols = [c.lower() for c in columns.get("output", [])]
    input_cols = [c.lower() for c in columns.get("input", [])]
    all_cols = output_cols + input_cols
    
    operations = cte.get("operations", {})
    aggregations = [a.upper() for a in operations.get("aggregations", [])]
    window_funcs = [w.upper() for w in operations.get("window_functions", [])]
    all_funcs = aggregations + window_funcs
    
    has_group_by = bool(operations.get("group_by"))
    has_distinct = operations.get("distinct", False)
    has_unions = operations.get("unions", False)
    has_joins = bool(operations.get("joins"))
    
    for pattern_name, pattern_def in CTE_PATTERNS.items():
        match = PatternMatch(pattern=pattern_name, confidence=0.0)
        
        # Name hints
        for hint in pattern_def.get("name_hints", []):
            if hint in cte_name:
                match.add_signal(0.4, f"name contains '{hint}'")
                break
        
        # Column hints
        col_matches = sum(1 for hint in pattern_def.get("column_hints", []) 
                        if any(hint in col for col in all_cols))
        if col_matches:
            match.add_signal(min(0.3, col_matches * 0.1), f"{col_matches} column matches")
        
        # Function hints
        func_matches = sum(1 for hint in pattern_def.get("function_hints", [])
                         if any(hint.upper() in f for f in all_funcs))
        if func_matches:
            match.add_signal(min(0.3, func_matches * 0.15), f"{func_matches} function matches")
        
        # Table hints
        for hint in pattern_def.get("table_hints", []):
            if any(hint.lower() in t for t in tables):
                match.add_signal(0.25, f"table matches '{hint}'")
                break
        
        # Operation hints
        op_hints = pattern_def.get("operation_hints", {})
        if op_hints.get("distinct") and has_distinct:
            match.add_signal(0.2, "uses DISTINCT")
        if op_hints.get("group_by") and has_group_by:
            match.add_signal(0.15, "uses GROUP BY")
        if op_hints.get("aggregations") and aggregations:
            match.add_signal(0.15, "uses aggregations")
        if op_hints.get("unions") and has_unions:
            match.add_signal(0.3, "uses UNION")
        if op_hints.get("joins") and has_joins:
            match.add_signal(0.1, "uses JOIN")
        
        if match.confidence >= 0.25:
            matches.append(match)
    
    matches.sort(key=lambda m: m.confidence, reverse=True)
    return matches[:10]


def detect_intent_patterns(features: Dict[str, Any]) -> List[str]:
    """
    Detect high-level intent patterns for entire query.
    
    Combines signals from:
    - All CTEs
    - Final SELECT
    - Query metadata
    - Overall structure
    """
    patterns = set()
    
    # Collect from CTEs
    for cte in features.get("ctes", []):
        cte_patterns = detect_cte_intent(cte)
        patterns.update(cte_patterns)
    
    # Check query-level patterns
    tables = features.get("tables", [])
    table_names = [t.get("full_name", "").lower() for t in tables]
    
    # Cross-chain detection
    chains = ["ethereum", "solana", "bnb", "polygon", "arbitrum", "optimism", "base", "avalanche"]
    chains_found = set()
    for table in table_names:
        for chain in chains:
            if chain in table:
                chains_found.add(chain)
    if len(chains_found) > 1:
        patterns.add("cross_chain_union")
    
    # Complexity patterns
    complexity = features.get("complexity", {})
    if complexity.get("cte_count", 0) > 5:
        patterns.add("complex_pipeline")
    if complexity.get("union_count", 0) > 0:
        patterns.add("data_unification")
    if complexity.get("join_count", 0) > 3:
        patterns.add("multi_source_join")
    
    # Check final select
    final_select = features.get("final_select", {})
    final_ops = final_select.get("operations", {})
    
    if final_ops.get("aggregations"):
        patterns.add("aggregation")
    if final_ops.get("window_functions"):
        patterns.add("window_analysis")
    
    # Limit check
    if final_ops.get("limit"):
        limit = final_ops.get("limit")
        if isinstance(limit, int) and limit <= 100:
            patterns.add("top_n_query")
    
    return sorted(list(patterns))


def detect_domain_hints(features: Dict[str, Any]) -> List[str]:
    """
    Detect domain/vertical hints from tables and columns.
    """
    hints = set()
    
    # Collect all table names
    tables = features.get("tables", [])
    table_names = " ".join(t.get("full_name", "") for t in tables).lower()
    
    # Add CTE tables
    for cte in features.get("ctes", []):
        table_names += " " + " ".join(cte.get("tables", []))
    
    # Domain patterns
    domain_patterns = {
        "dex": ["dex", "trades", "swap", "uniswap", "sushiswap", "curve", "balancer", "raydium", "jupiter"],
        "lending": ["aave", "compound", "morpho", "spark", "borrow", "supply", "liquidat"],
        "nft": ["nft", "opensea", "blur", "seaport", "erc721", "erc1155", "magic_eden"],
        "token": ["erc20", "evt_transfer", "token"],
        "bridge": ["bridge", "wormhole", "layerzero", "stargate", "hop", "across"],
        "staking": ["staking", "validator", "delegation", "lido", "rocket_pool"],
        "social": ["farcaster", "lens", "neynar", "cast", "profile"],
        "governance": ["governance", "vote", "proposal", "snapshot", "dao"],
    }
    
    for domain, keywords in domain_patterns.items():
        if any(kw in table_names for kw in keywords):
            hints.add(domain)
    
    # Chain detection
    chain_patterns = {
        "ethereum": ["ethereum", "_ethereum"],
        "solana": ["solana", "_solana"],
        "bnb": ["bnb", "bsc", "_bnb"],
        "polygon": ["polygon", "matic", "_polygon"],
        "arbitrum": ["arbitrum", "_arbitrum"],
        "optimism": ["optimism", "_optimism"],
        "base": ["base", "_base"],
    }
    
    for chain, keywords in chain_patterns.items():
        if any(kw in table_names for kw in keywords):
            hints.add(chain)
    
    # Tags
    tags = features.get("metadata", {}).get("tags", [])
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in ["defi", "nft", "dex", "lending", "bridge", "gaming"]:
            hints.add(tag_lower)
    
    return sorted(list(hints))


def detect_output_type(features: Dict[str, Any]) -> str:
    """
    Determine what type of output the query produces.
    """
    final_select = features.get("final_select", {})
    columns = final_select.get("columns", {})
    output_cols = [c.lower() for c in columns.get("output", [])]
    operations = final_select.get("operations", {})
    
    # Time columns
    time_cols = ["day", "date", "dt", "hour", "week", "month", "period", "timestamp", "block_time"]
    has_time = any(any(tc in col for tc in time_cols) for col in output_cols)
    
    # Ranking
    has_rank = any("rank" in col or "rn" in col for col in output_cols)
    
    # Events
    event_cols = ["tx_hash", "transaction_hash", "evt", "event", "log_index"]
    has_events = any(any(ec in col for ec in event_cols) for col in output_cols)
    
    # Entity columns
    entity_cols = ["address", "wallet", "account", "user", "holder", "trader"]
    has_entity = any(any(ec in col for ec in entity_cols) for col in output_cols)
    
    # Aggregations
    has_agg = bool(operations.get("aggregations"))
    has_group = bool(operations.get("group_by"))
    
    # Determine type
    if has_rank:
        return "ranking"
    if operations.get("limit") == 1 and has_agg:
        return "single_value"
    if has_time and has_group:
        return "time_series"
    if has_events:
        return "event_log"
    if has_entity and not has_time and not has_agg:
        return "entity_list"
    if has_agg and not has_group:
        return "summary"
    if has_time:
        return "time_series"
    
    return "table"


# =============================================================================
# Utility Functions
# =============================================================================

def explain_pattern(pattern_name: str) -> str:
    """Get description of a pattern."""
    pattern = CTE_PATTERNS.get(pattern_name, {})
    return pattern.get("description", f"Unknown pattern: {pattern_name}")


def get_pattern_hints(pattern_name: str) -> Dict[str, List[str]]:
    """Get all hints for a pattern."""
    pattern = CTE_PATTERNS.get(pattern_name, {})
    return {
        "name_hints": pattern.get("name_hints", []),
        "column_hints": pattern.get("column_hints", []),
        "function_hints": pattern.get("function_hints", []),
        "table_hints": pattern.get("table_hints", []),
    }


def list_all_patterns() -> List[str]:
    """Get list of all defined patterns."""
    return sorted(CTE_PATTERNS.keys())