"""
Duplicate detection and visualization for collected query data.
Finds exact and near-duplicate queries using SQL normalization and fingerprinting.
"""
import re
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass

# Import shared utilities
from src.utils import load_queries, iter_queries

DATA_DIR = Path("data")
OUTPUT_DIR = Path("analysis") / "duplicates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DuplicateGroup:
    """Group of duplicate/similar queries."""
    canonical_id: int
    member_ids: List[int]
    similarity: float
    sql_preview: str


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison by removing noise.
    
    Args:
        sql: Raw SQL string
        
    Returns:
        Normalized SQL string
    """
    if not sql:
        return ""
    
    # Remove comments
    sql = re.sub(r'--[^\n]*', '', sql)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    
    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql).strip().lower()
    
    # Normalize literals
    sql = re.sub(r"'[^']*'", "'?'", sql)
    sql = re.sub(r'\b\d+\.?\d*\b', '?', sql)
    sql = re.sub(r'0x[a-f0-9]+', '0x?', sql)
    sql = re.sub(r'\{\{[^}]+\}\}', '{{?}}', sql)
    
    return sql


def compute_sql_hash(sql: str) -> str:
    """Compute hash of normalized SQL."""
    normalized = normalize_sql(sql)
    return hashlib.md5(normalized.encode()).hexdigest()


def find_exact_duplicates(queries: List[Dict]) -> List[DuplicateGroup]:
    """
    Find exact duplicate queries based on normalized SQL hash.
    
    Args:
        queries: List of query dictionaries
        
    Returns:
        List of DuplicateGroup objects
    """
    hash_groups: Dict[str, List[int]] = defaultdict(list)
    hash_to_sql: Dict[str, str] = {}
    
    for q in queries:
        qid = q.get("query_id")
        sql = q.get("query_sql", "")
        
        if not sql:
            continue
        
        sql_hash = compute_sql_hash(sql)
        hash_groups[sql_hash].append(qid)
        
        if sql_hash not in hash_to_sql:
            hash_to_sql[sql_hash] = normalize_sql(sql)[:200]
    
    groups = []
    for sql_hash, ids in hash_groups.items():
        if len(ids) > 1:
            groups.append(DuplicateGroup(
                canonical_id=ids[0],
                member_ids=ids,
                similarity=1.0,
                sql_preview=hash_to_sql.get(sql_hash, "")
            ))
    
    return groups


def plot_duplicate_distribution(groups: List[DuplicateGroup], save_path: Path = None):
    """
    Create histogram of duplicate group sizes.
    
    Args:
        groups: List of DuplicateGroup objects
        save_path: Path to save the plot
    """
    sizes = [len(g.member_ids) for g in groups]
    
    plt.figure(figsize=(12, 6))
    plt.hist(sizes, bins=50, edgecolor='black', alpha=0.7, color='coral')
    plt.title("Distribution of Duplicate Group Sizes", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Duplicates in Group", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved distribution plot → {save_path}")
    
    plt.close()


def plot_duplicate_pie(total: int, unique: int, save_path: Path = None):
    """
    Create pie chart showing unique vs duplicate queries.
    
    Args:
        total: Total number of queries
        unique: Number of unique queries
        save_path: Path to save the plot
    """
    duplicates = total - unique
    
    plt.figure(figsize=(8, 8))
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)
    
    plt.pie(
        [unique, duplicates],
        labels=[f"Unique\n{unique:,}", f"Duplicates\n{duplicates:,}"],
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'},
        shadow=True
    )
    plt.title("Query Uniqueness", fontsize=16, fontweight='bold', pad=20)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved pie chart → {save_path}")
    
    plt.close()


def plot_top_duplicates(groups: List[DuplicateGroup], top_n: int = 20, save_path: Path = None):
    """
    Create bar chart of top duplicate groups.
    
    Args:
        groups: List of DuplicateGroup objects
        top_n: Number of top groups to show
        save_path: Path to save the plot
    """
    sorted_groups = sorted(groups, key=lambda g: len(g.member_ids), reverse=True)[:top_n]
    
    labels = [f"ID {g.canonical_id}" for g in sorted_groups]
    sizes = [len(g.member_ids) for g in sorted_groups]
    
    plt.figure(figsize=(14, 6))
    bars = plt.barh(labels, sizes, color='steelblue', edgecolor='black')
    plt.xlabel("Number of Duplicates", fontsize=12)
    plt.ylabel("Canonical Query ID", fontsize=12)
    plt.title(f"Top {top_n} Most Duplicated Queries", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved top duplicates chart → {save_path}")
    
    plt.close()


def analyze_duplicates(queries: List[Dict]) -> Dict:
    """
    Analyze duplicate queries and generate visualizations.
    
    Args:
        queries: List of query dictionaries
        
    Returns:
        Dictionary with duplicate statistics
    """
    print(f"\n[INFO] Analyzing {len(queries):,} queries for duplicates...")
    
    groups = find_exact_duplicates(queries)
    
    # Calculate statistics
    total = len(queries)
    total_in_groups = sum(len(g.member_ids) for g in groups)
    unique_after_dedup = total - total_in_groups + len(groups)
    duplicates_removed = total_in_groups - len(groups)
    
    stats = {
        "total_queries": total,
        "duplicate_groups": len(groups),
        "total_duplicates": duplicates_removed,
        "unique_queries": unique_after_dedup,
        "duplicate_rate": duplicates_removed / total * 100 if total > 0 else 0,
        "largest_group_size": max(len(g.member_ids) for g in groups) if groups else 0,
        "average_group_size": np.mean([len(g.member_ids) for g in groups]) if groups else 0,
    }
    
    # Print statistics
    print(f"\n[INFO] === Duplicate Analysis Statistics ===")
    print(f"[INFO] Total queries: {stats['total_queries']:,}")
    print(f"[INFO] Duplicate groups: {stats['duplicate_groups']:,}")
    print(f"[INFO] Total duplicates: {stats['total_duplicates']:,}")
    print(f"[INFO] Unique queries: {stats['unique_queries']:,}")
    print(f"[INFO] Duplicate rate: {stats['duplicate_rate']:.2f}%")
    print(f"[INFO] Largest group: {stats['largest_group_size']:,}")
    print(f"[INFO] Average group size: {stats['average_group_size']:.1f}")
    
    # Generate visualizations
    if groups:
        print(f"\n[INFO] Generating visualizations...")
        plot_duplicate_distribution(groups, save_path=OUTPUT_DIR / "duplicate_distribution.png")
        plot_duplicate_pie(total, unique_after_dedup, save_path=OUTPUT_DIR / "duplicate_pie.png")
        plot_top_duplicates(groups, top_n=20, save_path=OUTPUT_DIR / "top_duplicates.png")
    
    # Save groups to JSON
    groups_data = [
        {
            "canonical_id": g.canonical_id,
            "member_ids": g.member_ids,
            "count": len(g.member_ids),
            "sql_preview": g.sql_preview
        }
        for g in sorted(groups, key=lambda g: len(g.member_ids), reverse=True)
    ]
    
    output_file = OUTPUT_DIR / "duplicate_groups.json"
    with open(output_file, 'w') as f:
        json.dump(groups_data, f, indent=2)
    print(f"[INFO] Saved duplicate groups → {output_file}")
    
    return stats


def main():
    """Main execution: analyze duplicates in collected queries."""
    print("[INFO] === Dune Query Duplicate Analysis ===\n")
    
    # Load queries
    queries = list(iter_queries(DATA_DIR))
    
    if not queries:
        print("[ERROR] No queries found in data directory")
        return
    
    # Analyze
    stats = analyze_duplicates(queries)
    
    print(f"\n✅ Duplicate analysis complete! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        raise