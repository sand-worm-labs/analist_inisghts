"""
Clustering pipeline for grouping similar Dune queries.

Supports two modes:
- semantic: Cluster by name, description, tags (WHAT it's about)
- sql: Cluster by query SQL patterns (HOW it works)

Usage:
    python -m src.pipeline.cluster --mode semantic --limit 10000
    python -m src.pipeline.cluster --mode sql --limit 50000
"""

import argparse
from pathlib import Path
from typing import Optional

from src.utils.io import get_query_objects
from src.cluster_queries import (
    QueryClusterer,
    save_clusters,
    print_cluster_summary,
)


DEFAULT_DATA_DIR = Path("dataset")
OUTPUT_DIR_SEMANTIC = Path("clusters/semantic")
OUTPUT_DIR_SQL = Path("clusters/sql")


# Model recommendations by mode
MODELS = {
    "semantic": "nvidia/NV-Embed-v2",
    "sql": "s2593817/sft-sql-embedding",
}


def run_clustering(
    mode: str = "semantic",
    limit: Optional[int] = None,
    model: Optional[str] = None,
    min_cluster_size: int = 15,
    min_samples: int = 3,
    workers: Optional[int] = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    """
    Run the query clustering pipeline.
    
    Args:
        mode: 'semantic' or 'sql'
        limit: Max queries to process (None for all)
        model: Model name (auto-selected if None)
        min_cluster_size: HDBSCAN min cluster size
        min_samples: HDBSCAN min samples
        workers: Worker threads (None for auto)
        data_dir: Directory containing query parquet files
    """
    print(f"\n{'='*60}")
    print(f"DUNE QUERY CLUSTERING - {mode.upper()} MODE")
    print(f"{'='*60}\n")
    
    # Select model
    model_name = model or MODELS.get(mode, MODELS["semantic"])
    print(f"[INFO] Model: {model_name}")
    
    # Select output directory
    output_dir = OUTPUT_DIR_SQL if mode == "sql" else OUTPUT_DIR_SEMANTIC
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output: {output_dir}")
    
    # Load queries
    print(f"\n[INFO] Loading queries from {data_dir}...")
    query_objects = get_query_objects(data_dir, limit=limit)
    
    if not query_objects:
        print("[ERROR] No queries found!")
        return
    
    print(f"[INFO] Loaded {len(query_objects):,} queries")
    
    # Initialize clusterer
    clusterer = QueryClusterer(
        model_name=model_name,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        mode=mode,
        max_workers=workers,
    )
    
    # Create embeddings
    batch_size = 64 if mode == "semantic" else 256
    clusterer.create_embeddings(query_objects, batch_size=batch_size)
    
    # Reduce dimensions
    clusterer.reduce_dimensions(n_components=5, n_neighbors=15)
    
    # Cluster
    cluster_labels = clusterer.cluster()
    
    # Print summary
    print_cluster_summary(query_objects, cluster_labels)
    
    # Save results
    save_clusters(query_objects, cluster_labels, output_dir, mode)
    clusterer.save_model(output_dir / "clusterer_model.pkl")
    
    print(f"\n✅ Clustering complete! Results saved to {output_dir}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cluster Dune queries by semantic similarity or SQL patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Cluster by meaning (name, description, tags)
    python -m src.pipeline.cluster --mode semantic --limit 10000
    
    # Cluster by SQL patterns
    python -m src.pipeline.cluster --mode sql --limit 50000
    
    # Full dataset with custom parameters
    python -m src.pipeline.cluster --mode sql --min-cluster-size 50 --workers 16
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["semantic", "sql"], 
        default="semantic",
        help="Clustering mode"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max queries to process")
    parser.add_argument("--model", type=str, default=None, help="Model name (auto if not set)")
    parser.add_argument("--min-cluster-size", type=int, default=15, help="HDBSCAN min cluster size")
    parser.add_argument("--min-samples", type=int, default=3, help="HDBSCAN min samples")
    parser.add_argument("--workers", type=int, default=None, help="Worker threads")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Data directory")
    
    args = parser.parse_args()
    
    run_clustering(
        mode=args.mode,
        limit=args.limit,
        model=args.model,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        workers=args.workers,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()