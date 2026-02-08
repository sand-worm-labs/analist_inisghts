
"""
Intent extraction pipeline for analyzing Dune query patterns.

Extracts:
- Tables and schemas used
- CTE names and structure
- Output columns
- SQL patterns (aggregations, joins, windows)
- Comments and annotations
- Builds signatures for embedding/clustering

Usage:
    python -m src.pipeline.extract --limit 1000
    python -m src.pipeline.extract --output ./output/intents
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.io import get_query_objects
from src.intent_extraction.features import extract_query_features
from src.intent_extraction.signatures import build_cte_signature, build_query_signature


DEFAULT_DATA_DIR = Path("dataset")
DEFAULT_OUTPUT_DIR = Path("output/intents")


def process_single_query(query: Dict) -> Optional[Dict]:
    """
    Process a single query and extract features.
    
    Args:
        query: Query dictionary with query_sql field
        
    Returns:
        Extracted features dictionary or None on error
    """
    try:
        sql = query.get("query_sql", "")
        if not sql or not sql.strip():
            return None
        
        # Extract features
        features = extract_query_features(query)
        
        # Build signatures
        features["query_signature"] = build_query_signature(features)
        
        # Add CTE signatures
        for cte in features.get("ctes", []):
            cte["signature"] = build_cte_signature(cte)
        
        return features
        
    except Exception as e:
        return {
            "query_id": query.get("query_id"),
            "error": str(e),
            "query_signature": "",
            "ctes": [],
        }


def run_extraction(
    limit: Optional[int] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
    workers: int = 4,
    batch_size: int = 10_000,
) -> None:
    """
    Run the intent extraction pipeline.
    
    Args:
        limit: Max queries to process (None for all)
        output_dir: Directory to save results
        data_dir: Directory containing query parquet files
        workers: Number of parallel workers
        batch_size: Queries per output file
    """
    print(f"\n{'='*60}")
    print("DUNE QUERY INTENT EXTRACTION PIPELINE")
    print(f"{'='*60}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load queries
    print(f"[INFO] Loading queries from {data_dir}...")
    query_objects = get_query_objects(data_dir, limit=limit)
    
    if not query_objects:
        print("[ERROR] No queries found!")
        return
    
    print(f"[INFO] Loaded {len(query_objects):,} queries")
    print(f"[INFO] Workers: {workers}")
    print(f"[INFO] Output: {output_dir}")
    
    # Process queries
    all_features: List[Dict] = []
    all_ctes: List[Dict] = []
    errors = 0
    
    print(f"\n[INFO] Extracting features...")
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_single_query, q): q["query_id"]
            for q in query_objects
        }
        
        with tqdm(total=len(futures), desc="Extracting") as pbar:
            for future in as_completed(futures):
                query_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        if "error" in result:
                            errors += 1
                        else:
                            all_features.append(result)
                            
                            # Extract CTEs separately
                            for cte in result.get("ctes", []):
                                cte_record = {
                                    "query_id": result["query_id"],
                                    "cte_name": cte["name"],
                                    "signature": cte.get("signature", ""),
                                    "tables": cte.get("tables", []),
                                    "output_columns": cte.get("columns", {}).get("output", []),
                                    "functions": list(set(
                                        cte.get("operations", {}).get("aggregations", []) +
                                        cte.get("operations", {}).get("window_functions", [])
                                    )),
                                    "comment_before": cte.get("comments", {}).get("before"),
                                }
                                all_ctes.append(cte_record)
                                
                except Exception as e:
                    errors += 1
                    
                pbar.update(1)
    
    # Statistics
    print(f"\n[INFO] === Extraction Statistics ===")
    print(f"[INFO] Queries processed: {len(all_features):,}")
    print(f"[INFO] CTEs extracted: {len(all_ctes):,}")
    print(f"[INFO] Errors: {errors:,}")
    
    if all_features:
        avg_ctes = len(all_ctes) / len(all_features)
        print(f"[INFO] Avg CTEs per query: {avg_ctes:.2f}")
    
    # Save query features
    print(f"\n[INFO] Saving results...")
    
    # Save as parquet
    features_file = output_dir / "query_features.parquet"
    features_df = pa.Table.from_pylist(all_features)
    pq.write_table(features_df, features_file, compression="zstd")
    print(f"[INFO] ✅ Saved query features to {features_file}")
    
    # Save CTEs separately
    if all_ctes:
        ctes_file = output_dir / "cte_features.parquet"
        ctes_df = pa.Table.from_pylist(all_ctes)
        pq.write_table(ctes_df, ctes_file, compression="zstd")
        print(f"[INFO] ✅ Saved CTE features to {ctes_file}")
    
    # Save signatures as text for inspection
    signatures_file = output_dir / "signatures.txt"
    with open(signatures_file, "w", encoding="utf-8") as f:
        for feat in all_features[:1000]:  # First 1000 for inspection
            f.write(f"{feat.get('query_id')}\t{feat.get('query_signature', '')}\n")
    print(f"[INFO] ✅ Saved sample signatures to {signatures_file}")
    
    # Save summary stats
    stats = {
        "total_queries": len(all_features),
        "total_ctes": len(all_ctes),
        "errors": errors,
        "avg_ctes_per_query": len(all_ctes) / len(all_features) if all_features else 0,
    }
    stats_file = output_dir / "extraction_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] ✅ Saved stats to {stats_file}")
    
    print(f"\n✅ Extraction complete! Results saved to {output_dir}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract intent features from Dune queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract from first 1000 queries
    python -m src.pipeline.extract --limit 1000
    
    # Full extraction with custom output
    python -m src.pipeline.extract --output ./intents --workers 8
    
    # Process specific data directory
    python -m src.pipeline.extract --data-dir ./my_queries --limit 5000
        """
    )
    
    parser.add_argument("--limit", type=int, default=None, help="Max queries to process")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Data directory")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Batch size for output")
    
    args = parser.parse_args()
    
    run_extraction(
        limit=args.limit,
        output_dir=args.output,
        data_dir=args.data_dir,
        workers=args.workers,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()