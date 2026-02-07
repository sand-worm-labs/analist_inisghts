"""
Collection pipeline for fetching Dune Analytics queries.

Usage:
    python -m src.pipeline.collect --start 1 --end 10000
    python -m src.pipeline.collect --retry
"""

import argparse
from pathlib import Path

from src.config import Config
from src.collector import DuneCollector
from src.utils.io import get_saved_ids, find_missing_ids, group_consecutive_ids


DEFAULT_DATA_DIR = Path("dataset")


def run_collection(
    start_id: int,
    end_id: int,
    batch_size: int = 10_000,
    max_workers: int = 20,
    delay: float = 0.1,
) -> None:
    """
    Run the query collection pipeline.
    
    Args:
        start_id: First query ID to fetch
        end_id: Last query ID to fetch
        batch_size: Number of queries per batch
        max_workers: Concurrent worker threads
        delay: Delay between requests (seconds)
    """
    print(f"\n{'='*60}")
    print("DUNE QUERY COLLECTION PIPELINE")
    print(f"{'='*60}\n")
    
    print(f"[INFO] Range: {start_id:,} to {end_id:,}")
    print(f"[INFO] Batch size: {batch_size:,}")
    print(f"[INFO] Workers: {max_workers}")
    
    config = Config()
    collector = DuneCollector(
        config=config,
        max_workers=max_workers,
        delay=delay,
        retry_config={
            "max_retries": 100,
            "backoff_factor": 2.0,
            "retry_on_statuses": (429, 500, 502, 503, 504)
        }
    )
    
    collector.collect_queries(
        start_id=start_id,
        end_id=end_id,
        batch_size=batch_size
    )
    
    print("\n✅ Collection complete!")


def run_retry(
    start_id: int = 1,
    end_id: int = 200_000,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    """
    Retry fetching missing queries.
    
    Args:
        start_id: Start of ID range to check
        end_id: End of ID range to check
        data_dir: Directory containing existing parquet files
    """
    print(f"\n{'='*60}")
    print("DUNE QUERY RETRY PIPELINE")
    print(f"{'='*60}\n")
    
    # Find missing IDs
    saved_ids = get_saved_ids(data_dir)
    missing_ids = find_missing_ids(start_id, end_id, saved_ids)
    
    total_ids = end_id - start_id + 1
    saved_count = len(saved_ids)
    missing_count = len(missing_ids)
    
    print(f"[INFO] Total IDs in range: {total_ids:,}")
    print(f"[INFO] Already saved: {saved_count:,}")
    print(f"[INFO] Missing: {missing_count:,}")
    print(f"[INFO] Coverage: {saved_count/total_ids*100:.2f}%")
    
    if not missing_ids:
        print("\n✅ No missing IDs! Collection is complete.")
        return
    
    # Group consecutive IDs into ranges
    ranges = group_consecutive_ids(missing_ids)
    print(f"\n[INFO] Found {len(ranges)} ranges to retry")
    
    # Show largest gaps
    range_sizes = [(end - start + 1, start, end) for start, end in ranges]
    range_sizes.sort(reverse=True)
    
    print(f"\n[INFO] Largest gaps:")
    for size, start, end in range_sizes[:5]:
        print(f"  - IDs {start:,} to {end:,} ({size:,} queries)")
    
    # Retry each range
    config = Config()
    collector = DuneCollector(
        config=config,
        max_workers=20,
        delay=0.1,
        retry_config={
            "max_retries": 100,
            "backoff_factor": 2.0,
            "retry_on_statuses": (429, 500, 502, 503, 504)
        }
    )
    
    for i, (range_start, range_end) in enumerate(ranges, 1):
        range_size = range_end - range_start + 1
        print(f"\n[INFO] === Range {i}/{len(ranges)} ===")
        print(f"[INFO] IDs: {range_start:,} to {range_end:,} ({range_size:,} queries)")
        
        collector.collect_queries(
            start_id=range_start,
            end_id=range_end,
            batch_size=2_000
        )
    
    print("\n✅ Retry complete!")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Collect Dune Analytics queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Collect queries 1-10000
    python -m src.pipeline.collect --start 1 --end 10000
    
    # Retry missing queries
    python -m src.pipeline.collect --retry --start 1 --end 200000
    
    # Custom batch size and workers
    python -m src.pipeline.collect --start 1 --end 50000 --batch-size 5000 --workers 30
        """
    )
    
    parser.add_argument("--start", type=int, default=1, help="Start query ID")
    parser.add_argument("--end", type=int, required=True, help="End query ID")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Batch size")
    parser.add_argument("--workers", type=int, default=20, help="Worker threads")
    parser.add_argument("--delay", type=float, default=0.1, help="Request delay")
    parser.add_argument("--retry", action="store_true", help="Retry missing queries only")
    
    args = parser.parse_args()
    
    if args.retry:
        run_retry(start_id=args.start, end_id=args.end)
    else:
        run_collection(
            start_id=args.start,
            end_id=args.end,
            batch_size=args.batch_size,
            max_workers=args.workers,
            delay=args.delay,
        )


if __name__ == "__main__":
    main()