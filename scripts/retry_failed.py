"""
Retry script for fetching missing query IDs.
Identifies gaps in collected data and retries them efficiently.
"""
from pathlib import Path

# Import from shared utilities and collector
from src.utils import get_saved_ids, find_missing_ids, group_consecutive_ids
from src.collector import DuneCollector
from src.config import Config

DATA_DIR = Path("data")


def main():
    """Main execution: find and retry missing queries."""
    start_id, end_id = 1, 200000
    total_ids = end_id - start_id + 1

    print("[INFO] Scanning for missing queries...")
    
    # Get saved and missing IDs using shared utilities
    saved_ids = get_saved_ids(DATA_DIR)
    missing_ids = find_missing_ids(start_id, end_id, saved_ids)

    saved_count = len(saved_ids)
    missing_count = len(missing_ids)
    percentage_saved = (saved_count / total_ids) * 100

    print(f"\n[INFO] === Collection Status ===")
    print(f"[INFO] Total IDs: {total_ids:,}")
    print(f"[INFO] Saved IDs: {saved_count:,}")
    print(f"[INFO] Missing IDs: {missing_count:,}")
    print(f"[INFO] Coverage: {percentage_saved:.2f}%")

    if not missing_ids:
        print("\n[INFO] ✅ No missing IDs! Collection is complete.")
        return

    # Group consecutive IDs into ranges for efficient collection
    print(f"\n[INFO] Grouping missing IDs into consecutive ranges...")
    ranges = group_consecutive_ids(missing_ids)
    
    print(f"[INFO] Found {len(ranges)} ranges to retry")
    
    # Show range summary
    total_to_retry = sum(end - start + 1 for start, end in ranges)
    print(f"[INFO] Total queries to retry: {total_to_retry:,}")
    
    # Show largest gaps
    range_sizes = [(end - start + 1, start, end) for start, end in ranges]
    range_sizes.sort(reverse=True)
    
    print(f"\n[INFO] Largest gaps:")
    for size, start, end in range_sizes[:5]:
        print(f"  - IDs {start:,} to {end:,} ({size:,} queries)")

    # Confirm before proceeding
    print(f"\n[INFO] 🔄 Starting retry process...")
    config = Config()
    collector = DuneCollector(
        config=config,
        max_workers=20,
        delay=0.1,
        retry_config={"max_retries": 100, "backoff_factor": 2.0, "retry_on_statuses": (429, 500, 502, 503, 504)}
    )
    
    # Retry each range
    for i, (range_start, range_end) in enumerate(ranges, 1):
        range_size = range_end - range_start + 1
        print(f"\n[INFO] === Range {i}/{len(ranges)} ===")
        print(f"[INFO] IDs: {range_start:,} to {range_end:,} ({range_size:,} queries)")
        
        collector.collect_queries(start_id=range_start, end_id=range_end, batch_size=2_000)
    
    print("\n✅ Retry complete! Re-run this script to check for any remaining gaps.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user. Progress has been saved.")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        raise