import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from itertools import cycle
from src.collect import fetch_dune_query, save_parquet

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "retried"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

lock = threading.Lock()
key_lock = threading.Lock()
request_times = []

def get_saved_ids(data_dir: Path):
    saved_ids = set()
    for parquet_file in data_dir.glob("*.parquet"):
        table = pq.read_table(parquet_file)
        if "query_id" in table.schema.names:
            saved_ids.update(table.column("query_id").to_pylist())
    return saved_ids

def find_missing_ids(start_id, end_id, saved_ids):
    expected_ids = set(range(start_id, end_id + 1))
    missing_ids = sorted(expected_ids - saved_ids)
    print(f"[INFO] Found {len(missing_ids)} missing IDs in range {start_id}-{end_id}")
    return missing_ids

if __name__ == "__main__":
    start_id, end_id = 1, 70000
    total_ids = end_id - start_id + 1

    saved_ids = get_saved_ids(DATA_DIR)
    missing_ids = find_missing_ids(start_id, end_id, saved_ids)

    saved_count = len(saved_ids)
    missing_count = len(missing_ids)
    percentage_saved = (saved_count / total_ids) * 100

    print(f"[INFO] Total IDs: {total_ids}")
    print(f"[INFO] Saved IDs: {saved_count}")
    print(f"[INFO] Missing IDs: {missing_count}")
    print(f"[INFO] Percentage of queries stored: {percentage_saved:.2f}%")

    if not missing_ids:
        print("[INFO] No missing IDs. Exiting.")
        exit(0)