import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from itertools import cycle
from src.collect import fetch_dune_query, save_parquet
from src.calculated_failed import find_missing_ids, get_saved_ids

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "retried"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

lock = threading.Lock()
key_lock = threading.Lock()
request_times = []


if __name__ == "__main__":
    start_id, end_id = 1, 70000
    total_ids = end_id - start_id + 1

    saved_ids = get_saved_ids(DATA_DIR)
    missing_ids = find_missing_ids(start_id, end_id, saved_ids)
    print(f"[INFO] Found {missing_ids}")

    # saved_count = len(saved_ids)
    # missing_count = len(missing_ids)
    # percentage_saved = (saved_count / total_ids) * 100

    # print(f"[INFO] Total IDs: {total_ids}")
    # print(f"[INFO] Saved IDs: {saved_count}")
    # print(f"[INFO] Missing IDs: {missing_count}")
    # print(f"[INFO] Percentage of queries stored: {percentage_saved:.2f}%")

    # if not missing_ids:
    #     print("[INFO] No missing IDs. Exiting.")
    #     exit(0)