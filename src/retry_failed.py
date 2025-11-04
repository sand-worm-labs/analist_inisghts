import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from src.collect import fetch_dune_query, save_parquet  # Import your existing functions

# ---------------- Paths ---------------- #
DATA_DIR = Path("data")  # Folder containing your Parquet files
OUTPUT_DIR = DATA_DIR / "retried"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Globals ---------------- #
lock = threading.Lock()
request_times = []

# ---------------- Utilities ---------------- #
def get_saved_ids(data_dir: Path):
    """Read all Parquet files and return a set of saved query IDs (without pandas)."""
    saved_ids = set()
    for parquet_file in data_dir.glob("*.parquet"):
        table = pq.read_table(parquet_file)
        if "query_id" in table.schema.names:
            # Convert column directly to Python list
            saved_ids.update(table.column("query_id").to_pylist())
        else:
            print(f"[WARN] 'query_id' column not found in {parquet_file}")
    return saved_ids


def find_missing_ids(start_id: int, end_id: int, saved_ids: set):
    """Return a sorted list of missing query IDs in the given range."""
    expected_ids = set(range(start_id, end_id + 1))
    missing_ids = sorted(expected_ids - saved_ids)
    print(f"[INFO] Found {len(missing_ids)} missing IDs in range {start_id}-{end_id}")
    return missing_ids

# ---------------- Retry Function ---------------- #
def retry_queries_while_loop(missing_ids, batch_size=20, delay=0.0):
    """
    Retry missing queries using a while loop and multi-threading.
    """
    records = []
    cursor_index = 0
    total = len(missing_ids)

    while cursor_index < total:
        batch_ids = missing_ids[cursor_index: cursor_index + batch_size]
        futures = []

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            for qid in batch_ids:
                futures.append(executor.submit(fetch_dune_query, qid))
            
            for i, future in enumerate(as_completed(futures)):
                data, req_time = future.result()
                request_times.append(req_time)
                if data:
                    with lock:
                        records.append(data)
                        print(f"[INFO] Retried query {batch_ids[i]}")
                else:
                    print(f"[WARN] Failed retry for query {batch_ids[i]}")

        cursor_index += batch_size
        if delay > 0:
            time.sleep(delay)

    return records

# ---------------- Main ---------------- #
if __name__ == "__main__":
    # Define the ID range to check
    start_id = 1
    end_id = 60000

    print("[INFO] Scanning Parquet files for saved IDs...")
    saved_ids = get_saved_ids(DATA_DIR)

    missing_ids = find_missing_ids(start_id, end_id, saved_ids)

    if not missing_ids:
        print("[INFO] No missing IDs detected. Exiting.")
        exit(0)

    print("[INFO] Retrying missing queries...")
    retried_records = retry_queries_while_loop(missing_ids, batch_size=10, delay=0.1)

    # Save retried queries
    save_parquet(retried_records, "retried_missing_queries.parquet")

    if request_times:
        avg_request_time = sum(request_times) / len(request_times)
        print(f"[INFO] Average request time: {avg_request_time:.3f}s")

    print(f"[INFO] Completed retrying missing queries. Total retried: {len(retried_records)}")
