import json
import time
import requests
from pathlib import Path
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow as pa
import pyarrow.parquet as pq
import random
from src.config import DUNE_API_KEYS, DATA_PATH, PROGRAM_CURSOR, DEBUG

# Paths
CURSOR_FILE = Path(PROGRAM_CURSOR) / "cursor.json"
OUTPUT_DIR = Path(DATA_PATH)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Thread lock for safe updates
lock = threading.Lock()
key_lock = threading.Lock()

times = []

def load_cursor() -> dict:
    """Load or initialize cursor JSON."""
    if CURSOR_FILE.exists():
        try:
            with open(CURSOR_FILE, "r", encoding="utf-8") as f:
                cursor = json.load(f)
                if DEBUG:
                    print(f"[DEBUG] Loaded cursor: {cursor}")
                return cursor
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Failed to load cursor, initializing new: {e}")

    cursor = {
        "total_processed": 0,
        "total_success": 0,
        "total_failed": 0,
        "last_run": None
    }
    save_cursor(cursor)
    return cursor

def save_cursor(cursor: dict):
    """Save the cursor JSON to file."""
    cursor["last_run"] = datetime.utcnow().isoformat()
    CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CURSOR_FILE, "w", encoding="utf-8") as f:
        json.dump(cursor, f, ensure_ascii=False, indent=2)
    if DEBUG:
        print(f"[DEBUG] Saved cursor: {cursor}")


def get_api_key():
    """Thread-safe random key selection."""
    import threading
    with threading.Lock():
        return random.choice(DUNE_API_KEYS)


def fetch_dune_query(query_id: int):
    """
    Fetch Dune query metadata using the API and measure request time.

    Returns:
        tuple: (data, request_time_in_seconds)
    """
    api_key = get_api_key() 
    if DEBUG:
        print(f"[INFO] Using API key: {api_key}")
    url = f"https://api.dune.com/api/v1/query/{query_id}"
    headers = {"X-DUNE-API-KEY": api_key}

    start = time.perf_counter()
    try:
        response = requests.get(url, headers=headers)
        end = time.perf_counter()
        request_time = end - start

        if response.status_code == 200:
            return response.json(), request_time
        elif response.status_code == 404:
            if DEBUG:
                print(f"[DEBUG] Query {query_id} not found (404).")
            return None, request_time
        else:
            if DEBUG:
                print(f"[ERROR] Query {query_id}: {response.status_code} - {response.text}")
            return None, request_time

    except requests.RequestException as e:
        end = time.perf_counter()
        request_time = end - start
        if DEBUG:
            print(f"[ERROR] Request exception for query {query_id}: {e}")
        return None, request_time

def save_parquet(records: list, filename: str):
    """Save a list of records to a Parquet file."""
    if not records:
        return
    table = pa.Table.from_pylist(records)
    parquet_path = OUTPUT_DIR / filename
    pq.write_table(table, parquet_path, compression='zstd')
    if DEBUG:
        print(f"[DEBUG] Saved {len(records)} records to {parquet_path}")

def fetch_and_process(qid: int, cursor: dict, records: list, delay: float):
    """Fetch a single query and update cursor & records safely."""
    data, request_time = fetch_dune_query(qid)
    with lock:
        cursor["total_processed"] += 1
        if data:
            records.append(data)
            cursor["total_success"] += 1
            if DEBUG:
                print(f"[DEBUG] Fetched query {qid}")
        else:
            cursor["total_failed"] += 1
            if DEBUG:
                print(f"[DEBUG] Failed query {qid}")
        save_cursor(cursor)
    if delay > 0:
        time.sleep(delay)

def collect_queries(start_id: int, end_id: int, max_workers: int = 5, delay: float = 0.1):
    """Collect Dune queries concurrently and save them in Parquet."""
    cursor = load_cursor()
    records = []
    parquet_file = f"{start_id}-{end_id}.parquet"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(fetch_and_process, qid, cursor, records, delay)
            for qid in range(start_id, end_id + 1)
        ]
        for _ in as_completed(futures):
            pass  # Optional: could update a progress bar here

    save_parquet(records, parquet_file)

    print("\n✅ Collection complete!")
    print(f"Parquet file: {OUTPUT_DIR / parquet_file}")
    print(f"Total processed: {cursor['total_processed']}, "
          f"Success: {cursor['total_success']}, Failed: {cursor['total_failed']}")
def collect_queries_in_batches(
    start_id: int,
    end_id: int,
    batch_size: int = 500,
    max_workers: int = 10,
    delay: float = 0.1
):
    """
    Run the collect_queries function in batches.

    Args:
        start_id (int): Starting query ID.
        end_id (int): Ending query ID.
        batch_size (int): Number of queries per batch.
        max_workers (int): Number of threads.
        delay (float): Delay between requests in seconds.
    """
    current_start = start_id

    while current_start <= end_id:
        current_end = min(current_start + batch_size - 1, end_id)
        estimate_collection_time_print(
            start_id=current_start,
            end_id=current_end,
            batch_size=batch_size,
            max_workers=max_workers,
            delay=delay
        )
        collect_queries(
            start_id=current_start,
            end_id=current_end,
            max_workers=max_workers,
            delay=delay
        )
        current_start = current_end + 1
def estimate_collection_time_print(
    start_id: int,
    end_id: int,
    batch_size: int,
    max_workers: int,
    delay: float = 0.0,
    avg_request_time: float = 1.0
):
    """
    Estimate total time to fetch queries in batches with multi-threading,
    printing the details for each batch.

    Args:
        start_id (int): Starting query ID.
        end_id (int): Ending query ID.
        batch_size (int): Number of queries per batch.
        max_workers (int): Number of concurrent threads.
        delay (float): Delay between requests per thread in seconds.
        avg_request_time (float): Average time to complete a request (network/API) in seconds.
    """
    total_queries = end_id - start_id + 1
    num_batches = (total_queries + batch_size - 1) // batch_size  # ceil division
    total_rounds = 0

    print(f"Estimating collection time for {total_queries} queries")
    print(f"Batch size: {batch_size}, Max workers: {max_workers}, Delay per request: {delay}s")
    print(f"Average request time per round: {avg_request_time}s\n")

    for batch_num in range(num_batches):
        batch_start = start_id + batch_num * batch_size
        batch_end = min(batch_start + batch_size - 1, end_id)
        current_batch_size = batch_end - batch_start + 1
        rounds = (current_batch_size + max_workers - 1) // max_workers  # ceil division
        total_rounds += rounds

        print(f"Batch {batch_num + 1}/{num_batches}: queries {batch_start}-{batch_end}")
        print(f"  Queries in batch: {current_batch_size}")
        print(f"  Rounds needed (with {max_workers} workers): {rounds}\n")

    total_time_sec = total_rounds * (avg_request_time + delay)
    print("==== Summary ====")
    print(f"Total queries: {total_queries}")
    print(f"Total batches: {num_batches}")
    print(f"Total rounds (20 workers per round): {total_rounds}")
    print(f"Estimated total time: {total_time_sec:.1f} sec / {total_time_sec/60:.1f} min / {total_time_sec/3600:.2f} hr")

if __name__ == "__main__":
    # Example usage
    collect_queries_in_batches(
        start_id=50999,
        end_id=200000,
        batch_size=2000,
        max_workers=20,
        delay=0
    )
