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

def retry_queries_while_loop(missing_ids, batch_size=10, delay=0.1):
    records = []
    total = len(missing_ids)

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for i in range(0, total, batch_size):
            batch = missing_ids[i:i + batch_size]
            futures = {executor.submit(fetch_dune_query, qid): qid for qid in batch}

            for future in as_completed(futures):
                qid = futures[future]
                try:
                    data, req_time = future.result()
                    request_times.append(req_time)
                    if data:
                        with lock:
                            records.append(data)
                            print(f"[INFO] Retried query {qid}")
                    else:
                        print(f"[WARN] No data for {qid}")
                except Exception as e:
                    print(f"[ERROR] Query {qid} failed: {e}")

            if delay > 0:
                time.sleep(delay)
    return records

if __name__ == "__main__":
    start_id, end_id = 1, 60000
    saved_ids = get_saved_ids(DATA_DIR)
    missing_ids = find_missing_ids(start_id, end_id, saved_ids)

    if not missing_ids:
        print("[INFO] No missing IDs. Exiting.")
        exit(0)

    retried_records = retry_queries_while_loop(missing_ids)
    save_parquet(retried_records, OUTPUT_DIR / "retried_missing_queries.parquet")

    if request_times:
        avg_request_time = sum(request_times) / len(request_times)
        print(f"[INFO] Average request time: {avg_request_time:.3f}s")

    print(f"[INFO] Completed retrying missing queries. Total retried: {len(retried_records)}")
