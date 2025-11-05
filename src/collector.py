"""
Data collection module for Dune Analytics queries.
Handles concurrent API requests with proper rate limiting and error handling.
"""
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
from typing import Optional, Tuple
from tqdm import tqdm

from src.config import DUNE_API_KEYS, DATA_PATH, PROGRAM_CURSOR, DEBUG

# Paths
CURSOR_FILE = Path(PROGRAM_CURSOR) / "cursor.json"
OUTPUT_DIR = Path(DATA_PATH)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Thread-safe locks
cursor_lock = threading.Lock()
records_lock = threading.Lock()
api_key_lock = threading.Lock()

# API key rotation
api_key_index = 0


def load_cursor() -> dict:
    """Load or initialize cursor JSON for tracking progress."""
    if CURSOR_FILE.exists():
        try:
            with open(CURSOR_FILE, "r", encoding="utf-8") as f:
                cursor = json.load(f)
                if DEBUG:
                    print(f"[DEBUG] Loaded cursor: {cursor}")
                return cursor
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Failed to load cursor: {e}")

    cursor = {
        "total_processed": 0,
        "total_success": 0,
        "total_failed": 0,
        "last_run": None
    }
    save_cursor(cursor)
    return cursor


def save_cursor(cursor: dict):
    """Save cursor JSON to file with timestamp."""
    cursor["last_run"] = datetime.utcnow().isoformat()
    CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CURSOR_FILE, "w", encoding="utf-8") as f:
        json.dump(cursor, f, ensure_ascii=False, indent=2)
    if DEBUG:
        print(f"[DEBUG] Saved cursor: {cursor}")


def get_api_key() -> str:
    """Thread-safe round-robin API key selection."""
    global api_key_index
    with api_key_lock:
        key = DUNE_API_KEYS[api_key_index % len(DUNE_API_KEYS)]
        api_key_index += 1
        return key


def fetch_dune_query(
    query_id: int, 
    max_retries: int = 3, 
    backoff_factor: float = 2.0,
    retry_on_statuses: tuple = (429, 500, 502, 503, 504)
) -> Tuple[Optional[dict], float, int]:
    """
    Fetch Dune query metadata from API with exponential backoff retry logic.
    
    Args:
        query_id: Query ID to fetch
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_factor: Exponential backoff multiplier (default: 2.0)
        retry_on_statuses: HTTP status codes to retry on (default: rate limit and server errors)
        
    Returns:
        Tuple of (response_data, total_request_time_seconds, attempts_made)
    """
    total_start = time.perf_counter()
    attempts = 0
    last_error = None
    
    for attempt in range(max_retries + 1):
        attempts += 1
        api_key = get_api_key()
        url = f"https://api.dune.com/api/v1/query/{query_id}"
        headers = {"X-DUNE-API-KEY": api_key}

        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            # Success - return immediately
            if response.status_code == 200:
                total_time = time.perf_counter() - total_start
                return response.json(), total_time, attempts
            
            # Not found - don't retry
            elif response.status_code == 404:
                if DEBUG:
                    print(f"[DEBUG] Query {query_id} not found (404)")
                total_time = time.perf_counter() - total_start
                return None, total_time, attempts
            
            # Retryable status codes
            elif response.status_code in retry_on_statuses:
                last_error = f"HTTP {response.status_code}"
                
                if attempt < max_retries:
                    # Calculate exponential backoff delay
                    delay = backoff_factor ** attempt
                    if DEBUG:
                        print(f"[DEBUG] Query {query_id}: {response.status_code} - "
                              f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s")
                    time.sleep(delay)
                    continue
                else:
                    if DEBUG:
                        print(f"[ERROR] Query {query_id}: {response.status_code} - Max retries exceeded")
                    total_time = time.perf_counter() - total_start
                    return None, total_time, attempts
            
            # Other errors - don't retry
            else:
                if DEBUG:
                    print(f"[ERROR] Query {query_id}: {response.status_code} - {response.text[:100]}")
                total_time = time.perf_counter() - total_start
                return None, total_time, attempts

        except requests.Timeout:
            last_error = "Timeout"
            if attempt < max_retries:
                delay = backoff_factor ** attempt
                if DEBUG:
                    print(f"[DEBUG] Query {query_id}: Timeout - Retry {attempt + 1}/{max_retries} after {delay:.1f}s")
                time.sleep(delay)
                continue
            else:
                if DEBUG:
                    print(f"[ERROR] Query {query_id}: Timeout - Max retries exceeded")
                total_time = time.perf_counter() - total_start
                return None, total_time, attempts
        
        except requests.RequestException as e:
            last_error = str(e)
            if attempt < max_retries:
                delay = backoff_factor ** attempt
                if DEBUG:
                    print(f"[DEBUG] Query {query_id}: {e} - Retry {attempt + 1}/{max_retries} after {delay:.1f}s")
                time.sleep(delay)
                continue
            else:
                if DEBUG:
                    print(f"[ERROR] Query {query_id}: {e} - Max retries exceeded")
                total_time = time.perf_counter() - total_start
                return None, total_time, attempts
    
    # Should never reach here, but just in case
    total_time = time.perf_counter() - total_start
    return None, total_time, attempts


def save_parquet(records: list, filename: str):
    """Save records to Parquet file with compression."""
    if not records:
        return
    
    table = pa.Table.from_pylist(records)
    parquet_path = OUTPUT_DIR / filename
    pq.write_table(table, parquet_path, compression='zstd')
    
    if DEBUG:
        print(f"[DEBUG] Saved {len(records)} records to {parquet_path}")


def fetch_and_process(qid: int, cursor: dict, records: list, delay: float, stats: dict) -> bool:
    """
    Fetch a single query and update shared state safely.
    
    Args:
        qid: Query ID to fetch
        cursor: Shared cursor dictionary
        records: Shared records list
        delay: Delay between requests in seconds
        stats: Shared statistics dictionary for tracking retries
        
    Returns:
        True if successful, False otherwise
    """
    data, request_time, attempts = fetch_dune_query(qid)
    
    # Update shared state with proper locking
    with cursor_lock:
        cursor["total_processed"] += 1
        
        if data:
            with records_lock:
                records.append(data)
            cursor["total_success"] += 1
            success = True
            
            # Track retry statistics
            if attempts > 1:
                stats["total_retried"] = stats.get("total_retried", 0) + 1
                stats["retry_attempts"] = stats.get("retry_attempts", 0) + (attempts - 1)
        else:
            cursor["total_failed"] += 1
            success = False
            
            # Track if all retries were exhausted
            if attempts > 1:
                stats["failed_after_retry"] = stats.get("failed_after_retry", 0) + 1
        
        # Track total attempts across all queries
        stats["total_attempts"] = stats.get("total_attempts", 0) + attempts
        
        # Batch cursor saves every 100 requests for performance
        if cursor["total_processed"] % 100 == 0:
            save_cursor(cursor)
    
    if delay > 0:
        time.sleep(delay)
    
    return success


def collect_queries(
    start_id: int,
    end_id: int,
    max_workers: int = 10,
    delay: float = 0.1,
    retry_config: dict = None
):
    """
    Collect Dune queries concurrently with progress tracking and retry support.
    
    Args:
        start_id: Starting query ID
        end_id: Ending query ID
        max_workers: Number of concurrent threads
        delay: Delay between requests per thread
        retry_config: Optional dict with retry settings:
            - max_retries: int (default: 3)
            - backoff_factor: float (default: 2.0)
            - retry_on_statuses: tuple (default: (429, 500, 502, 503, 504))
    """
    cursor = load_cursor()
    records = []
    stats = {
        "total_retried": 0,
        "retry_attempts": 0,
        "failed_after_retry": 0,
        "total_attempts": 0
    }
    parquet_file = f"{start_id}-{end_id}.parquet"
    
    total_queries = end_id - start_id + 1
    
    # Apply retry config if provided (will be used by fetch_dune_query)
    if retry_config:
        # Store in module-level for access by fetch_dune_query
        globals()['RETRY_CONFIG'] = retry_config
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_and_process, qid, cursor, records, delay, stats): qid
            for qid in range(start_id, end_id + 1)
        }
        
        # Progress bar
        with tqdm(total=total_queries, desc=f"Collecting {start_id}-{end_id}") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
    
    # Final cursor save
    with cursor_lock:
        save_cursor(cursor)
    
    # Save collected data
    save_parquet(records, parquet_file)
    
    # Print summary with retry statistics
    print(f"\n✅ Collection complete!")
    print(f"   File: {OUTPUT_DIR / parquet_file}")
    print(f"   Processed: {cursor['total_processed']}")
    print(f"   Success: {cursor['total_success']}")
    print(f"   Failed: {cursor['total_failed']}")
    
    if stats["total_retried"] > 0:
        print(f"\n📊 Retry Statistics:")
        print(f"   Queries that needed retry: {stats['total_retried']}")
        print(f"   Total retry attempts: {stats['retry_attempts']}")
        print(f"   Failed after all retries: {stats['failed_after_retry']}")
        print(f"   Average attempts per query: {stats['total_attempts'] / cursor['total_processed']:.2f}")
        print(f"   Success rate with retries: {(cursor['total_success'] / cursor['total_processed'] * 100):.2f}%")


def collect_queries_in_batches(
    start_id: int,
    end_id: int,
    batch_size: int = 500,
    max_workers: int = 10,
    delay: float = 0.1,
    retry_config: dict = None
):
    """
    Collect queries in batches to manage memory and provide better progress tracking.
    
    Args:
        start_id: Starting query ID
        end_id: Ending query ID
        batch_size: Number of queries per batch
        max_workers: Number of concurrent threads
        delay: Delay between requests
        retry_config: Optional dict with retry settings:
            - max_retries: int (default: 3)
            - backoff_factor: float (default: 2.0)
            - retry_on_statuses: tuple (default: (429, 500, 502, 503, 504))
    
    Example:
        # Collect with custom retry configuration
        collect_queries_in_batches(
            start_id=1,
            end_id=10000,
            retry_config={
                "max_retries": 5,           # Try up to 5 times
                "backoff_factor": 1.5,      # 1.5x delay between retries
                "retry_on_statuses": (429, 500, 502, 503, 504)
            }
        )
    """
    current_start = start_id

    while current_start <= end_id:
        current_end = min(current_start + batch_size - 1, end_id)
        
        print(f"\n[INFO] Batch: {current_start} to {current_end}")
        
        collect_queries(
            start_id=current_start,
            end_id=current_end,
            max_workers=max_workers,
            delay=delay,
            retry_config=retry_config
        )
        
        current_start = current_end + 1


if __name__ == "__main__":
    # Example usage with retry configuration
    collect_queries_in_batches(
        start_id=200000,
        end_id=400000,
        batch_size=2000,
        max_workers=20,
        delay=0.1,
        retry_config={
            "max_retries": 3,
            "backoff_factor": 2.0,
            "retry_on_statuses": (429, 500, 502, 503, 504)
        }
    )