"""
Data collection module for Dune Analytics queries.
Handles concurrent API requests with proper rate limiting, retries,
cursor tracking, and logging (buffered to file and console).
"""

import json
import time
import requests
import threading
import atexit
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from tqdm import tqdm

from src.config import Config 
from src.utils import ensure_dirs, save_parquet


class BufferHandler(logging.Handler):
    """Custom logging handler that buffers logs in memory for later writing to file."""
    def __init__(self):
        super().__init__()
        self.buffer: List[str] = []

    def emit(self, record: logging.LogRecord):
        self.buffer.append(self.format(record))


class DuneCollector:
    """Collector class for fetching Dune Analytics queries concurrently."""

    def __init__(
        self,
        config: Config,
        max_workers: int = 10,
        delay: float = 0.1,
        retry_config: Optional[Dict] = None
    ):
        self.config = config
        self.logger = config.logger
        self.max_workers = max_workers
        self.delay = delay

        # Retry parameters
        self.max_retries = retry_config.get("max_retries", 3) if retry_config else 3
        self.backoff_factor = retry_config.get("backoff_factor", 2.0) if retry_config else 2.0
        self.retry_on_statuses = retry_config.get("retry_on_statuses", (429, 500, 502, 503, 504)) if retry_config else (429, 500, 502, 503, 504)

        # Paths
        self.cursor_file = Path(self.config.PROGRAM_CURSOR) / "cursor.json"
        self.output_dir = Path(self.config.DATA_PATH)
        ensure_dirs(self.output_dir, self.cursor_file.parent)

        # Thread-safe locks
        self.cursor_lock = threading.Lock()
        self.records_lock = threading.Lock()
        self.api_key_lock = threading.Lock()

        # State
        self.api_key_index = 0
        self.log_buffer_handler = BufferHandler()
        self.logger.addHandler(self.log_buffer_handler)

        # Ensure logs are saved on exit
        atexit.register(self._save_logs)

    def _save_logs(self):
        log_file = Path(self.config.LOGS_PATH) / "dune_collector.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(self.log_buffer_handler.buffer) + "\n")
        print(f"[INFO] Logs saved to {log_file}")

    def load_cursor(self) -> dict:
        """
        Load or initialize cursor, setting total_processed to the largest query_id
        already saved in Parquet files.
        """
        cursor = {
            "total_processed": 0,
            "total_success": 0,
            "total_failed": 0,
            "last_run": None
        }

        if self.cursor_file.exists():
            try:
                with open(self.cursor_file, "r", encoding="utf-8") as f:
                    cursor_json = json.load(f)
                    cursor.update(cursor_json)
                    if self.config.DEBUG:
                        self.logger.debug("Loaded cursor JSON: %s", cursor_json)
            except Exception as e:
                self.logger.warning("Failed to load cursor JSON: %s", e)

        max_query_id = 0
        for parquet_file in self.output_dir.glob("*.parquet"):
            try:
                table = pq.read_table(parquet_file)
                if "query_id" in table.schema.names:
                    ids = table.column("query_id").to_pylist()
                    if ids:
                        max_query_id = max(max_query_id, max(ids))
            except Exception as e:
                self.logger.warning("Failed to read Parquet file %s: %s", parquet_file, e)

        cursor["total_processed"] = max(cursor.get("total_processed", 0), max_query_id)
        cursor["total_success"] = cursor["total_processed"] 

        if self.config.DEBUG:
            self.logger.debug("Cursor after scanning Parquet files: %s", cursor)

        # Save updated cursor
        self.save_cursor(cursor)
        return cursor

    def save_cursor(self, cursor: dict):
        cursor["last_run"] = datetime.utcnow().isoformat()
        self.cursor_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cursor_file, "w", encoding="utf-8") as f:
            json.dump(cursor, f, ensure_ascii=False, indent=2)
        if self.config.DEBUG:
            self.logger.debug("Saved cursor: %s", cursor)

    def get_api_key(self) -> str:
        with self.api_key_lock:
            key = self.config.DUNE_API_KEYS[self.api_key_index % len(self.config.DUNE_API_KEYS)]
            self.api_key_index += 1
            return key

    def fetch_query(self, query_id: int) -> Tuple[Optional[dict], float, int]:
        """Fetch a single query with retry logic and exponential backoff."""
        total_start = time.perf_counter()
        attempts = 0

        for attempt in range(self.max_retries + 1):
            attempts += 1
            api_key = self.get_api_key()
            url = f"https://api.dune.com/api/v1/query/{query_id}"
            headers = {"X-DUNE-API-KEY": api_key}

            try:
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    return response.json(), time.perf_counter() - total_start, attempts
                elif response.status_code == 404:
                    if self.config.DEBUG:
                        self.logger.debug("Query %d not found (404)", query_id)
                    return None, time.perf_counter() - total_start, attempts
                elif response.status_code in self.retry_on_statuses:
                    if attempt < self.max_retries:
                        delay = self.backoff_factor ** attempt
                        if self.config.DEBUG:
                            self.logger.debug(
                                "Query %d: %d - Retry %d/%d after %.1fs",
                                query_id, response.status_code, attempt+1, self.max_retries, delay
                            )
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error("Query %d: %d - Max retries exceeded", query_id, response.status_code)
                        return None, time.perf_counter() - total_start, attempts
                else:
                    self.logger.error("Query %d: %d - %s", query_id, response.status_code, response.text[:100])
                    return None, time.perf_counter() - total_start, attempts

            except requests.RequestException as e:
                if attempt < self.max_retries:
                    delay = self.backoff_factor ** attempt
                    if self.config.DEBUG:
                        self.logger.debug(
                            "Query %d: %s - Retry %d/%d after %.1fs",
                            query_id, e, attempt+1, self.max_retries, delay
                        )
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error("Query %d failed: %s - Max retries exceeded", query_id, e)
                    return None, time.perf_counter() - total_start, attempts

        return None, time.perf_counter() - total_start, attempts


    def fetch_and_process(
        self,
        query_id: int,
        cursor: dict,
        records: List[dict],
        stats: dict
    ) -> bool:
        """Fetch a query and update cursor/stats with proper locking."""
        data, _, attempts = self.fetch_query(query_id)

        with self.cursor_lock:
            cursor["total_processed"] += 1
            if data:
                with self.records_lock:
                    records.append(data)
                cursor["total_success"] += 1
                stats["total_retried"] += max(0, attempts-1)
            else:
                cursor["total_failed"] += 1
                stats["failed_after_retry"] += max(0, attempts-1)
            stats["total_attempts"] += attempts

            if cursor["total_processed"] % 100 == 0:
                self.save_cursor(cursor)

        if self.delay > 0:
            time.sleep(self.delay)

        return data is not None

    def collect_queries(
        self,
        start_id: int,
        end_id: int,
        batch_size: int = 500
    ):
        """Collect queries in batches to avoid excessive memory usage."""
        cursor = self.load_cursor()
        records: List[dict] = []
        stats = {"total_retried": 0, "failed_after_retry": 0, "total_attempts": 0}

        current_start = start_id
        while current_start <= end_id:
            current_end = min(current_start + batch_size - 1, end_id)
            self.logger.info("Processing batch: %d to %d", current_start, current_end)

            total_queries = current_end - current_start + 1
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.fetch_and_process, qid, cursor, records, stats): qid
                    for qid in range(current_start, current_end + 1)
                }
                with tqdm(total=total_queries, desc=f"Batch {current_start}-{current_end}") as pbar:
                    for future in as_completed(futures):
                        pbar.update(1)

            # Save intermediate results
            self.save_cursor(cursor)
            self.save_parquet(records, f"{current_start}-{current_end}.parquet")
            records.clear()
            current_start = current_end + 1

        self.logger.info("✅ Collection complete")
        self.logger.info(
            "Processed: %d, Success: %d, Failed: %d",
            cursor["total_processed"], cursor["total_success"], cursor["total_failed"]
        )
        self.logger.info(
            "Retry stats - Total retried: %d, Failed after retry: %d, Avg attempts/query: %.2f",
            stats["total_retried"], stats["failed_after_retry"],
            stats["total_attempts"] / cursor["total_processed"]
        )


# === Example usage ===
if __name__ == "__main__":
    config = Config()
    collector = DuneCollector(
        config=config,
        max_workers=20,
        delay=0.1,
        retry_config={"max_retries": 100, "backoff_factor": 2.0, "retry_on_statuses": (429, 500, 502, 503, 504)}
    )
    collector.collect_queries(start_id=236000, end_id=400000, batch_size=2_000)
