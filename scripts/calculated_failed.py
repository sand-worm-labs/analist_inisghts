import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from src.utils import find_missing_ids, get_saved_ids

DATA_DIR = Path("dataset")


def main():
    start_id, end_id = 1, 1300000
    total_ids = end_id - start_id + 1

    saved_ids = get_saved_ids(DATA_DIR)
    missing_ids = find_missing_ids(start_id, end_id, saved_ids)

    saved_count = len(saved_ids)
    missing_count = len(missing_ids)
    percentage_saved = (saved_count / total_ids) * 100 if total_ids > 0 else 0

    print("[INFO] Total IDs:", total_ids)
    print("[INFO] Saved IDs:", saved_count)
    print("[INFO] Missing IDs:", missing_count)
    print(f"[INFO] Percentage of queries stored: {percentage_saved:.2f}%")

    if not missing_ids:
        print("[INFO] No missing IDs. Exiting.")
        return


if __name__ == "__main__":
    main()
