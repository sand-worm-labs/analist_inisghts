"""
Query duplication analysis.
Finds and reports exact or near-duplicate queries within the dataset itself
(before or outside clustering).

Supports two modes:
- 'semantic': Detects duplicate metadata text (name, description, tags)
- 'sql': Detects duplicate SQL queries (after removing comments & normalizing)
"""

from pathlib import Path
import pandas as pd
import re
from typing import List, Dict
from src.utils import get_query_objects, clean_sql, normalize_sql, normalize_text, compute_hash

DATA_DIR = Path("data")
OUTPUT_DIR = Path("duplicates")
DUPLICATE_REMOVED_DIR = Path("duplicates_removed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class QueryDuplicateFinder:
    """Detect exact duplicates within the dataset."""

    def __init__(self, mode: str = "sql"):
        """
        Args:
            mode: 'sql' (query_sql) or 'semantic' (name/description/tags)
        """
        self.mode = mode.lower()
        if self.mode not in ["sql", "semantic"]:
            raise ValueError("mode must be 'sql' or 'semantic'")
        print(f"[INFO] Mode: {self.mode.upper()}")

    def prepare_text(self, query_obj: Dict) -> str:
        """Prepare text for hashing based on mode."""
        if self.mode == "sql":
            sql = query_obj.get("query_sql", "")
            return normalize_sql(sql)
        else:
            parts = []
            if query_obj.get("name"):
                parts.append(query_obj["name"])
            if query_obj.get("description"):
                parts.append(query_obj["description"])
            if query_obj.get("tags"):
                parts.extend(query_obj["tags"])
            joined = " ".join(parts)
            return normalize_text(joined)

    def find_duplicates(self, query_objects: List[Dict]) -> pd.DataFrame:
        """Find and summarize duplicates."""
        print(f"[INFO] Checking {len(query_objects):,} queries for duplicates...")

        df = pd.DataFrame(query_objects)
        df["normalized_text"] = df.apply(self.prepare_text, axis=1)
        df["hash"] = df["normalized_text"].apply(compute_hash)

        # Count duplicates
        dup_counts = df["hash"].value_counts()
        duplicate_hashes = dup_counts[dup_counts > 1]

        if duplicate_hashes.empty:
            print("[INFO] ✅ No exact duplicates found!")
            return pd.DataFrame()

        print(f"[INFO] Found {len(duplicate_hashes):,} unique duplicate patterns")

        duplicates_df = df[df["hash"].isin(duplicate_hashes.index)].copy()

        # Group duplicates
        grouped = (
            duplicates_df.groupby("hash")
            .agg(
                count=("hash", "size"),
                sample_name=("name", lambda x: list(x.head(3))),
                owners=("owner", lambda x: list(set(x))),
            )
            .reset_index()
            .sort_values("count", ascending=False)
        )

        total = len(df)
        dup_total = len(duplicates_df)
        dup_rate = dup_total / total * 100

        print("\n[INFO] === DUPLICATION SUMMARY ===")
        print(f"[INFO] Total queries:        {total:,}")
        print(f"[INFO] Duplicate entries:    {dup_total:,}")
        print(f"[INFO] Duplicate ratio:      {dup_rate:.2f}%")

        print("\n[INFO] === TOP 10 DUPLICATE GROUPS ===")
        for _, row in grouped.head(10).iterrows():
            print(
                f"- {row['count']:>3d}×  |  Owners: {', '.join(row['owners'][:3])}  "
                f"|  Samples: {', '.join(row['sample_name'][:3])}"
            )

        return duplicates_df

    def save_results(self, duplicates_df: pd.DataFrame, output_dir: Path):
        """Save duplicates to disk."""
        if duplicates_df.empty:
            print("[INFO] No duplicates to save.")
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"duplicates_{self.mode}.parquet"
        duplicates_df.to_parquet(out_file, compression="zstd")
        print(f"[INFO] ✅ Saved duplicate details to {out_file}")
    
    def replace_parquet_file_afeter_removeing_duplicates(self, file_path: Path):
        # if file_path.exists():
        #     file_path.unlink()
        # new_file_path.rename(file_path)

def main():
    """Main entry: find duplicates in raw query dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Find duplicate Dune queries")
    parser.add_argument("--mode", choices=["sql", "semantic"], default="sql",
                        help="Duplicate detection mode")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit for faster testing")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(f"DUNE QUERY DUPLICATION ANALYSIS - {args.mode.upper()} MODE")
    print("=" * 80 + "\n")

    query_objects = get_query_objects(DATA_DIR, limit=args.limit)
    if not query_objects:
        print("[ERROR] No queries found!")
        return

    finder = QueryDuplicateFinder(mode=args.mode)
    duplicates = finder.find_duplicates(query_objects)
    finder.save_results(duplicates, OUTPUT_DIR)

    print("\n✅ Duplicate analysis complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
