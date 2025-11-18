"""
Remove duplicate queries from the dataset.
Keeps only the first occurrence of each duplicate query.
"""

from pathlib import Path
import pandas as pd
from typing import List, Dict
from src.utils import get_query_objects, clean_sql, normalize_sql, normalize_text, compute_hash
import json

DATA_DIR = Path("dataset")
OUTPUT_DIR = Path("dataset_deduplicated")
REPORT_DIR = Path("duplicates_removed")


class DuplicateRemover:
    """Remove duplicate queries and save cleaned dataset."""

    def __init__(self, mode: str = "sql"):
        """
        Args:
            mode: 'sql' (deduplicate by query_sql) or 'semantic' (by name/description/tags)
        """
        self.mode = mode.lower()
        if self.mode not in ["sql", "semantic"]:
            raise ValueError("mode must be 'sql' or 'semantic'")
        print(f"[INFO] Deduplication mode: {self.mode.upper()}")

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

    def remove_duplicates(self, query_objects: List[Dict]) -> tuple[List[Dict], pd.DataFrame]:
        """
        Remove duplicates, keeping first occurrence.
        Returns: (deduplicated_queries, removed_duplicates_df)
        """
        print(f"[INFO] Processing {len(query_objects):,} queries...")

        df = pd.DataFrame(query_objects)
        df["normalized_text"] = df.apply(self.prepare_text, axis=1)
        df["hash"] = df["normalized_text"].apply(compute_hash)

        # Identify duplicates
        df["is_duplicate"] = df.duplicated(subset=["hash"], keep="first")
        
        original_count = len(df)
        duplicate_count = df["is_duplicate"].sum()
        kept_count = original_count - duplicate_count

        print("\n[INFO] === DEDUPLICATION SUMMARY ===")
        print(f"[INFO] Original queries:     {original_count:,}")
        print(f"[INFO] Duplicates removed:   {duplicate_count:,}")
        print(f"[INFO] Unique queries kept:  {kept_count:,}")
        print(f"[INFO] Reduction:            {(duplicate_count/original_count*100):.2f}%")

        # Get removed duplicates for reporting
        removed_df = df[df["is_duplicate"]].copy()
        
        # Keep only unique queries
        unique_df = df[~df["is_duplicate"]].copy()
        
        # Convert back to list of dicts (without helper columns)
        unique_df = unique_df.drop(columns=["normalized_text", "hash", "is_duplicate"])
        deduplicated_queries = unique_df.to_dict("records")

        return deduplicated_queries, removed_df

    def save_deduplicated_data(self, deduplicated_queries: List[Dict], output_dir: Path):
        """Save deduplicated queries to disk in same format as original."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet
        df = pd.DataFrame(deduplicated_queries)
        parquet_file = output_dir / "queries_deduplicated.parquet"
        df.to_parquet(parquet_file, compression="zstd")
        print(f"[INFO] ✅ Saved deduplicated dataset to {parquet_file}")
        
        # Optionally save as JSON lines for readability
        jsonl_file = output_dir / "queries_deduplicated.jsonl"
        with open(jsonl_file, 'w') as f:
            for query in deduplicated_queries:
                f.write(json.dumps(query) + '\n')
        print(f"[INFO] ✅ Saved deduplicated dataset to {jsonl_file}")

    def save_removal_report(self, removed_df: pd.DataFrame, report_dir: Path):
        """Save report of removed duplicates."""
        if removed_df.empty:
            print("[INFO] No duplicates were removed.")
            return
        
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save removed duplicates
        removed_file = report_dir / f"removed_duplicates_{self.mode}.parquet"
        removed_df.to_parquet(removed_file, compression="zstd")
        print(f"[INFO] ✅ Saved removal report to {removed_file}")
        
        # Create summary report
        summary = removed_df.groupby("hash").agg(
            count=("hash", "size"),
            sample_names=("name", lambda x: list(x.head(3))),
            owners=("owner", lambda x: list(set(x))),
        ).reset_index().sort_values("count", ascending=False)
        
        summary_file = report_dir / f"removal_summary_{self.mode}.csv"
        summary.to_csv(summary_file, index=False)
        print(f"[INFO] ✅ Saved removal summary to {summary_file}")
        
        print("\n[INFO] === TOP 10 MOST COMMON DUPLICATES (REMOVED) ===")
        for _, row in summary.head(10).iterrows():
            print(
                f"- {row['count']:>3d}× duplicates  |  Owners: {', '.join(row['owners'][:3])}  "
                f"|  Samples: {', '.join(row['sample_names'][:2])}"
            )


def main():
    """Main entry: remove duplicates from dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Remove duplicate Dune queries")
    parser.add_argument("--mode", choices=["sql", "semantic"], default="sql",
                        help="Deduplication mode (default: sql)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit for testing")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(f"DUNE QUERY DEDUPLICATION - {args.mode.upper()} MODE")
    print("=" * 80 + "\n")

    # Load queries
    query_objects = get_query_objects(DATA_DIR, limit=args.limit)
    if not query_objects:
        print("[ERROR] No queries found!")
        return

    # Remove duplicates
    remover = DuplicateRemover(mode=args.mode)
    deduplicated_queries, removed_df = remover.remove_duplicates(query_objects)

    # Save results
    remover.save_deduplicated_data(deduplicated_queries, OUTPUT_DIR)
    remover.save_removal_report(removed_df, REPORT_DIR)

    print("\n✅ Deduplication complete!")
    print(f"[INFO] Deduplicated dataset saved to: {OUTPUT_DIR}")
    print(f"[INFO] Removal report saved to: {REPORT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()