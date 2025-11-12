"""
Find duplicate queries by analyzing SQL content from clustered parquet files.
Estimates duplication levels across Dune queries using pure SQL comparison.
"""
from pathlib import Path
import hashlib
import json
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import pandas as pd
import re

# Import shared utilities
from src.utils import clean_sql
# from  src.query_duplicate import QueryDuplicateFinder

CLUSTERS_DIR = Path("data")
OUTPUT_DIR = Path("duplicates")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class QueryDuplicateRemover:
    """
    Find duplicate queries by analyzing SQL content.
    
    Supports multiple comparison modes:
    - 'exact': Exact SQL match (after normalization)
    - 'hash': Hash-based matching (fastest)
    - 'structure': Structural similarity (ignores literals)
    """
    def name(one, two):
        """
        Purpose: one
        """
        
    # end def

def main():
    """Main execution: find duplicate queries."""

    
    print("\n" + "=" * 80)
    print(f"QUERY DUPLICATE FINDER MODE")
    print(f"Analyzing: clusters")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] ⚠️  Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] ❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise