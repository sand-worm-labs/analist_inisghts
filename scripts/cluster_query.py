import re

import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from itertools import cycle
from src.collect import fetch_dune_query, save_parquet

DATA_DIR = Path("data")


def normalize_text(query):
    query = query.lower()
    query = re.sub(r"\s+", " ", query)  # collapse whitespace
    query = re.sub(r"'[^']*'", "'<val>'", query)  # replace string literals
    query = re.sub(r"\b0x[a-f0-9]+\b", "<address>", query)  # replace hex addresses
    return query.strip()


from pathlib import Path
from collections import Counter

def save_all_tags_with_count(query_objects, output_file: Path):
    """
    Collect all tags from query objects, count their occurrences, 
    and write them to a file with counts.

    Args:
        query_objects (list[dict]): List of query objects containing 'tags'.
        output_file (Path): Path to the output file to save tags and counts.
    """
    tag_counter = Counter()
    
    # Count tags across all queries
    for query in query_objects:
        tags = query.get("tags", [])
        for tag in tags:
            normalized_tag = tag.strip().lower()
            if normalized_tag:
                tag_counter[normalized_tag] += 1

    # Write tags and counts to file
    with output_file.open("w", encoding="utf-8") as f:
        for tag, count in tag_counter.most_common():  # sorted by frequency
            f.write(f"{tag}\t{count}\n")

    print(f"Saved {len(tag_counter)} unique tags with counts to {output_file}")


def get_queries_objects(data_dir: Path, limit: int = 10):
    """
    Fetch a limited number of query objects from Parquet files
    and normalize their text fields (name, description) and keep tags.
    """
    queries = []
    for parquet_file in data_dir.glob("*.parquet"):
        table = pq.read_table(parquet_file).to_pydict()
        ids = table.get("query_id", [])
        names = table.get("name", [])
        descriptions = table.get("description", [])
        tags_list = table.get("tags", [])  # assuming tags is a list per row

        for qid, name, desc, tags in zip(ids, names, descriptions, tags_list):
            if len(queries) >= limit:
                return queries
            queries.append({
                "query_id": qid,
                "name": normalize_text(name),
                "description": normalize_text(desc),
                "tags": [t.lower().strip() for t in tags] if tags else []
            })
    return queries

# def get_queries(data_dir: Path, limit:1000):
#     saved_ids = []
#     for parquet_file in data_dir.glob("*.parquet"):
#         table = pq.read_table(parquet_file)
#         if "query_id" in table.schema.names:
#             ids = table.column("query_id").to_pylist()
#             for qid in ids:
#                 if len(saved_ids) >= limit:
#                     return saved_ids
#                 saved_ids.append(qid)
#     return saved_ids


# if __name__ == "__main__":
#     query = "SELECT * FROM table WHERE column = 'value'"
#     normalized_query = normalize_query(query)
#     print(normalized_query)

if __name__ == "__main__":
    # Example usage with your query objects
    query_objects = get_queries_objects(DATA_DIR, limit=200000)  # your function
    save_all_tags_with_count(query_objects, Path("all_tags.txt"))