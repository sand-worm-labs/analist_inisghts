import re
from pathlib import Path
from collections import Counter
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import plotly.express as px
from itertools import chain
from src.collect import fetch_dune_query, save_parquet  # your custom imports

DATA_DIR = Path("data")
OUTPUT_PARQUET_FILE = DATA_DIR / "tags_analysis/tags_analysis.parquet"

def normalize_text(query):
    query = query.lower()
    query = re.sub(r"\s+", " ", query)  # collapse whitespace
    query = re.sub(r"'[^']*'", "'<val>'", query)  # replace string literals
    query = re.sub(r"\b0x[a-f0-9]+\b", "<address>", query)  # replace hex addresses
    return query.strip()

def get_queries_objects(data_dir: Path, limit: int = None):
    queries = []
    for parquet_file in data_dir.glob("*.parquet"):
        table = pq.read_table(parquet_file).to_pydict()
        ids = table.get("query_id", [])
        names = table.get("name", [])
        descriptions = table.get("description", [])
        tags_list = table.get("tags", [])

        for qid, name, desc, tags in zip(ids, names, descriptions, tags_list):
            if limit and len(queries) >= limit:
                return queries
            queries.append({
                "query_id": qid,
                "name": normalize_text(name),
                "description": normalize_text(desc),
                "tags": [t.lower().strip() for t in tags] if tags else []
            })
    return queries

def get_all_tags_with_count(query_objects):
    """
    Count all tags across queries using a single loop.
    """
    tag_counter = Counter()
    for tags in (query.get("tags", []) for query in query_objects):
        tag_counter.update(tag.strip().lower() for tag in tags if tag.strip())
    return tag_counter

def save_tags_parquet(tag_counter, output_file: Path):
    df = pd.DataFrame(tag_counter.items(), columns=["tag", "count"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)
    print(f"Saved Parquet file to {output_file}")

def interactive_plots(tag_counter, top_n=50):
    df = pd.DataFrame(tag_counter.items(), columns=["tag", "count"])
    df_sorted = df.sort_values("count", ascending=False).head(top_n)

    # Scatter plot
    fig_scatter = px.scatter(
        df_sorted,
        x="tag",
        y="count",
        size="count",
        color="count",
        hover_name="tag",
        title=f"Top {top_n} Tags (Interactive Scatter)"
    )
    fig_scatter.update_layout(xaxis_tickangle=-45)
    fig_scatter.show()

    # Pie chart
    fig_pie = px.pie(
        df_sorted,
        names="tag",
        values="count",
        title=f"Top {top_n} Tags (Pie Chart)"
    )
    fig_pie.show()

if __name__ == "__main__":
    print("Loading queries...")
    query_objects = get_queries_objects(DATA_DIR, limit=200000)
    print(f"Loaded {len(query_objects)} queries.")

    print("Counting tags...")
    tag_counter = get_all_tags_with_count(query_objects)

    print("Saving Parquet file...")
    save_tags_parquet(tag_counter, OUTPUT_PARQUET_FILE)

    print("Creating interactive plots...")
    interactive_plots(tag_counter, top_n=50)