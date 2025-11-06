"""
Tag analysis module for Dune queries.
Extracts, counts, and visualizes tags from collected query data.
"""
from pathlib import Path
from collections import Counter
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import plotly.express as px

from typing import List

from src.utils import normalize_text, get_saved_ids, get_query_objects

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "tags_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def count_tags(query_objects: List[dict]) -> Counter:
    """
    Count tag occurrences across all queries efficiently.
    
    Args:
        query_objects: List of query dictionaries
        
    Returns:
        Counter object with tag frequencies
    """
    tag_counter = Counter()
    
    # Efficient counting using Counter.update
    for query in query_objects:
        tags = query.get("tags", [])
        # Normalize and filter empty tags
        normalized_tags = [tag.strip().lower() for tag in tags if tag and tag.strip()]
        tag_counter.update(normalized_tags)
    
    return tag_counter


def get_tag_statistics(query_objects, tag_counter: Counter) -> dict:
    """
    Calculate comprehensive tag statistics.
    
    Args:
        query_objects: List of query dictionaries
        tag_counter: Counter with tag frequencies
        
    Returns:
        Dictionary with various statistics
    """
    total_queries = len(query_objects)
    queries_with_tags = sum(1 for q in query_objects if q.get("tags"))
    queries_without_tags = total_queries - queries_with_tags
    
    total_tag_assignments = sum(len(q.get("tags", [])) for q in query_objects)
    avg_tags_per_query = total_tag_assignments / total_queries if total_queries > 0 else 0
    
    tags_per_query = [len(q.get("tags", [])) for q in query_objects]
    
    stats = {
        "total_queries": total_queries,
        "unique_tags": len(tag_counter),
        "total_tag_assignments": total_tag_assignments,
        "queries_with_tags": queries_with_tags,
        "queries_without_tags": queries_without_tags,
        "avg_tags_per_query": avg_tags_per_query,
        "min_tags_per_query": min(tags_per_query) if tags_per_query else 0,
        "max_tags_per_query": max(tags_per_query) if tags_per_query else 0,
        "percentage_with_tags": (queries_with_tags / total_queries * 100) if total_queries > 0 else 0
    }
    
    return stats


def find_tag_cooccurrences(query_objects, top_n: int = 10) -> List[tuple]:
    """
    Find most common tag pairs that appear together.
    
    Args:
        query_objects: List of query dictionaries
        top_n: Number of top pairs to return
        
    Returns:
        List of ((tag1, tag2), count) tuples
    """
    tag_pairs = Counter()
    
    for query in query_objects:
        tags = sorted(set(query.get("tags", [])))  # Remove duplicates and sort
        if len(tags) >= 2:
            # Generate all pairs
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    pair = (tags[i], tags[j])
                    tag_pairs[pair] += 1
    
    return tag_pairs.most_common(top_n)


def save_tags_parquet(tag_counter: Counter, output_file: Path):
    """
    Save tag counts to parquet file with compression.
    
    Args:
        tag_counter: Counter with tag frequencies
        output_file: Path to save the parquet file
    """
    df = pd.DataFrame(tag_counter.items(), columns=["tag", "count"])
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file, compression='zstd')
    
    print(f"[INFO] Saved {len(tag_counter)} unique tags to {output_file}")


def save_tags_text(tag_counter: Counter, output_file: Path):
    """
    Save tag counts to human-readable text file.
    
    Args:
        tag_counter: Counter with tag frequencies
        output_file: Path to save the text file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with output_file.open("w", encoding="utf-8") as f:
        f.write("TAG\tCOUNT\n")
        f.write("=" * 50 + "\n")
        for tag, count in tag_counter.most_common():
            f.write(f"{tag}\t{count}\n")
    
    print(f"[INFO] Saved tag list to {output_file}")


def save_cooccurrences_text(cooccurrences: List[tuple], output_file: Path):
    """
    Save tag co-occurrences to text file.
    
    Args:
        cooccurrences: List of ((tag1, tag2), count) tuples
        output_file: Path to save the text file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with output_file.open("w", encoding="utf-8") as f:
        f.write("TAG 1\tTAG 2\tCOUNT\n")
        f.write("=" * 60 + "\n")
        for (tag1, tag2), count in cooccurrences:
            f.write(f"{tag1}\t{tag2}\t{count}\n")
    
    print(f"[INFO] Saved co-occurrences to {output_file}")


def create_visualizations(tag_counter: Counter, top_n: int = 50, output_dir: Path = OUTPUT_DIR):
    """
    Create and save interactive visualizations as HTML files.
    
    Args:
        tag_counter: Counter with tag frequencies
        top_n: Number of top tags to visualize
        output_dir: Directory to save visualizations
    """
    df = pd.DataFrame(tag_counter.items(), columns=["tag", "count"])
    df_sorted = df.sort_values("count", ascending=False).head(top_n)
    
    print(f"[INFO] Creating visualizations for top {top_n} tags...")
    
    # 1. Bar chart
    fig_bar = px.bar(
        df_sorted,
        x="tag",
        y="count",
        title=f"Top {top_n} Tags by Frequency",
        labels={"tag": "Tag", "count": "Number of Queries"},
        color="count",
        color_continuous_scale="viridis",
        hover_data={"count": ":,"}
    )
    fig_bar.update_layout(
        xaxis_tickangle=-45,
        height=600,
        hovermode='x unified'
    )
    bar_path = output_dir / "tags_bar_chart.html"
    fig_bar.write_html(bar_path)
    print(f"[INFO] ✓ Bar chart → {bar_path}")
    
    # 2. Scatter plot
    fig_scatter = px.scatter(
        df_sorted,
        x="tag",
        y="count",
        size="count",
        color="count",
        hover_name="tag",
        title=f"Top {top_n} Tags (Interactive Scatter)",
        color_continuous_scale="plasma",
        hover_data={"count": ":,"}
    )
    fig_scatter.update_layout(
        xaxis_tickangle=-45,
        height=600
    )
    scatter_path = output_dir / "tags_scatter.html"
    fig_scatter.write_html(scatter_path)
    print(f"[INFO] ✓ Scatter plot → {scatter_path}")
    
    # 3. Pie chart (top 20 for readability)
    df_pie = df_sorted.head(20)
    fig_pie = px.pie(
        df_pie,
        names="tag",
        values="count",
        title=f"Top 20 Tags Distribution",
        hole=0.3,  # Donut chart
        hover_data={"count": ":,"}
    )
    fig_pie.update_traces(textposition='inside', textinfo='label+percent')
    pie_path = output_dir / "tags_pie_chart.html"
    fig_pie.write_html(pie_path)
    print(f"[INFO] ✓ Pie chart → {pie_path}")
    
    # 4. Treemap
    fig_tree = px.treemap(
        df_sorted,
        path=["tag"],
        values="count",
        title=f"Top {top_n} Tags (Treemap)",
        color="count",
        color_continuous_scale="blues",
        hover_data={"count": ":,"}
    )
    tree_path = output_dir / "tags_treemap.html"
    fig_tree.write_html(tree_path)
    print(f"[INFO] ✓ Treemap → {tree_path}")


def print_statistics(stats, tag_counter: Counter):
    """
    Print formatted statistics to console.
    
    Args:
        stats: Dictionary with statistics
        tag_counter: Counter with tag frequencies
    """
    print("\n" + "=" * 60)
    print("TAG STATISTICS")
    print("=" * 60)
    print(f"Total Queries:              {stats['total_queries']:,}")
    print(f"Queries with Tags:          {stats['queries_with_tags']:,} ({stats['percentage_with_tags']:.1f}%)")
    print(f"Queries without Tags:       {stats['queries_without_tags']:,}")
    print(f"Unique Tags:                {stats['unique_tags']:,}")
    print(f"Total Tag Assignments:      {stats['total_tag_assignments']:,}")
    print(f"Average Tags per Query:     {stats['avg_tags_per_query']:.2f}")
    print(f"Min Tags per Query:         {stats['min_tags_per_query']}")
    print(f"Max Tags per Query:         {stats['max_tags_per_query']}")
    print("=" * 60)
    
    print(f"\nTOP 15 MOST COMMON TAGS:")
    print("-" * 60)
    for i, (tag, count) in enumerate(tag_counter.most_common(15), 1):
        percentage = count / stats['total_queries'] * 100
        bar = "█" * int(percentage / 2)  # Visual bar
        print(f"{i:2d}. {tag:25s} {count:6,} ({percentage:5.1f}%) {bar}")


def main():
    """Main execution: comprehensive tag analysis."""
    print("\n" + "=" * 60)
    print("DUNE QUERY TAG ANALYSIS")
    print("=" * 60 + "\n")
    
    # Load queries
    query_objects = get_query_objects(DATA_DIR, limit=None)
    
    if not query_objects:
        print("[ERROR] No queries found in data directory!")
        print(f"[INFO] Make sure parquet files exist in: {DATA_DIR}")
        return
    
    # Count tags
    print("\n[INFO] Analyzing tags...")
    tag_counter = count_tags(query_objects)
    
    # Calculate statistics
    stats = get_tag_statistics(query_objects, tag_counter)
    
    # Print statistics
    print_statistics(stats, tag_counter)
    
    # Find co-occurrences
    print("\n[INFO] Finding tag co-occurrences...")
    cooccurrences = find_tag_cooccurrences(query_objects, top_n=20)
    
    print(f"\nTOP 10 TAG CO-OCCURRENCES:")
    print("-" * 60)
    for i, ((tag1, tag2), count) in enumerate(cooccurrences[:10], 1):
        print(f"{i:2d}. {tag1:20s} + {tag2:20s} → {count:5,} queries")
    
    # Save results
    print("\n[INFO] Saving results...")
    save_tags_parquet(tag_counter, OUTPUT_DIR / "tags_analysis.parquet")
    save_tags_text(tag_counter, OUTPUT_DIR / "tags_list.txt")
    save_cooccurrences_text(cooccurrences, OUTPUT_DIR / "tag_cooccurrences.txt")
    
    # Create visualizations
    print("\n[INFO] Creating visualizations...")
    create_visualizations(tag_counter, top_n=50, output_dir=OUTPUT_DIR)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    print(f"  📊 tags_analysis.parquet     - Tag data for further analysis")
    print(f"  📄 tags_list.txt             - All tags sorted by frequency")
    print(f"  🔗 tag_cooccurrences.txt     - Common tag pairs")
    print(f"  📈 tags_bar_chart.html       - Interactive bar chart")
    print(f"  🎯 tags_scatter.html         - Interactive scatter plot")
    print(f"  🥧 tags_pie_chart.html       - Interactive pie chart")
    print(f"  🗺️  tags_treemap.html         - Interactive treemap")
    print("\n💡 Open the .html files in your browser to explore!")
    print("=" * 60 + "\n")


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