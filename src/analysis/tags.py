"""
Tag analysis and visualization for collected query data.
Analyzes tag distribution, suggests missing tags, and generates visualizations.
"""
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import Counter
from wordcloud import WordCloud

# Import shared utilities
from src.utils import load_queries, iter_queries

DATA_DIR = Path("data")
OUTPUT_DIR = Path("analysis") / "tags"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Tag Extraction Rules (table patterns → suggested tags)
# =============================================================================

TABLE_TAG_RULES = {
    # Chains
    r"ethereum\.": ["ethereum", "evm"],
    r"polygon\.": ["polygon", "evm"],
    r"arbitrum\.": ["arbitrum", "l2"],
    r"optimism\.": ["optimism", "l2"],
    r"base\.": ["base", "l2"],
    r"solana\.": ["solana"],
    r"_solana\.": ["solana"],
    
    # DEX
    r"uniswap": ["uniswap", "dex"],
    r"sushiswap": ["sushiswap", "dex"],
    r"curve": ["curve", "dex"],
    r"dex\.trades": ["dex"],
    r"dex_solana": ["dex", "solana"],
    r"orca": ["orca", "dex", "solana"],
    r"raydium": ["raydium", "dex", "solana"],
    
    # Lending
    r"aave": ["aave", "lending"],
    r"compound": ["compound", "lending"],
    r"morpho": ["morpho", "lending"],
    
    # NFT
    r"nft\.": ["nft"],
    r"opensea": ["opensea", "nft"],
    r"blur": ["blur", "nft"],
    r"erc721": ["nft"],
    
    # Other
    r"prices\.usd": ["prices"],
    r"labels\.": ["labels"],
    r"farcaster": ["farcaster", "social"],
    r"lens": ["lens", "social"],
    r"bridge": ["bridge"],
    r"lido": ["lido", "staking"],
}

PATTERN_TAG_RULES = {
    r"date_trunc\s*\(\s*'day'": ["daily"],
    r"date_trunc\s*\(\s*'hour'": ["hourly"],
    r"row_number\s*\(": ["ranking"],
    r"sum\s*\([^)]+\)\s*over": ["running-total"],
    r"union\s+all": ["multi-chain"],
    r"volume|amount_usd": ["volume"],
    r"holder|holders": ["holders"],
    r"transfer": ["transfers"],
    r"swap": ["swaps"],
    r"pnl|profit": ["pnl"],
    r"tvl": ["tvl"],
}


def extract_implicit_tags(query: Dict) -> Set[str]:
    """
    Extract suggested tags from SQL content.
    
    Args:
        query: Query dictionary with query_sql
        
    Returns:
        Set of suggested tags
    """
    tags = set()
    sql = query.get("query_sql", "").lower()
    
    for pattern, tag_list in TABLE_TAG_RULES.items():
        if re.search(pattern, sql):
            tags.update(tag_list)
    
    for pattern, tag_list in PATTERN_TAG_RULES.items():
        if re.search(pattern, sql):
            tags.update(tag_list)
    
    return tags


def plot_tag_distribution(tag_counts: Counter, top_n: int = 30, save_path: Path = None):
    """
    Create bar chart of tag distribution.
    
    Args:
        tag_counts: Counter of tag occurrences
        top_n: Number of top tags to show
        save_path: Path to save the plot
    """
    top_tags = tag_counts.most_common(top_n)
    tags, counts = zip(*top_tags) if top_tags else ([], [])
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(list(reversed(tags)), list(reversed(counts)), color='#3498db', edgecolor='black')
    plt.xlabel("Number of Queries", fontsize=12)
    plt.ylabel("Tag", fontsize=12)
    plt.title(f"Top {top_n} Tags by Frequency", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved tag distribution → {save_path}")
    
    plt.close()


def plot_tag_coverage_pie(with_tags: int, without_tags: int, save_path: Path = None):
    """
    Create pie chart showing queries with vs without tags.
    
    Args:
        with_tags: Number of queries with tags
        without_tags: Number of queries without tags
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 8))
    colors = ['#27ae60', '#95a5a6']
    explode = (0.05, 0)
    
    plt.pie(
        [with_tags, without_tags],
        labels=[f"With Tags\n{with_tags:,}", f"No Tags\n{without_tags:,}"],
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'},
        shadow=True
    )
    plt.title("Query Tag Coverage", fontsize=16, fontweight='bold', pad=20)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved coverage pie → {save_path}")
    
    plt.close()


def plot_wordcloud(tag_counts: Counter, save_path: Path = None):
    """
    Create word cloud of tags.
    
    Args:
        tag_counts: Counter of tag occurrences
        save_path: Path to save the plot
    """
    if not tag_counts:
        return
    
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate_from_frequencies(dict(tag_counts))
    
    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Tag Word Cloud", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved word cloud → {save_path}")
    
    plt.close()


def plot_implicit_vs_explicit(explicit: Counter, implicit: Counter, top_n: int = 15, save_path: Path = None):
    """
    Create grouped bar chart comparing explicit vs implicit tags.
    
    Args:
        explicit: Counter of explicit tags
        implicit: Counter of implicit (suggested) tags
        top_n: Number of tags to show
        save_path: Path to save the plot
    """
    # Get top tags from both
    all_tags = set(dict(explicit.most_common(top_n)).keys()) | set(dict(implicit.most_common(top_n)).keys())
    all_tags = sorted(all_tags, key=lambda t: explicit.get(t, 0) + implicit.get(t, 0), reverse=True)[:top_n]
    
    explicit_counts = [explicit.get(t, 0) for t in all_tags]
    implicit_counts = [implicit.get(t, 0) for t in all_tags]
    
    x = np.arange(len(all_tags))
    width = 0.35
    
    plt.figure(figsize=(14, 8))
    bars1 = plt.bar(x - width/2, explicit_counts, width, label='Explicit Tags', color='#3498db')
    bars2 = plt.bar(x + width/2, implicit_counts, width, label='Implicit (Suggested)', color='#e74c3c', alpha=0.7)
    
    plt.xlabel("Tag", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Explicit vs Implicit Tags", fontsize=14, fontweight='bold')
    plt.xticks(x, all_tags, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved comparison chart → {save_path}")
    
    plt.close()


def analyze_tags(queries: List[Dict]) -> Dict:
    """
    Analyze tag distribution and generate visualizations.
    
    Args:
        queries: List of query dictionaries
        
    Returns:
        Dictionary with tag statistics
    """
    print(f"\n[INFO] Analyzing tags for {len(queries):,} queries...")
    
    # Count explicit tags
    explicit_tags: Counter = Counter()
    queries_with_tags = 0
    queries_without_tags = 0
    
    for q in queries:
        tags = q.get("tags", [])
        if tags:
            queries_with_tags += 1
            explicit_tags.update(t.lower() for t in tags)
        else:
            queries_without_tags += 1
    
    # Count implicit (suggested) tags
    implicit_tags: Counter = Counter()
    queries_needing_tags = 0
    
    for q in queries:
        existing = set(t.lower() for t in q.get("tags", []))
        suggested = extract_implicit_tags(q)
        new_tags = suggested - existing
        
        if new_tags:
            queries_needing_tags += 1
        
        implicit_tags.update(suggested)
    
    # Calculate statistics
    stats = {
        "total_queries": len(queries),
        "queries_with_tags": queries_with_tags,
        "queries_without_tags": queries_without_tags,
        "tag_coverage_rate": queries_with_tags / len(queries) * 100 if queries else 0,
        "total_explicit_tags": sum(explicit_tags.values()),
        "unique_explicit_tags": len(explicit_tags),
        "total_implicit_tags": sum(implicit_tags.values()),
        "unique_implicit_tags": len(implicit_tags),
        "queries_needing_tags": queries_needing_tags,
        "top_explicit_tags": explicit_tags.most_common(20),
        "top_implicit_tags": implicit_tags.most_common(20),
    }
    
    # Print statistics
    print(f"\n[INFO] === Tag Analysis Statistics ===")
    print(f"[INFO] Total queries: {stats['total_queries']:,}")
    print(f"[INFO] Queries with tags: {stats['queries_with_tags']:,}")
    print(f"[INFO] Queries without tags: {stats['queries_without_tags']:,}")
    print(f"[INFO] Tag coverage: {stats['tag_coverage_rate']:.1f}%")
    print(f"[INFO] Unique explicit tags: {stats['unique_explicit_tags']:,}")
    print(f"[INFO] Unique implicit tags: {stats['unique_implicit_tags']:,}")
    print(f"[INFO] Queries that could use more tags: {stats['queries_needing_tags']:,}")
    
    print(f"\n[INFO] Top 10 Explicit Tags:")
    for tag, count in stats['top_explicit_tags'][:10]:
        print(f"[INFO]   {tag}: {count:,}")
    
    print(f"\n[INFO] Top 10 Implicit Tags:")
    for tag, count in stats['top_implicit_tags'][:10]:
        print(f"[INFO]   {tag}: {count:,}")
    
    # Generate visualizations
    print(f"\n[INFO] Generating visualizations...")
    plot_tag_distribution(explicit_tags, top_n=30, save_path=OUTPUT_DIR / "tag_distribution.png")
    plot_tag_coverage_pie(queries_with_tags, queries_without_tags, save_path=OUTPUT_DIR / "tag_coverage.png")
    
    try:
        plot_wordcloud(explicit_tags, save_path=OUTPUT_DIR / "tag_wordcloud.png")
    except Exception as e:
        print(f"[WARN] Could not generate word cloud: {e}")
    
    plot_implicit_vs_explicit(explicit_tags, implicit_tags, top_n=15, save_path=OUTPUT_DIR / "explicit_vs_implicit.png")
    
    # Save tag data to JSON
    tag_data = {
        "statistics": {k: v for k, v in stats.items() if not k.startswith("top_")},
        "explicit_tags": dict(explicit_tags.most_common()),
        "implicit_tags": dict(implicit_tags.most_common()),
    }
    
    output_file = OUTPUT_DIR / "tag_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(tag_data, f, indent=2)
    print(f"[INFO] Saved tag analysis → {output_file}")
    
    return stats


def main():
    """Main execution: analyze tags in collected queries."""
    print("[INFO] === Dune Query Tag Analysis ===\n")
    
    # Load queries
    queries = list(iter_queries(DATA_DIR))
    
    if not queries:
        print("[ERROR] No queries found in data directory")
        return
    
    # Analyze
    stats = analyze_tags(queries)
    
    print(f"\n✅ Tag analysis complete! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        raise