"""
Gap analysis and visualization for collected query data.
Generates statistical visualizations of data coverage and missing IDs.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

# Import shared utilities
from src.utils import get_saved_ids, find_missing_ids

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "gap_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_histogram(diffs: np.ndarray, bins: int = 50, save_path: Path = None):
    """
    Create histogram showing distribution of gap sizes between IDs.
    
    Args:
        diffs: Array of differences between consecutive IDs
        bins: Number of histogram bins
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.hist(diffs, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
    plt.title("Distribution of Gaps Between Query IDs (Log Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("Gap Size Between Consecutive IDs", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved histogram → {save_path}")
    
    plt.close()


def plot_scatter(ids: np.ndarray, diffs: np.ndarray, save_path: Path = None):
    """
    Create scatter plot showing where large gaps appear across ID range.
    
    Args:
        ids: Array of sorted query IDs
        diffs: Array of differences between consecutive IDs
        save_path: Path to save the plot
    """
    plt.figure(figsize=(14, 6))
    plt.scatter(ids[:-1], diffs, s=10, alpha=0.6, color='orange')
    plt.title("Gap Distribution Across Query ID Range", fontsize=14, fontweight='bold')
    plt.xlabel("Query ID", fontsize=12)
    plt.ylabel("Gap to Next ID", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved scatter plot → {save_path}")
    
    plt.close()


def plot_coverage_pie(total_found: int, total_range: int, save_path: Path = None):
    """
    Create pie chart showing proportion of found vs missing IDs.
    
    Args:
        total_found: Number of IDs found
        total_range: Total ID range size
        save_path: Path to save the plot
    """
    total_missing = total_range - total_found
    coverage = total_found / total_range * 100

    plt.figure(figsize=(8, 8))
    colors = ['#66c2a5', '#fc8d62']
    explode = (0.05, 0)
    
    plt.pie(
        [total_found, total_missing],
        labels=[f"Found\n{coverage:.1f}%", f"Missing\n{100-coverage:.1f}%"],
        autopct=lambda p: f'{p:.1f}%\n({int(p/100 * total_range):,} IDs)',
        colors=colors,
        explode=explode,
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'},
        shadow=True
    )
    plt.title("Query ID Coverage", fontsize=16, fontweight='bold', pad=20)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved pie chart → {save_path}")
    
    plt.close()


def plot_coverage_timeline(ids: np.ndarray, total_range: int, save_path: Path = None):
    """
    Create timeline showing cumulative coverage across ID range.
    
    Args:
        ids: Array of sorted query IDs
        total_range: Total ID range size
        save_path: Path to save the plot
    """
    # Calculate cumulative coverage percentage
    coverage_pct = np.arange(1, len(ids) + 1) / total_range * 100
    
    plt.figure(figsize=(14, 6))
    plt.plot(ids, coverage_pct, linewidth=2, color='#2ecc71')
    plt.fill_between(ids, coverage_pct, alpha=0.3, color='#2ecc71')
    plt.title("Cumulative Coverage Across Query ID Range", fontsize=14, fontweight='bold')
    plt.xlabel("Query ID", fontsize=12)
    plt.ylabel("Cumulative Coverage (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved timeline → {save_path}")
    
    plt.close()


def analyze_gaps(ids: List[int], total_range: int, prefix: str = "analysis") -> Dict:
    """
    Analyze gap distribution and generate visualizations.
    
    Args:
        ids: List of query IDs
        total_range: Total size of ID range
        prefix: Prefix for output filenames
        
    Returns:
        Dictionary with gap statistics
    """
    if not ids:
        print("[WARN] No IDs provided for analysis")
        return {}

    ids = sorted(set(ids))
    ids_array = np.array(ids)
    diffs = np.diff(ids_array)

    # Calculate statistics
    stats = {
        "total_ids": len(ids),
        "min_id": int(ids[0]),
        "max_id": int(ids[-1]),
        "id_range_span": int(ids[-1] - ids[0] + 1),
        "average_gap": float(np.mean(diffs)),
        "median_gap": float(np.median(diffs)),
        "max_gap": int(np.max(diffs)),
        "gaps_over_1": int(np.sum(diffs > 1)),
        "gaps_over_10": int(np.sum(diffs > 10)),
        "gaps_over_100": int(np.sum(diffs > 100)),
        "coverage_percentage": float(len(ids) / total_range * 100)
    }

    # Print statistics
    print(f"\n[INFO] === {prefix.replace('_', ' ').title()} Statistics ===")
    print(f"[INFO] Total IDs: {stats['total_ids']:,}")
    print(f"[INFO] ID Range: {stats['min_id']:,} to {stats['max_id']:,}")
    print(f"[INFO] Coverage: {stats['coverage_percentage']:.2f}%")
    print(f"[INFO] Average Gap: {stats['average_gap']:.2f}")
    print(f"[INFO] Median Gap: {stats['median_gap']:.0f}")
    print(f"[INFO] Max Gap: {stats['max_gap']:,}")
    print(f"[INFO] Gaps > 1: {stats['gaps_over_1']:,}")
    print(f"[INFO] Gaps > 10: {stats['gaps_over_10']:,}")
    print(f"[INFO] Gaps > 100: {stats['gaps_over_100']:,}")

    # Generate visualizations
    print(f"\n[INFO] Generating visualizations...")
    plot_histogram(diffs, bins=50, save_path=OUTPUT_DIR / f"{prefix}_histogram.png")
    plot_scatter(ids_array, diffs, save_path=OUTPUT_DIR / f"{prefix}_scatter.png")
    plot_coverage_pie(len(ids), total_range, save_path=OUTPUT_DIR / f"{prefix}_pie.png")
    plot_coverage_timeline(ids_array, total_range, save_path=OUTPUT_DIR / f"{prefix}_timeline.png")

    return stats


def main():
    """Main execution: analyze data gaps and generate visualizations."""
    print("[INFO] === Dune Query Gap Analysis ===\n")
    
    start_id, end_id = 1, 400000
    total_range = end_id - start_id + 1

    print(f"[INFO] Analyzing ID range: {start_id:,} to {end_id:,}")
    print(f"[INFO] Total range size: {total_range:,}")

    # Get saved and missing IDs
    saved_ids = get_saved_ids(DATA_DIR)
    missing_ids = find_missing_ids(start_id, end_id, saved_ids)

    print(f"\n[INFO] Total saved IDs: {len(saved_ids):,}")
    print(f"[INFO] Total missing IDs: {len(missing_ids):,}")

    # Analyze saved IDs
    print(f"\n{'='*60}")
    print("[INFO] ANALYZING SAVED IDs")
    print(f"{'='*60}")
    saved_stats = analyze_gaps(list(saved_ids), total_range, prefix="saved_ids")

    # Analyze missing IDs
    if missing_ids:
        print(f"\n{'='*60}")
        print("[INFO] ANALYZING MISSING IDs")
        print(f"{'='*60}")
        missing_stats = analyze_gaps(missing_ids, total_range, prefix="missing_ids")

    print(f"\n✅ Analysis complete! Visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        raise