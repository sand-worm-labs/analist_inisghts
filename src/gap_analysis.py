import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from calculated_failed import find_missing_ids, get_saved_ids

DATA_DIR = Path("data")
ANALYZE_DIR = Path("analyze")
ANALYZE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 📊 HISTOGRAM — shows distribution of gap sizes
# ============================================================
def plot_histogram(diffs, bins=50, save_path=None, show=True):
    plt.figure(figsize=(10, 6))
    plt.hist(diffs, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xscale('log')
    plt.title("Histogram of Gaps Between Query IDs (log scale)")
    plt.xlabel("Gap Size Between Consecutive IDs")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved histogram → {save_path}")

    if show:
        plt.show()
    plt.close()


# ============================================================
# ⚪ SCATTER — shows where large gaps appear along ID range
# ============================================================
def plot_scatter(ids, diffs, save_path=None, show=True):
    plt.figure(figsize=(10, 6))
    plt.scatter(ids[:-1], diffs, s=8, alpha=0.6, color='orange')
    plt.title("Scatter Plot of Gaps Across Query IDs")
    plt.xlabel("Query ID (sorted)")
    plt.ylabel("Gap to Next ID")
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved scatter plot → {save_path}")

    if show:
        plt.show()
    plt.close()


# ============================================================
# 🥧 PIE CHART — shows proportion of found vs missing IDs
# ============================================================
def plot_pie(total_found, total_range, save_path=None, show=True):
    total_missing = total_range - total_found
    coverage = total_found / total_range * 100

    plt.figure(figsize=(6, 6))
    plt.pie(
        [total_found, total_missing],
        labels=[f"Found ({coverage:.1f}%)", f"Missing ({100-coverage:.1f}%)"],
        autopct="%1.1f%%",
        colors=["#66c2a5", "#fc8d62"],
        startangle=140,
        textprops={"fontsize": 10},
    )
    plt.title("Coverage of Query ID Range")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved pie chart → {save_path}")

    if show:
        plt.show()
    plt.close()


# ============================================================
# 🧠 MAIN ANALYSIS FUNCTION
# ============================================================
def analyze_id_distribution(ids, total_range, bins=50, show=True, prefix="id_analysis"):
    if not ids:
        print("[WARN] No IDs provided.")
        return None

    ids = sorted(set(ids))
    diffs = np.diff(ids)

    stats = {
        "total_ids": len(ids),
        "min_id": ids[0],
        "max_id": ids[-1],
        "average_gap": float(np.mean(diffs)) if len(diffs) > 0 else 0,
        "median_gap": float(np.median(diffs)) if len(diffs) > 0 else 0,
        "max_gap": int(np.max(diffs)) if len(diffs) > 0 else 0,
        "missing_estimate": int(np.sum(diffs > 1)) if len(diffs) > 0 else 0,
    }

    print("\n[INFO] ID Gap Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Generate each visualization
    plot_histogram(diffs, bins, save_path=ANALYZE_DIR / f"{prefix}_histogram.png", show=show)
    plot_scatter(ids, diffs, save_path=ANALYZE_DIR / f"{prefix}_scatter.png", show=show)
    plot_pie(len(ids), total_range, save_path=ANALYZE_DIR / f"{prefix}_pie.png", show=show)

    return stats


# ============================================================
# 🚀 EXECUTION
# ============================================================
if __name__ == "__main__":
    start_id, end_id = 1, 200000
    total_range = end_id - start_id + 1

    saved_ids = get_saved_ids(DATA_DIR)
    missing_ids = find_missing_ids(start_id, end_id, saved_ids)

    print(f"[INFO] Total saved IDs: {len(saved_ids)}")
    print(f"[INFO] Total missing IDs: {len(missing_ids)}")

    # Analyze saved IDs
    analyze_id_distribution(
        saved_ids,
        total_range=total_range,
        bins=30,
        show=True,
        prefix="saved_id"
    )

    # Analyze missing IDs (optional)
    analyze_id_distribution(
        missing_ids,
        total_range=total_range,
        bins=30,
        show=True,
        prefix="missing_id"
    )
