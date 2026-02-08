"""
Query clustering using HDBSCAN with quality metrics.

Clusters query embeddings using:
- UMAP for dimensionality reduction
- HDBSCAN for density-based clustering
- Comprehensive quality metrics (silhouette, homogeneity, tightness)

Supports multi-threading for efficient processing.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from tqdm import tqdm

# Lazy imports for heavy dependencies
hdbscan = None
UMAP = None


def _load_hdbscan():
    """Lazy load HDBSCAN."""
    global hdbscan
    if hdbscan is None:
        import hdbscan as hdb
        hdbscan = hdb
    return hdbscan


def _load_umap():
    """Lazy load UMAP."""
    global UMAP
    if UMAP is None:
        from umap import UMAP as UMAPClass
        UMAP = UMAPClass
    return UMAP


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ClusterMetrics:
    """Quality metrics for clustering results."""
    n_clusters: int
    n_noise: int
    n_clustered: int
    noise_ratio: float
    
    # Clustering quality
    silhouette_score: Optional[float] = None
    davies_bouldin_index: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    
    # Homogeneity metrics
    homogeneity_score: Optional[float] = None
    completeness_score: Optional[float] = None
    v_measure_score: Optional[float] = None
    
    # Cluster tightness
    avg_cluster_tightness: Optional[float] = None
    cluster_tightness: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'n_clusters': self.n_clusters,
            'n_noise': self.n_noise,
            'n_clustered': self.n_clustered,
            'noise_ratio': self.noise_ratio,
            'silhouette_score': self.silhouette_score,
            'davies_bouldin_index': self.davies_bouldin_index,
            'calinski_harabasz_score': self.calinski_harabasz_score,
            'homogeneity_score': self.homogeneity_score,
            'completeness_score': self.completeness_score,
            'v_measure_score': self.v_measure_score,
            'avg_cluster_tightness': self.avg_cluster_tightness,
            'cluster_tightness': {str(k): v for k, v in self.cluster_tightness.items()},
        }
    
    def print_summary(self):
        """Print formatted metrics summary."""
        print(f"\n[INFO] === Clustering Results ===")
        print(f"[INFO] Number of clusters: {self.n_clusters}")
        print(f"[INFO] Noise points: {self.n_noise} ({self.noise_ratio:.1%})")
        print(f"[INFO] Clustered points: {self.n_clustered}")
        
        if self.silhouette_score is not None:
            print(f"\n[INFO] === Quality Metrics ===")
            print(f"[INFO] Silhouette Score:        {self.silhouette_score:.4f}  (higher is better)")
            print(f"[INFO] Davies-Bouldin Index:    {self.davies_bouldin_index:.4f}  (lower is better)")
            print(f"[INFO] Calinski-Harabasz Score: {self.calinski_harabasz_score:.2f}  (higher is better)")
        
        if self.homogeneity_score is not None:
            print(f"\n[INFO] === Homogeneity Metrics ===")
            print(f"[INFO] Homogeneity Score:       {self.homogeneity_score:.4f}")
            print(f"[INFO] Completeness Score:      {self.completeness_score:.4f}")
            print(f"[INFO] V-Measure Score:         {self.v_measure_score:.4f}")
        
        if self.avg_cluster_tightness is not None:
            print(f"[INFO] Avg Cluster Tightness:   {self.avg_cluster_tightness:.4f}")
        
        # Overall assessment
        print(f"\n[INFO] === Assessment ===")
        if self.silhouette_score is not None:
            if self.silhouette_score > 0.5:
                print(f"[INFO] ✅ EXCELLENT cluster separation")
            elif self.silhouette_score > 0.3:
                print(f"[INFO] ✅ GOOD cluster separation")
            elif self.silhouette_score > 0.1:
                print(f"[INFO] ⚠️  FAIR cluster separation")
            else:
                print(f"[INFO] ❌ POOR cluster separation")


@dataclass
class ClusterStats:
    """Statistics for a single cluster."""
    cluster_id: int
    size: int
    top_tags: List[str]
    sample_names: List[str]
    avg_name_length: float


# =============================================================================
# Query Clusterer Class
# =============================================================================

class QueryClusterer:
    """
    Cluster query embeddings using HDBSCAN.
    
    Example:
        clusterer = QueryClusterer(min_cluster_size=50)
        labels = clusterer.fit(embeddings)
        metrics = clusterer.get_metrics()
    """
    
    def __init__(
        self,
        min_cluster_size: int = 50,
        min_samples: int = 10,
        cluster_selection_epsilon: float = 0.05,
        metric: str = 'euclidean',
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the clusterer.
        
        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for core points
            cluster_selection_epsilon: Epsilon for cluster selection
            metric: Distance metric
            max_workers: Number of threads
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 16)
        self.max_workers = max_workers
        
        # State
        self.embeddings = None
        self.umap_embeddings = None
        self.cluster_labels = None
        self.clusterer = None
        self.metrics = None
        
        print(f"[INFO] QueryClusterer initialized")
        print(f"[INFO] min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        print(f"[INFO] Workers: {max_workers}")
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = 5,
        n_neighbors: int = 15,
        min_dist: float = 0.0,
    ) -> np.ndarray:
        """
        Reduce dimensionality using UMAP.
        
        Args:
            embeddings: Input embeddings
            n_components: Number of output dimensions
            n_neighbors: UMAP neighbors parameter
            min_dist: UMAP min_dist parameter
            
        Returns:
            Reduced embeddings
        """
        print(f"[INFO] Reducing dimensions with UMAP...")
        print(f"[INFO] Input shape: {embeddings.shape}")
        print(f"[INFO] n_components={n_components}, n_neighbors={n_neighbors}")
        
        UMAPClass = _load_umap()
        
        reducer = UMAPClass(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            random_state=42,
            n_jobs=self.max_workers,
        )
        
        self.umap_embeddings = reducer.fit_transform(embeddings)
        print(f"[INFO] UMAP output shape: {self.umap_embeddings.shape}")
        
        return self.umap_embeddings
    
    def fit(
        self,
        embeddings: np.ndarray,
        use_umap: bool = True,
        umap_components: int = 5,
        umap_neighbors: int = 15,
    ) -> np.ndarray:
        """
        Fit the clusterer to embeddings.
        
        Args:
            embeddings: Input embeddings
            use_umap: Whether to apply UMAP first
            umap_components: UMAP output dimensions
            umap_neighbors: UMAP neighbors parameter
            
        Returns:
            Cluster labels (-1 for noise)
        """
        self.embeddings = embeddings
        
        # Optionally reduce dimensions
        if use_umap:
            embeddings_to_cluster = self.reduce_dimensions(
                embeddings,
                n_components=umap_components,
                n_neighbors=umap_neighbors,
            )
        else:
            embeddings_to_cluster = embeddings
        
        # Cluster
        print(f"[INFO] Clustering with HDBSCAN...")
        print(f"[INFO] min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}")
        
        hdb = _load_hdbscan()
        
        self.clusterer = hdb.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method='eom',
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=True,
            core_dist_n_jobs=self.max_workers,
        )
        
        self.cluster_labels = self.clusterer.fit_predict(embeddings_to_cluster)
        
        # Calculate metrics
        self._calculate_metrics(embeddings_to_cluster)
        
        return self.cluster_labels
    
    def _calculate_metrics(self, embeddings: np.ndarray):
        """Calculate comprehensive clustering metrics."""
        from sklearn.metrics import (
            silhouette_score,
            davies_bouldin_score,
            calinski_harabasz_score,
            homogeneity_score,
            completeness_score,
            v_measure_score,
        )
        from sklearn.metrics.pairwise import euclidean_distances
        
        labels = self.cluster_labels
        mask = labels != -1
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        n_clustered = mask.sum()
        n_total = len(labels)
        
        # Basic metrics
        self.metrics = ClusterMetrics(
            n_clusters=n_clusters,
            n_noise=n_noise,
            n_clustered=n_clustered,
            noise_ratio=n_noise / n_total if n_total > 0 else 0,
        )
        
        if n_clustered < 2 or n_clusters < 2:
            print("[WARN] Not enough clustered points for quality metrics")
            return
        
        labels_filtered = labels[mask]
        embeddings_filtered = embeddings[mask]
        
        try:
            # Quality metrics
            sample_size = min(10000, len(labels_filtered))
            self.metrics.silhouette_score = silhouette_score(
                embeddings_filtered,
                labels_filtered,
                metric='euclidean',
                sample_size=sample_size,
            )
            
            self.metrics.davies_bouldin_index = davies_bouldin_score(
                embeddings_filtered,
                labels_filtered,
            )
            
            self.metrics.calinski_harabasz_score = calinski_harabasz_score(
                embeddings_filtered,
                labels_filtered,
            )
            
            # Homogeneity metrics using k-means as pseudo ground truth
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            pseudo_labels = kmeans.fit_predict(embeddings_filtered)
            
            self.metrics.homogeneity_score = homogeneity_score(pseudo_labels, labels_filtered)
            self.metrics.completeness_score = completeness_score(pseudo_labels, labels_filtered)
            self.metrics.v_measure_score = v_measure_score(pseudo_labels, labels_filtered)
            
            # Cluster tightness
            tightness = self._calculate_cluster_tightness(embeddings_filtered, labels_filtered)
            self.metrics.cluster_tightness = tightness
            self.metrics.avg_cluster_tightness = np.mean(list(tightness.values())) if tightness else 0
            
        except Exception as e:
            print(f"[WARN] Could not calculate metrics: {e}")
    
    def _calculate_cluster_tightness(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, float]:
        """Calculate intra-cluster distances (tightness)."""
        from sklearn.metrics.pairwise import euclidean_distances
        
        unique_clusters = set(labels)
        tightness = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id
            cluster_points = embeddings[cluster_mask]
            
            if len(cluster_points) > 1:
                distances = euclidean_distances(cluster_points, cluster_points)
                n = len(cluster_points)
                avg_dist = distances.sum() / (n * (n - 1)) if n > 1 else 0
                tightness[int(cluster_id)] = float(avg_dist)
            else:
                tightness[int(cluster_id)] = 0.0
        
        return tightness
    
    def get_metrics(self) -> Optional[ClusterMetrics]:
        """Get clustering metrics."""
        return self.metrics
    
    def save(self, filepath: Path):
        """Save clusterer state."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'cluster_selection_epsilon': self.cluster_selection_epsilon,
            'metric': self.metric,
            'embeddings': self.embeddings,
            'umap_embeddings': self.umap_embeddings,
            'cluster_labels': self.cluster_labels,
            'metrics': self.metrics.to_dict() if self.metrics else None,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"[INFO] Saved clusterer to {filepath}")
        
        # Also save metrics as JSON
        if self.metrics:
            metrics_file = filepath.parent / 'metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics.to_dict(), f, indent=2)
            print(f"[INFO] Saved metrics to {metrics_file}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'QueryClusterer':
        """Load clusterer from disk."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        clusterer = cls(
            min_cluster_size=state['min_cluster_size'],
            min_samples=state['min_samples'],
            cluster_selection_epsilon=state['cluster_selection_epsilon'],
            metric=state['metric'],
        )
        
        clusterer.embeddings = state['embeddings']
        clusterer.umap_embeddings = state['umap_embeddings']
        clusterer.cluster_labels = state['cluster_labels']
        
        if state['metrics']:
            clusterer.metrics = ClusterMetrics(**state['metrics'])
        
        return clusterer


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_clusters(
    queries: List[Dict],
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Analyze cluster characteristics.
    
    Args:
        queries: List of query dictionaries
        cluster_labels: Cluster labels for each query
        
    Returns:
        DataFrame with cluster statistics
    """
    from collections import Counter
    
    df = pd.DataFrame(queries)
    df['cluster'] = cluster_labels
    
    cluster_stats = []
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_queries = df[df['cluster'] == cluster_id]
        
        # Get most common tags
        all_tags = []
        for tags in cluster_queries['tags']:
            if tags:
                all_tags.extend(tags)
        
        tag_counts = Counter(all_tags)
        top_tags = [tag for tag, _ in tag_counts.most_common(5)]
        
        # Get sample names
        sample_names = cluster_queries['name'].head(3).tolist()
        
        stats = ClusterStats(
            cluster_id=cluster_id,
            size=len(cluster_queries),
            top_tags=top_tags,
            sample_names=sample_names,
            avg_name_length=cluster_queries['name'].str.len().mean(),
        )
        
        cluster_stats.append(vars(stats))
    
    return pd.DataFrame(cluster_stats)


def print_cluster_summary(
    queries: List[Dict],
    cluster_labels: np.ndarray,
    max_clusters: int = 20,
):
    """Print a summary of clustering results."""
    cluster_stats = analyze_clusters(queries, cluster_labels)
    
    print("\n" + "=" * 80)
    print("CLUSTER SUMMARY")
    print("=" * 80)
    
    shown = 0
    for _, row in cluster_stats.iterrows():
        if shown >= max_clusters and row['cluster_id'] != -1:
            continue
        
        cluster_id = row['cluster_id']
        size = row['size']
        
        if cluster_id == -1:
            print(f"\n🔸 NOISE: {size} queries")
        else:
            print(f"\n📊 CLUSTER {cluster_id}: {size} queries")
            print(f"   Tags: {', '.join(row['top_tags'][:5]) if row['top_tags'] else 'None'}")
            print(f"   Samples:")
            for i, name in enumerate(row['sample_names'][:3], 1):
                name_truncated = name[:60] + '...' if len(name) > 60 else name
                print(f"     {i}. {name_truncated}")
            shown += 1
    
    if shown < len(cluster_stats) - 1:
        print(f"\n... and {len(cluster_stats) - shown - 1} more clusters")
    
    print("\n" + "=" * 80)


def save_clusters(
    queries: List[Dict],
    cluster_labels: np.ndarray,
    output_dir: Path,
):
    """Save clustering results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(queries)
    df['cluster'] = cluster_labels
    
    # Save main results
    parquet_path = output_dir / 'clustered_queries.parquet'
    df.to_parquet(parquet_path, compression='zstd')
    print(f"[INFO] Saved to {parquet_path}")
    
    # Save statistics
    stats = analyze_clusters(queries, cluster_labels)
    stats_path = output_dir / 'cluster_statistics.csv'
    stats.to_csv(stats_path, index=False)
    print(f"[INFO] Saved statistics to {stats_path}")
    
    # Save individual clusters
    clusters_dir = output_dir / 'individual_clusters'
    clusters_dir.mkdir(exist_ok=True)
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_df = df[df['cluster'] == cluster_id]
        cluster_file = clusters_dir / f'cluster_{cluster_id}.parquet'
        cluster_df.to_parquet(cluster_file, compression='zstd')
    
    print(f"[INFO] Saved individual clusters to {clusters_dir}")


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Cluster query embeddings')
    parser.add_argument('--embeddings', type=str, required=True, help='Input embeddings .npy file')
    parser.add_argument('--queries', type=str, required=True, help='Input queries JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--min-cluster-size', type=int, default=50)
    parser.add_argument('--min-samples', type=int, default=10)
    parser.add_argument('--umap-components', type=int, default=5)
    parser.add_argument('--no-umap', action='store_true')
    args = parser.parse_args()
    
    # Load data
    print(f"[INFO] Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings)
    
    print(f"[INFO] Loading queries from {args.queries}...")
    with open(args.queries) as f:
        queries = [json.loads(line) for line in f if line.strip()]
    
    print(f"[INFO] Loaded {len(queries)} queries, embeddings shape: {embeddings.shape}")
    
    # Cluster
    clusterer = QueryClusterer(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    
    labels = clusterer.fit(
        embeddings,
        use_umap=not args.no_umap,
        umap_components=args.umap_components,
    )
    
    # Print metrics
    if clusterer.metrics:
        clusterer.metrics.print_summary()
    
    # Print cluster summary
    print_cluster_summary(queries, labels)
    
    # Save
    output_dir = Path(args.output)
    save_clusters(queries, labels, output_dir)
    clusterer.save(output_dir / 'clusterer.pkl')
    
    print(f"\n✅ Done! Results saved to {output_dir}")