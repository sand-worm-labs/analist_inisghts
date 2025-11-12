"""
Query clustering using HDBSCAN and sentence embeddings with multi-threading.
Supports TWO modes: semantic (text) and sql (query patterns).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import hdbscan
from umap import UMAP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from sklearn.metrics.pairwise import euclidean_distances

# Import shared utilities
from src.utils import get_query_objects, clean_sql

DATA_DIR = Path("data")
OUTPUT_DIR_SEMANTIC = Path("clusters") / "semantic"
OUTPUT_DIR_SQL = Path("clusters") / "sql"
OUTPUT_DIR_SEMANTIC.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_SQL.mkdir(parents=True, exist_ok=True)


class QueryClusterer:
    """
    Cluster queries using embeddings and HDBSCAN with multi-threading support.
    
    Supports two modes:
    - 'semantic': Clusters by name + description + tags (WHAT it's about)
    - 'sql': Clusters by query_sql (HOW it works)
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        min_cluster_size: int = 1000,
        min_samples: int = 90,
        mode: str = 'semantic',
        max_workers: int = 20
    ):
        """
        Initialize the clusterer.
        
        Args:
            model_name: Sentence transformer model to use
            min_cluster_size: Minimum size for HDBSCAN clusters
            min_samples: Minimum samples for HDBSCAN
            mode: 'semantic' (text fields) or 'sql' (query_sql only)
            max_workers: Number of worker threads (default: CPU count)
        """
        self.mode = mode.lower()
        
        if self.mode not in ['semantic', 'sql']:
            raise ValueError(f"mode must be 'semantic' or 'sql', got: {mode}")
        
        # Set workers
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 20)
        self.max_workers = 20
        
        print(f"[INFO] Mode: {self.mode.upper()}")
        print(f"[INFO] Workers: {self.max_workers}")
        print(f"[INFO] Loading model: {model_name}")
        
        self.model = SentenceTransformer(model_name)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.embeddings = None
        self.cluster_labels = None
        self.umap_embeddings = None
        self.clusterer = None
        self.metrics = None
        
    def prepare_text(self, query_obj) -> str:
        """
        Prepare text for embedding based on mode.
        
        Args:
            query_obj: Query object dictionary
            
        Returns:
            Text string to embed
        """
        if self.mode == 'sql':
            sql = query_obj.get('query_sql', '')
            sql = clean_sql(sql)  
            return sql
        
        else:  # semantic mode
            parts = []
            
            if query_obj.get('name'):
                parts.append(query_obj['name'])
            
            if query_obj.get('description'):
                parts.append(query_obj['description'])
            
            if query_obj.get('owner'):
                parts.append(query_obj['owner'])
            
            tags = query_obj.get('tags', [])
            if tags:
                parts.append(' '.join(tags))
            
            return ' '.join(parts)
    
    def prepare_texts_batch(self, query_objects: List[Dict], start_idx: int, end_idx: int) -> List[str]:
        """Prepare texts for a batch of queries (for parallel processing)."""
        return [self.prepare_text(q) for q in query_objects[start_idx:end_idx]]
    
    def create_embeddings(
        self, 
        query_objects: List[Dict], 
        batch_size: int = 32,
        use_parallel: bool = True
    ) -> np.ndarray:
        """
        Create embeddings for all queries with optional parallel text preparation.
        
        Args:
            query_objects: List of query dictionaries
            batch_size: Batch size for encoding
            use_parallel: Whether to use parallel processing for text preparation
            
        Returns:
            Numpy array of embeddings
        """
        print(f"[INFO] Preparing text for {len(query_objects):,} queries (mode={self.mode})...")
        
        # Parallel text preparation
        if use_parallel and len(query_objects) > 10000:
            texts = self._prepare_texts_parallel(query_objects)
        else:
            texts = [self.prepare_text(q) for q in tqdm(query_objects, desc="Preparing text")]
        
        # Check for empty texts
        empty_count = sum(1 for t in texts if not t.strip())
        if empty_count > 0:
            print(f"[WARN] {empty_count} queries have empty text in {self.mode} mode")
        
        print(f"[INFO] Generating embeddings (batch_size={batch_size})...")
        
        # Sentence transformers already uses multi-processing internally
        # We can increase batch size for better throughput
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # Keep raw embeddings
        )
        
        self.embeddings = embeddings
        print(f"[INFO] Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def _prepare_texts_parallel(self, query_objects: List[Dict]) -> List[str]:
        """Prepare texts in parallel using ThreadPoolExecutor."""
        print(f"[INFO] Using {self.max_workers} workers for parallel text preparation...")
        
        # Split into chunks
        chunk_size = max(100, len(query_objects) // (self.max_workers * 4))
        chunks = []
        for i in range(0, len(query_objects), chunk_size):
            chunks.append((i, min(i + chunk_size, len(query_objects))))
        
        texts = [''] * len(query_objects)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.prepare_texts_batch, query_objects, start, end): (start, end)
                for start, end in chunks
            }
            
            with tqdm(total=len(chunks), desc="Text preparation") as pbar:
                for future in as_completed(futures):
                    start, end = futures[future]
                    try:
                        batch_texts = future.result()
                        texts[start:end] = batch_texts
                    except Exception as e:
                        print(f"\n[ERROR] Failed to process chunk {start}-{end}: {e}")
                    pbar.update(1)
        
        return texts
    
    def reduce_dimensions(self, n_components: int = 5, n_neighbors: int = 15):
        """
        Reduce dimensionality using UMAP for better clustering.
        
        Args:
            n_components: Number of UMAP components
            n_neighbors: UMAP neighbors parameter
        """
        print(f"[INFO] Reducing dimensions with UMAP...")
        print(f"[INFO] n_components={n_components}, n_neighbors={n_neighbors}")
        
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric='cosine',
            random_state=42,
            n_jobs=self.max_workers  # Use multi-threading in UMAP
        )
        
        self.umap_embeddings = reducer.fit_transform(self.embeddings)
        print(f"[INFO] UMAP embeddings shape: {self.umap_embeddings.shape}")
        return self.umap_embeddings
    
    def cluster(self, embeddings: np.ndarray = None) -> np.ndarray:
        """
        Perform HDBSCAN clustering.
        
        Args:
            embeddings: Embeddings to cluster (uses UMAP if available)
            
        Returns:
            Cluster labels
        """
        # Determine which embeddings to use
        if embeddings is not None:
            embeddings_to_use = embeddings
        elif self.umap_embeddings is not None:
            embeddings_to_use = self.umap_embeddings
        else:
            embeddings_to_use = self.embeddings
        
        print(f"[INFO] Clustering with HDBSCAN...")
        print(f"[INFO] min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}")
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            cluster_selection_epsilon=0.05,
            core_dist_n_jobs=self.max_workers  # Use multi-threading in HDBSCAN
        )
        
        self.cluster_labels = self.clusterer.fit_predict(embeddings_to_use)
        
        # Calculate metrics
        self._calculate_metrics(embeddings_to_use)
        
        # Statistics
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        print(f"\n[INFO] === Clustering Results ===")
        print(f"[INFO] Number of clusters: {n_clusters}")
        print(f"[INFO] Noise points (unclustered): {n_noise}")
        print(f"[INFO] Clustered points: {len(self.cluster_labels) - n_noise}")
        
        return self.cluster_labels
    
    def _calculate_metrics(self, embeddings: np.ndarray):
        """Calculate clustering quality metrics including homogeneity."""
        mask = self.cluster_labels != -1
        
        if mask.sum() < 2:
            print("[WARN] Not enough clustered points for quality metrics")
            return
        
        labels_filtered = self.cluster_labels[mask]
        embeddings_filtered = embeddings[mask]
        n_clusters = len(set(labels_filtered))
        
        if n_clusters < 2:
            print("[WARN] Need at least 2 clusters for quality metrics")
            return
        
        try:
            # Basic metrics
            silhouette = silhouette_score(
                embeddings_filtered, 
                labels_filtered, 
                metric='euclidean', 
                sample_size=min(10000, len(labels_filtered))
            )
            
            davies_bouldin = davies_bouldin_score(embeddings_filtered, labels_filtered)
            calinski = calinski_harabasz_score(embeddings_filtered, labels_filtered)
            
            # Homogeneity metrics (using k-means as pseudo ground truth)
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            pseudo_labels = kmeans.fit_predict(embeddings_filtered)
            
            homogeneity = homogeneity_score(pseudo_labels, labels_filtered)
            completeness = completeness_score(pseudo_labels, labels_filtered)
            v_measure = v_measure_score(pseudo_labels, labels_filtered)
            
            # Cluster tightness (parallelized)
            cluster_tightness = self._calculate_cluster_tightness_parallel(
                embeddings_filtered, labels_filtered
            )
            avg_tightness = np.mean(list(cluster_tightness.values()))
            
            # Print results
            print(f"\n[INFO] === Clustering Quality Metrics ===")
            print(f"[INFO] Silhouette Score:        {silhouette:.4f}  (higher is better)")
            print(f"[INFO] Davies-Bouldin Index:    {davies_bouldin:.4f}  (lower is better)")
            print(f"[INFO] Calinski-Harabasz Score: {calinski:.2f}  (higher is better)")
            
            print(f"\n[INFO] === Homogeneity Metrics ===")
            print(f"[INFO] Homogeneity Score:       {homogeneity:.4f}  (higher = purer clusters)")
            print(f"[INFO] Completeness Score:      {completeness:.4f}  (higher is better)")
            print(f"[INFO] V-Measure Score:         {v_measure:.4f}  (harmonic mean)")
            print(f"[INFO] Avg Cluster Tightness:   {avg_tightness:.4f}  (lower = tighter)")
            
            # Top tightest clusters
            print(f"\n[INFO] === Top 5 Tightest Clusters ===")
            sorted_clusters = sorted(cluster_tightness.items(), key=lambda x: x[1])
            for cluster_id, tightness in sorted_clusters[:5]:
                cluster_size = (labels_filtered == cluster_id).sum()
                print(f"[INFO] Cluster {cluster_id:3d}: {tightness:.4f}  ({cluster_size:4d} queries)")
            
            # Overall assessment
            print(f"\n[INFO] === Overall Assessment ===")
            if silhouette > 0.5:
                print(f"[INFO] ✅ EXCELLENT separation")
            elif silhouette > 0.3:
                print(f"[INFO] ✅ GOOD separation")
            elif silhouette > 0.1:
                print(f"[INFO] ⚠️  FAIR separation")
            else:
                print(f"[INFO] ❌ POOR separation")
            
            if homogeneity > 0.7:
                print(f"[INFO] ✅ VERY HOMOGENEOUS (pure clusters)")
            elif homogeneity > 0.5:
                print(f"[INFO] ✅ HOMOGENEOUS (good purity)")
            elif homogeneity > 0.3:
                print(f"[INFO] ⚠️  MODERATELY HOMOGENEOUS")
            else:
                print(f"[INFO] ❌ LOW HOMOGENEITY (mixed clusters)")
            
            # Save metrics
            self.metrics = {
                'mode': self.mode,
                'silhouette_score': float(silhouette),
                'davies_bouldin_index': float(davies_bouldin),
                'calinski_harabasz_score': float(calinski),
                'homogeneity_score': float(homogeneity),
                'completeness_score': float(completeness),
                'v_measure_score': float(v_measure),
                'avg_cluster_tightness': float(avg_tightness),
                'cluster_tightness': cluster_tightness,
                'n_clusters': int(n_clusters),
                'n_noise': int(list(self.cluster_labels).count(-1)),
                'n_clustered': int(mask.sum())
            }
            
        except Exception as e:
            print(f"[WARN] Could not calculate metrics: {e}")
            import traceback
            traceback.print_exc()
            self.metrics = None
    
    def _calculate_single_cluster_tightness(self, cluster_id: int, embeddings: np.ndarray, labels: np.ndarray) -> tuple:
        """Calculate tightness for a single cluster."""
        cluster_mask = labels == cluster_id
        cluster_points = embeddings[cluster_mask]
        
        if len(cluster_points) > 1:
            distances = euclidean_distances(cluster_points, cluster_points)
            avg_distance = distances.sum() / (len(cluster_points) * (len(cluster_points) - 1))
            return (int(cluster_id), float(avg_distance))
        else:
            return (int(cluster_id), 0.0)
    
    def _calculate_cluster_tightness_parallel(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
        """Calculate cluster tightness in parallel."""
        unique_clusters = set(labels)
        cluster_tightness = {}
        
        # Only parallelize if we have many clusters
        if len(unique_clusters) > 10:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._calculate_single_cluster_tightness, cid, embeddings, labels): cid
                    for cid in unique_clusters
                }
                
                for future in as_completed(futures):
                    cid, tightness = future.result()
                    cluster_tightness[cid] = tightness
        else:
            # Sequential for small number of clusters
            for cluster_id in unique_clusters:
                cid, tightness = self._calculate_single_cluster_tightness(cluster_id, embeddings, labels)
                cluster_tightness[cid] = tightness
        
        return cluster_tightness
    
    def save_model(self, filepath: Path):
        """Save clusterer state including metrics."""
        state = {
            'mode': self.mode,
            'embeddings': self.embeddings,
            'cluster_labels': self.cluster_labels,
            'umap_embeddings': self.umap_embeddings,
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'metrics': self.metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"[INFO] Saved model state to {filepath}")
        
        # Also save metrics as JSON
        if self.metrics:
            metrics_file = filepath.parent / 'clustering_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            print(f"[INFO] Saved metrics to {metrics_file}")


def analyze_clusters(query_objects: List[Dict], cluster_labels: np.ndarray) -> pd.DataFrame:
    """Analyze cluster characteristics."""
    df = pd.DataFrame(query_objects)
    df['cluster'] = cluster_labels
    
    cluster_stats = []
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_queries = df[df['cluster'] == cluster_id]
        
        # Get most common tags
        all_tags = []
        for tags in cluster_queries['tags']:
            if tags:
                all_tags.extend(tags)
        
        from collections import Counter
        tag_counts = Counter(all_tags)
        top_tags = [tag for tag, _ in tag_counts.most_common(5)]
        
        # Get sample names
        sample_names = cluster_queries['name'].head(3).tolist()
        
        stats = {
            'cluster_id': cluster_id,
            'size': len(cluster_queries),
            'top_tags': top_tags,
            'sample_names': sample_names,
            'avg_name_length': cluster_queries['name'].str.len().mean()
        }
        
        cluster_stats.append(stats)
    
    return pd.DataFrame(cluster_stats)


def extract_cluster_keywords(query_objects: List[Dict], cluster_labels: np.ndarray, mode: str, top_n: int = 10):
    """Extract representative keywords for each cluster using TF-IDF."""
    print(f"[INFO] Extracting keywords for each cluster...")
    
    df = pd.DataFrame(query_objects)
    df['cluster'] = cluster_labels
    
    cluster_keywords = {}
    
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:  # Skip noise
            continue
        
        cluster_queries = df[df['cluster'] == cluster_id]
        
        # Prepare text based on mode
        cluster_texts = []
        for _, row in cluster_queries.iterrows():
            if mode == 'sql':
                text = row.get('query_sql', '')
            else:
                parts = []
                if row.get('name'):
                    parts.append(row['name'])
                if row.get('description'):
                    parts.append(row['description'])
                if row.get('tags'):
                    parts.extend(row['tags'])
                text = ' '.join(parts)
            
            if text.strip():
                cluster_texts.append(text)
        
        # TF-IDF to find important terms
        if cluster_texts:
            vectorizer = TfidfVectorizer(max_features=top_n, stop_words='english')
            try:
                vectorizer.fit_transform(cluster_texts)
                keywords = vectorizer.get_feature_names_out().tolist()
                cluster_keywords[str(cluster_id)] = keywords
            except:
                cluster_keywords[str(cluster_id)] = []
    
    return cluster_keywords


def save_clusters(
    query_objects: List[Dict],
    cluster_labels: np.ndarray,
    output_dir: Path,
    mode: str
):
    """Save clustering results in multiple formats."""
    print(f"[INFO] Saving clustering results...")
    
    try:
        # Create DataFrame
        df = pd.DataFrame(query_objects)
        df['cluster'] = cluster_labels
        
        # Save as parquet
        parquet_path = output_dir / 'clustered_queries.parquet'
        df.to_parquet(parquet_path, compression='zstd')
        print(f"[INFO] ✅ Saved to {parquet_path}")
        
        # Save cluster statistics
        cluster_stats = analyze_clusters(query_objects, cluster_labels)
        stats_path = output_dir / 'cluster_statistics.csv'
        cluster_stats.to_csv(stats_path, index=False)
        print(f"[INFO] ✅ Saved statistics to {stats_path}")
        
        # Save keywords
        keywords = extract_cluster_keywords(query_objects, cluster_labels, mode)
        keywords_path = output_dir / 'cluster_keywords.json'
        with open(keywords_path, 'w') as f:
            json.dump(keywords, f, indent=2)
        print(f"[INFO] ✅ Saved keywords to {keywords_path}")
        
        # Save individual cluster files
        clusters_dir = output_dir / 'individual_clusters'
        clusters_dir.mkdir(exist_ok=True)
        for cluster_id in sorted(set(cluster_labels)):
            cluster_df = df[df['cluster'] == cluster_id]
            cluster_file = clusters_dir / f'cluster_{cluster_id}.parquet'
            cluster_df.to_parquet(cluster_file, compression='zstd')
        print(f"[INFO] ✅ Saved individual clusters to {clusters_dir}")
        
    except Exception as e:
        print(f"[ERROR] ❌ Failed to save: {e}")
        import traceback
        traceback.print_exc()


def print_cluster_summary(query_objects: List[Dict], cluster_labels: np.ndarray):
    """Print a summary of clustering results."""
    cluster_stats = analyze_clusters(query_objects, cluster_labels)
    
    print("\n" + "=" * 80)
    print("CLUSTER SUMMARY")
    print("=" * 80)
    
    for _, row in cluster_stats.iterrows():
        cluster_id = row['cluster_id']
        size = row['size']
        
        if cluster_id == -1:
            print(f"\n🔸 NOISE: {size} queries")
        else:
            print(f"\n📊 CLUSTER {cluster_id}: {size} queries")
            print(f"   Tags: {', '.join(row['top_tags'][:5]) if row['top_tags'] else 'None'}")
            print(f"   Samples:")
            for i, name in enumerate(row['sample_names'][:3], 1):
                print(f"     {i}. {name[:60]}...")
    
    print("\n" + "=" * 80)


def main():
    """Main execution: cluster queries."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cluster Dune queries with multi-threading')
    parser.add_argument('--mode', choices=['semantic', 'sql'], default='semantic',
                        help='Clustering mode: semantic (text) or sql (patterns)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (auto-selected based on mode if not provided)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker threads (default: auto)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of queries to process')
    parser.add_argument('--no-parallel', action='store_false', dest='use_parallel',
                        help='Disable parallel text preparation')
    args = parser.parse_args()
    
    # Auto-select model based on mode
    if args.model is None:
        if args.mode == 'sql':
            model_name = 's2593817/sft-sql-embedding'
        else:
            model_name = 'all-MiniLM-L6-v2'
    else:
        model_name = args.model
    
    print("\n" + "=" * 80)
    print(f"DUNE QUERY CLUSTERING - {args.mode.upper()} MODE (MULTI-THREADED)")
    print("=" * 80 + "\n")
    
    # Load queries
    print("[INFO] Loading queries...")
    query_objects = get_query_objects(DATA_DIR, limit=args.limit)
    
    if not query_objects:
        print("[ERROR] No queries found!")
        return
    
    print(f"[INFO] Loaded {len(query_objects):,} queries")
    
    clusterer = QueryClusterer(
        model_name=model_name,
        min_cluster_size=15,
        min_samples=3,
        mode=args.mode,
        max_workers=args.workers
    )

    # Create embeddings with parallel text preparation
    clusterer.create_embeddings(
        query_objects, 
        batch_size=64 if args.mode=='semantic' else 256,
        use_parallel=args.use_parallel
    )

    # Reduce dimensions (uses UMAP's n_jobs parameter)
    clusterer.reduce_dimensions(n_components=5, n_neighbors=15)

    # Cluster (uses HDBSCAN's core_dist_n_jobs parameter)
    cluster_labels = clusterer.cluster()
    
    # Print summary
    print_cluster_summary(query_objects, cluster_labels)
    
    if args.mode == 'sql':
        output_subdir = OUTPUT_DIR_SQL 
    else:
        output_subdir = OUTPUT_DIR_SEMANTIC
    
    save_clusters(query_objects, cluster_labels, output_subdir, args.mode)
    clusterer.save_model(output_subdir / 'clusterer_model.pkl')
    
    print(f"\n✅ Clustering complete! Results saved to {output_subdir}")


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