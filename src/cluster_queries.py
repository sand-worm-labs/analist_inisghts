"""
Query clustering using HDBSCAN and sentence embeddings.
Groups similar Dune Analytics queries together for analysis.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import List, Dict
import pickle
import json

from sentence_transformers import SentenceTransformer
import hdbscan
from umap import UMAP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import shared utilities
from src.utils import get_query_objects, normalize_text

DATA_DIR = Path("data")
OUTPUT_DIR = Path("analysis") / "clusters"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class QueryClusterer:
    """Cluster queries using embeddings and HDBSCAN."""
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        min_cluster_size: int = 5,
        min_samples: int = 3
    ):
        """
        Initialize the clusterer.
        
        Args:
            model_name: Sentence transformer model to use
            min_cluster_size: Minimum size for HDBSCAN clusters
            min_samples: Minimum samples for HDBSCAN
        """
        print(f"[INFO] Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.embeddings = None
        self.cluster_labels = None
        self.umap_embeddings = None
        
    def prepare_text(self, query_obj: Dict) -> str:
        """
        Combine query fields into a single text for embedding.
        
        Args:
            query_obj: Query object dictionary
            
        Returns:
            Combined text string
        """
        # Combine name, description, and tags
        parts = []
        
        if query_obj.get('name'):
            parts.append(query_obj['name'])
        
        if query_obj.get('description'):
            parts.append(query_obj['description'])
        
        # Add tags (important for semantic similarity)
        tags = query_obj.get('tags', [])
        if tags:
            parts.append(' '.join(tags))
        
        # Optionally include query SQL (first 500 chars to avoid token limits)
        if query_obj.get('query_sql'):
            sql_snippet = query_obj['query_sql'][:500]
            parts.append(sql_snippet)
        
        return ' '.join(parts)
    
    def create_embeddings(self, query_objects: List[Dict], batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for all queries.
        
        Args:
            query_objects: List of query dictionaries
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        print(f"[INFO] Preparing text for {len(query_objects)} queries...")
        texts = [self.prepare_text(q) for q in query_objects]
        
        print(f"[INFO] Generating embeddings (this may take a while)...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.embeddings = embeddings
        print(f"[INFO] Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
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
            random_state=42
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
        if embeddings is None:
            # Use UMAP embeddings if available, otherwise raw embeddings
            embeddings = self.umap_embeddings if self.umap_embeddings is not None else self.embeddings
        
        print(f"[INFO] Clustering with HDBSCAN...")
        print(f"[INFO] min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        self.cluster_labels = clusterer.fit_predict(embeddings)
        
        # Statistics
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        print(f"\n[INFO] === Clustering Results ===")
        print(f"[INFO] Number of clusters: {n_clusters}")
        print(f"[INFO] Noise points (unclustered): {n_noise}")
        print(f"[INFO] Clustered points: {len(self.cluster_labels) - n_noise}")
        
        return self.cluster_labels
    
    def save_model(self, filepath: Path):
        """Save clusterer state."""
        state = {
            'embeddings': self.embeddings,
            'cluster_labels': self.cluster_labels,
            'umap_embeddings': self.umap_embeddings,
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"[INFO] Saved model state to {filepath}")


def analyze_clusters(query_objects: List[Dict], cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Analyze cluster characteristics.
    
    Args:
        query_objects: List of query dictionaries
        cluster_labels: Cluster labels from HDBSCAN
        
    Returns:
        DataFrame with cluster statistics
    """
    df = pd.DataFrame(query_objects)
    df['cluster'] = cluster_labels
    
    # Get cluster sizes
    cluster_stats = []
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_queries = df[df['cluster'] == cluster_id]
        
        # Get most common tags
        all_tags = []
        for tags in cluster_queries['tags']:
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


def extract_cluster_keywords(query_objects: List[Dict], cluster_labels: np.ndarray, top_n: int = 10):
    """
    Extract representative keywords for each cluster using TF-IDF.
    
    Args:
        query_objects: List of query dictionaries
        cluster_labels: Cluster labels
        top_n: Number of top keywords to extract
        
    Returns:
        Dictionary mapping cluster_id to keywords
    """
    print(f"[INFO] Extracting keywords for each cluster...")
    
    df = pd.DataFrame(query_objects)
    df['cluster'] = cluster_labels
    
    cluster_keywords = {}
    
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:  # Skip noise
            continue
        
        cluster_queries = df[df['cluster'] == cluster_id]
        
        # Combine all text from this cluster
        cluster_texts = []
        for _, row in cluster_queries.iterrows():
            parts = []
            if row.get('name'):
                parts.append(row['name'])
            if row.get('description'):
                parts.append(row['description'])
            if row.get('tags'):
                parts.extend(row['tags'])
            cluster_texts.append(' '.join(parts))
        
        # TF-IDF to find important terms
        if cluster_texts:
            vectorizer = TfidfVectorizer(max_features=top_n, stop_words='english')
            try:
                tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                keywords = vectorizer.get_feature_names_out().tolist()
                cluster_keywords[cluster_id] = keywords
            except:
                cluster_keywords[cluster_id] = []
    
    return cluster_keywords


def save_clusters(
    query_objects: List[Dict],
    cluster_labels: np.ndarray,
    output_dir: Path = OUTPUT_DIR
):
    """
    Save clustering results in multiple formats.
    
    Args:
        query_objects: List of query dictionaries
        cluster_labels: Cluster labels
        output_dir: Directory to save results
    """
    print(f"[INFO] Saving clustering results...")
    
    # Create DataFrame
    df = pd.DataFrame(query_objects)
    df['cluster'] = cluster_labels
    
    # Save as parquet
    df.to_parquet(output_dir / 'clustered_queries.parquet', compression='zstd')
    print(f"[INFO] Saved clustered queries to {output_dir / 'clustered_queries.parquet'}")
    
    # Save cluster statistics
    cluster_stats = analyze_clusters(query_objects, cluster_labels)
    cluster_stats.to_csv(output_dir / 'cluster_statistics.csv', index=False)
    print(f"[INFO] Saved cluster statistics to {output_dir / 'cluster_statistics.csv'}")
    
    # Save keywords
    keywords = extract_cluster_keywords(query_objects, cluster_labels)
    with open(output_dir / 'cluster_keywords.json', 'w') as f:
        json.dump(keywords, f, indent=2)
    print(f"[INFO] Saved cluster keywords to {output_dir / 'cluster_keywords.json'}")
    
    # Save individual cluster files
    clusters_dir = output_dir / 'individual_clusters'
    clusters_dir.mkdir(exist_ok=True)
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_df = df[df['cluster'] == cluster_id]
        cluster_df.to_parquet(
            clusters_dir / f'cluster_{cluster_id}.parquet',
            compression='zstd'
        )
    
    print(f"[INFO] Saved individual cluster files to {clusters_dir}")


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
            print(f"\n🔸 NOISE (Unclustered): {size} queries")
        else:
            print(f"\n📊 CLUSTER {cluster_id}: {size} queries")
            print(f"   Top Tags: {', '.join(row['top_tags'][:5]) if row['top_tags'] else 'None'}")
            print(f"   Sample Queries:")
            for i, name in enumerate(row['sample_names'][:3], 1):
                print(f"     {i}. {name[:60]}...")
    
    print("\n" + "=" * 80)


def main():
    """Main execution: cluster queries."""
    print("\n" + "=" * 80)
    print("DUNE QUERY CLUSTERING")
    print("=" * 80 + "\n")
    
    # Load queries
    print("[INFO] Loading queries...")
    query_objects = get_query_objects(DATA_DIR, limit=None)
    
    if not query_objects:
        print("[ERROR] No queries found!")
        return
    
    print(f"[INFO] Loaded {len(query_objects)} queries")
    
    # Initialize clusterer
    clusterer = QueryClusterer(
        model_name='all-MiniLM-L6-v2',  # Fast and efficient
        min_cluster_size=5,  # Adjust based on your data
        min_samples=3
    )
    
    # Create embeddings
    clusterer.create_embeddings(query_objects, batch_size=32)
    
    # Reduce dimensions (optional but recommended for large datasets)
    clusterer.reduce_dimensions(n_components=5, n_neighbors=15)
    
    # Cluster
    cluster_labels = clusterer.cluster()
    
    # Print summary
    print_cluster_summary(query_objects, cluster_labels)
    
    # Save results
    save_clusters(query_objects, cluster_labels, OUTPUT_DIR)
    
    # Save model
    clusterer.save_model(OUTPUT_DIR / 'clusterer_model.pkl')
    
    print(f"\n✅ Clustering complete! Results saved to {OUTPUT_DIR}")


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