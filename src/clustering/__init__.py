"""
Cluster module for Dune query clustering.

Provides tools for:
- Embeddings: Generate embeddings for queries (semantic or SQL)
- Clusterer: HDBSCAN clustering with quality metrics
- Keywords: Extract representative keywords per cluster
- SQL Features: Dynamic SQL pattern extraction

Usage:
    from src.cluster import QueryEmbedder, QueryClusterer, KeywordExtractor
    
    # Generate embeddings
    embedder = QueryEmbedder(mode='sql')
    embeddings = embedder.embed(queries)
    
    # Cluster queries
    clusterer = QueryClusterer(min_cluster_size=50)
    labels = clusterer.fit(embeddings)
    
    # Extract keywords
    extractor = KeywordExtractor()
    keywords = extractor.extract(queries, labels)
"""

from src.cluster.embeddings import (
    QueryEmbedder,
    prepare_semantic_text,
    prepare_sql_text,
)

from src.cluster.clusterer import (
    QueryClusterer,
    ClusterMetrics,
    analyze_clusters,
    print_cluster_summary,
)

from src.cluster.keywords import (
    KeywordExtractor,
    extract_cluster_keywords,
    extract_tfidf_keywords,
    extract_ngram_keywords,
    extract_sql_patterns,
    extract_table_patterns,
)

from src.cluster.sql_features import (
    extract_sql_features,
    aggregate_cluster_features,
    infer_cluster_domain,
    infer_cluster_pattern,
    SQLFeatures,
)

__all__ = [
    # Embeddings
    "QueryEmbedder",
    "prepare_semantic_text",
    "prepare_sql_text",
    # Clusterer
    "QueryClusterer",
    "ClusterMetrics",
    "analyze_clusters",
    "print_cluster_summary",
    # Keywords
    "KeywordExtractor",
    "extract_cluster_keywords",
    "extract_tfidf_keywords",
    "extract_ngram_keywords",
    "extract_sql_patterns",
    "extract_table_patterns",
    # SQL Features
    "extract_sql_features",
    "aggregate_cluster_features",
    "infer_cluster_domain",
    "infer_cluster_pattern",
    "SQLFeatures",
]