"""
Keyword extraction for query clusters.

Extracts representative keywords/phrases for each cluster using:
- TF-IDF: Term frequency-inverse document frequency
- N-grams: Bigrams and trigrams
- SQL features: Dynamic extraction via sqlglot (tables, functions, CTEs)

Helps understand and label clusters automatically.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import Counter
from dataclasses import dataclass, field

# Lazy imports
TfidfVectorizer = None
CountVectorizer = None


def _load_sklearn_vectorizers():
    """Lazy load sklearn vectorizers."""
    global TfidfVectorizer, CountVectorizer
    if TfidfVectorizer is None:
        from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF, CountVectorizer as CV
        TfidfVectorizer = TFIDF
        CountVectorizer = CV
    return TfidfVectorizer, CountVectorizer


# =============================================================================
# Dynamic SQL Feature Extraction (replaces hardcoded patterns)
# =============================================================================

# Import dynamic extractor
try:
    from src.cluster.sql_features import (
        extract_sql_features,
        aggregate_cluster_features,
        infer_cluster_domain,
        infer_cluster_pattern,
        SQLFeatures,
    )
    HAS_DYNAMIC_EXTRACTOR = True
except ImportError:
    HAS_DYNAMIC_EXTRACTOR = False
    print("[WARN] sql_features module not found, using legacy patterns")


def extract_sql_patterns(sql: str) -> Dict[str, int]:
    """
    Extract SQL patterns from a query.
    
    Uses dynamic sqlglot-based extraction when available,
    falls back to regex for simple pattern counting.
    
    Args:
        sql: SQL query string
        
    Returns:
        Dictionary of pattern -> count
    """
    if HAS_DYNAMIC_EXTRACTOR:
        features = extract_sql_features(sql)
        patterns = {}
        
        # Convert features to pattern dict
        for func in features.aggregate_funcs:
            patterns[f'agg_{func}'] = patterns.get(f'agg_{func}', 0) + 1
        
        for func in features.window_funcs:
            patterns[f'window_{func}'] = patterns.get(f'window_{func}', 0) + 1
        
        for join_type in features.join_types:
            patterns[f'join_{join_type}'] = patterns.get(f'join_{join_type}', 0) + 1
        
        if features.has_union:
            patterns['union'] = 1
        if features.has_subquery:
            patterns['subquery'] = features.subquery_count
        if features.has_distinct:
            patterns['distinct'] = 1
        if features.has_group_by:
            patterns['group_by'] = 1
        if features.has_order_by:
            patterns['order_by'] = 1
        if features.has_having:
            patterns['having'] = 1
        if features.has_limit:
            patterns['limit'] = 1
        if features.has_case_when:
            patterns['case_when'] = 1
        if features.cte_count > 0:
            patterns['cte'] = features.cte_count
        
        return patterns
    
    else:
        # Legacy regex fallback
        return _extract_sql_patterns_regex(sql)


def _extract_sql_patterns_regex(sql: str) -> Dict[str, int]:
    """Legacy regex-based pattern extraction."""
    sql_lower = sql.lower()
    patterns = {}
    
    # Basic patterns
    regex_patterns = {
        'sum': r'\bsum\s*\(',
        'count': r'\bcount\s*\(',
        'avg': r'\bavg\s*\(',
        'row_number': r'\brow_number\s*\(',
        'rank': r'\brank\s*\(',
        'left_join': r'\bleft\s+join\b',
        'inner_join': r'\binner\s+join\b',
        'union': r'\bunion\b',
        'subquery': r'\(\s*select\b',
        'cte': r'\bwith\s+\w+\s+as\s*\(',
        'date_trunc': r'\bdate_trunc\s*\(',
        'distinct': r'\bdistinct\b',
        'group_by': r'\bgroup\s+by\b',
        'order_by': r'\border\s+by\b',
    }
    
    for name, regex in regex_patterns.items():
        matches = re.findall(regex, sql_lower)
        if matches:
            patterns[name] = len(matches)
    
    return patterns


def extract_table_patterns(sql: str) -> Dict[str, int]:
    """
    Extract actual tables from a query (dynamic extraction).
    
    Args:
        sql: SQL query string
        
    Returns:
        Dictionary of table -> count
    """
    if HAS_DYNAMIC_EXTRACTOR:
        features = extract_sql_features(sql)
        # Return actual table names found
        return {table: 1 for table in features.tables}
    else:
        # Legacy: extract with regex
        return _extract_tables_regex(sql)


def _extract_tables_regex(sql: str) -> Dict[str, int]:
    """Legacy regex-based table extraction."""
    sql_lower = sql.lower()
    
    # Extract from FROM and JOIN clauses
    pattern = r'(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
    tables = re.findall(pattern, sql_lower)
    
    # Filter out common CTEs
    filtered = [t for t in tables if not t.startswith('query_')]
    
    return Counter(filtered)


# =============================================================================
# Keyword Extraction Classes
# =============================================================================

@dataclass
class ClusterKeywords:
    """Keywords for a single cluster."""
    cluster_id: int
    tfidf_keywords: List[str] = field(default_factory=list)
    ngram_keywords: List[str] = field(default_factory=list)
    sql_patterns: Dict[str, int] = field(default_factory=dict)
    table_patterns: Dict[str, int] = field(default_factory=dict)
    tag_keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'cluster_id': self.cluster_id,
            'tfidf_keywords': self.tfidf_keywords,
            'ngram_keywords': self.ngram_keywords,
            'sql_patterns': self.sql_patterns,
            'table_patterns': self.table_patterns,
            'tag_keywords': self.tag_keywords,
        }


class KeywordExtractor:
    """
    Extract representative keywords for query clusters.
    
    Example:
        extractor = KeywordExtractor()
        keywords = extractor.extract(queries, cluster_labels, mode='sql')
    """
    
    def __init__(
        self,
        max_tfidf_keywords: int = 10,
        max_ngram_keywords: int = 10,
        max_sql_patterns: int = 10,
        ngram_range: Tuple[int, int] = (2, 3),
    ):
        """
        Initialize the extractor.
        
        Args:
            max_tfidf_keywords: Maximum TF-IDF keywords per cluster
            max_ngram_keywords: Maximum n-gram keywords per cluster
            max_sql_patterns: Maximum SQL patterns per cluster
            ngram_range: Range of n-grams to extract (min, max)
        """
        self.max_tfidf_keywords = max_tfidf_keywords
        self.max_ngram_keywords = max_ngram_keywords
        self.max_sql_patterns = max_sql_patterns
        self.ngram_range = ngram_range
    
    def _prepare_text(self, query: Dict, mode: str) -> str:
        """Prepare text based on mode."""
        if mode == 'sql':
            sql = query.get('query_sql', '')
            # Light normalization for keyword extraction
            sql = re.sub(r'--.*?(\r?\n|$)', ' ', sql)
            sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
            return sql.lower()
        else:
            parts = []
            if query.get('name'):
                parts.append(query['name'])
            if query.get('description'):
                parts.append(query['description'])
            if query.get('tags'):
                parts.extend(query['tags'])
            return ' '.join(parts).lower()
    
    def extract(
        self,
        queries: List[Dict],
        cluster_labels: np.ndarray,
        mode: str = 'sql',
    ) -> Dict[int, ClusterKeywords]:
        """
        Extract keywords for all clusters.
        
        Args:
            queries: List of query dictionaries
            cluster_labels: Cluster labels for each query
            mode: 'sql' or 'semantic'
            
        Returns:
            Dictionary mapping cluster_id -> ClusterKeywords
        """
        print(f"[INFO] Extracting keywords for {len(set(cluster_labels))} clusters...")
        
        df = pd.DataFrame(queries)
        df['cluster'] = cluster_labels
        
        cluster_keywords = {}
        
        for cluster_id in sorted(set(cluster_labels)):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_queries = df[df['cluster'] == cluster_id]
            
            # Prepare texts
            texts = [
                self._prepare_text(row.to_dict(), mode)
                for _, row in cluster_queries.iterrows()
            ]
            
            # Extract keywords
            keywords = ClusterKeywords(cluster_id=cluster_id)
            
            # TF-IDF keywords
            keywords.tfidf_keywords = self._extract_tfidf(texts)
            
            # N-gram keywords
            keywords.ngram_keywords = self._extract_ngrams(texts)
            
            # SQL patterns (only for SQL mode)
            if mode == 'sql':
                keywords.sql_patterns = self._extract_sql_patterns(
                    cluster_queries['query_sql'].tolist()
                )
                keywords.table_patterns = self._extract_table_patterns(
                    cluster_queries['query_sql'].tolist()
                )
            
            # Tag keywords (always useful)
            keywords.tag_keywords = self._extract_tag_keywords(
                cluster_queries['tags'].tolist()
            )
            
            cluster_keywords[cluster_id] = keywords
        
        return cluster_keywords
    
    def _extract_tfidf(self, texts: List[str]) -> List[str]:
        """Extract TF-IDF keywords from texts."""
        if not texts:
            return []
        
        TFIDF, _ = _load_sklearn_vectorizers()
        
        try:
            vectorizer = TFIDF(
                max_features=self.max_tfidf_keywords,
                stop_words='english',
                min_df=2,
                max_df=0.9,
            )
            vectorizer.fit_transform(texts)
            return vectorizer.get_feature_names_out().tolist()
        except Exception:
            return []
    
    def _extract_ngrams(self, texts: List[str]) -> List[str]:
        """Extract common n-grams from texts."""
        if not texts:
            return []
        
        _, CV = _load_sklearn_vectorizers()
        
        try:
            vectorizer = CV(
                ngram_range=self.ngram_range,
                max_features=100,
                stop_words='english',
                min_df=2,
            )
            X = vectorizer.fit_transform(texts)
            
            # Get most common n-grams
            sums = np.array(X.sum(axis=0)).flatten()
            indices = sums.argsort()[::-1][:self.max_ngram_keywords]
            feature_names = vectorizer.get_feature_names_out()
            
            return [feature_names[i] for i in indices]
        except Exception:
            return []
    
    def _extract_sql_patterns(self, sqls: List[str]) -> Dict[str, int]:
        """Extract common SQL patterns from queries."""
        combined = Counter()
        
        for sql in sqls:
            if sql:
                patterns = extract_sql_patterns(sql)
                combined.update(patterns)
        
        # Return top patterns
        return dict(combined.most_common(self.max_sql_patterns))
    
    def _extract_table_patterns(self, sqls: List[str]) -> Dict[str, int]:
        """Extract common table patterns from queries."""
        combined = Counter()
        
        for sql in sqls:
            if sql:
                patterns = extract_table_patterns(sql)
                combined.update(patterns)
        
        return dict(combined.most_common(self.max_sql_patterns))
    
    def _extract_tag_keywords(self, tags_list: List[List[str]]) -> List[str]:
        """Extract most common tags."""
        tag_counter = Counter()
        
        for tags in tags_list:
            if tags:
                tag_counter.update(tags)
        
        return [tag for tag, _ in tag_counter.most_common(10)]


# =============================================================================
# Standalone Functions
# =============================================================================

def extract_cluster_keywords(
    queries: List[Dict],
    cluster_labels: np.ndarray,
    mode: str = 'sql',
    top_n: int = 10,
) -> Dict[str, List[str]]:
    """
    Convenience function to extract keywords.
    
    Args:
        queries: List of query dictionaries
        cluster_labels: Cluster labels
        mode: 'sql' or 'semantic'
        top_n: Number of keywords per cluster
        
    Returns:
        Dictionary mapping cluster_id -> keywords list
    """
    extractor = KeywordExtractor(max_tfidf_keywords=top_n)
    keywords = extractor.extract(queries, cluster_labels, mode)
    
    return {
        str(cid): kw.tfidf_keywords
        for cid, kw in keywords.items()
    }


def extract_tfidf_keywords(
    texts: List[str],
    top_n: int = 10,
) -> List[str]:
    """
    Extract TF-IDF keywords from a list of texts.
    
    Args:
        texts: List of text strings
        top_n: Number of keywords to extract
        
    Returns:
        List of keywords
    """
    TFIDF, _ = _load_sklearn_vectorizers()
    
    try:
        vectorizer = TFIDF(
            max_features=top_n,
            stop_words='english',
            min_df=1,
        )
        vectorizer.fit_transform(texts)
        return vectorizer.get_feature_names_out().tolist()
    except Exception:
        return []


def extract_ngram_keywords(
    texts: List[str],
    ngram_range: Tuple[int, int] = (2, 3),
    top_n: int = 10,
) -> List[str]:
    """
    Extract common n-grams from texts.
    
    Args:
        texts: List of text strings
        ngram_range: Range of n-grams (min, max)
        top_n: Number of n-grams to extract
        
    Returns:
        List of n-gram strings
    """
    _, CV = _load_sklearn_vectorizers()
    
    try:
        vectorizer = CV(
            ngram_range=ngram_range,
            max_features=100,
            stop_words='english',
            min_df=1,
        )
        X = vectorizer.fit_transform(texts)
        
        sums = np.array(X.sum(axis=0)).flatten()
        indices = sums.argsort()[::-1][:top_n]
        feature_names = vectorizer.get_feature_names_out()
        
        return [feature_names[i] for i in indices]
    except Exception:
        return []


def save_keywords(
    cluster_keywords: Dict[int, ClusterKeywords],
    output_path: Path,
):
    """Save cluster keywords to JSON."""
    import json
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        str(cid): kw.to_dict()
        for cid, kw in cluster_keywords.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[INFO] Saved keywords to {output_path}")


def print_cluster_keywords(
    cluster_keywords: Dict[int, ClusterKeywords],
    max_clusters: int = 10,
):
    """Print keyword summary for clusters."""
    print("\n" + "=" * 80)
    print("CLUSTER KEYWORDS")
    print("=" * 80)
    
    for i, (cluster_id, kw) in enumerate(sorted(cluster_keywords.items())):
        if i >= max_clusters:
            print(f"\n... and {len(cluster_keywords) - max_clusters} more clusters")
            break
        
        print(f"\n📊 CLUSTER {cluster_id}")
        
        if kw.tfidf_keywords:
            print(f"   TF-IDF: {', '.join(kw.tfidf_keywords[:5])}")
        
        if kw.ngram_keywords:
            print(f"   N-grams: {', '.join(kw.ngram_keywords[:5])}")
        
        if kw.sql_patterns:
            top_patterns = list(kw.sql_patterns.keys())[:5]
            print(f"   SQL Patterns: {', '.join(top_patterns)}")
        
        if kw.table_patterns:
            top_tables = list(kw.table_patterns.keys())[:5]
            print(f"   Tables: {', '.join(top_tables)}")
        
        if kw.tag_keywords:
            print(f"   Tags: {', '.join(kw.tag_keywords[:5])}")
    
    print("\n" + "=" * 80)


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Extract cluster keywords')
    parser.add_argument('--queries', type=str, required=True, help='Input queries JSONL')
    parser.add_argument('--labels', type=str, required=True, help='Cluster labels .npy file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--mode', choices=['sql', 'semantic'], default='sql')
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args()
    
    # Load data
    print(f"[INFO] Loading queries from {args.queries}...")
    with open(args.queries) as f:
        queries = [json.loads(line) for line in f if line.strip()]
    
    print(f"[INFO] Loading labels from {args.labels}...")
    labels = np.load(args.labels)
    
    print(f"[INFO] Loaded {len(queries)} queries, {len(set(labels))} clusters")
    
    # Extract keywords
    extractor = KeywordExtractor(max_tfidf_keywords=args.top_n)
    keywords = extractor.extract(queries, labels, mode=args.mode)
    
    # Print summary
    print_cluster_keywords(keywords)
    
    # Save
    save_keywords(keywords, args.output)
    
    print(f"\n✅ Done! Keywords saved to {args.output}")