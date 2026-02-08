"""
Query embedding generation module.

Generates embeddings for Dune queries using sentence transformers.
Supports two modes:
- semantic: Embed name + description + tags (WHAT the query is about)
- sql: Embed normalized SQL (HOW the query works)

Uses multi-threading for efficient batch processing.
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from dataclasses import dataclass, field
from tqdm import tqdm

from src.utils import (
    clean_sql,
    normalize_sql_for_embedding,
    prepare_semantic_text, 
    prepare_sql_text 
)

# Lazy import for sentence_transformers (heavy dependency)
SentenceTransformer = None


def _load_sentence_transformer():
    """Lazy load sentence transformers."""
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
    return SentenceTransformer


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = 'all-MiniLM-L6-v2'
    batch_size: int = 64
    normalize: bool = True
    max_seq_length: Optional[int] = None


class QueryEmbedder:
    """
    Generate embeddings for Dune queries.
    
    Supports two modes:
    - 'semantic': Embeds name + description + tags
    - 'sql': Embeds normalized SQL query
    
    Example:
        embedder = QueryEmbedder(mode='sql', model_name='s2593817/sft-sql-embedding')
        embeddings = embedder.embed(queries)
    """
    
    # Recommended models for each mode
    DEFAULT_MODELS = {
        'semantic': 'nvidia/NV-Embed-v2',
        'sql': 's2593817/sft-sql-embedding',
    }
    
    def __init__(
        self,
        mode: Literal['semantic', 'sql'] = 'semantic',
        model_name: Optional[str] = None,
        max_workers: Optional[int] = None,
        config: Optional[EmbeddingConfig] = None,
    ):
        """
        Initialize the embedder.
        
        Args:
            mode: 'semantic' (text fields) or 'sql' (query_sql only)
            model_name: Sentence transformer model (auto-selected if None)
            max_workers: Number of threads for parallel processing
            config: Additional configuration options
        """
        self.mode = mode.lower()
        if self.mode not in ['semantic', 'sql']:
            raise ValueError(f"mode must be 'semantic' or 'sql', got: {mode}")
        
        # Auto-select model based on mode
        if model_name is None:
            model_name = self.DEFAULT_MODELS[self.mode]
        
        self.model_name = model_name
        self.config = config or EmbeddingConfig(model_name=model_name)
        
        # Set workers
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 16)
        self.max_workers = max_workers
        
        # Model loaded lazily
        self._model = None
        self.embeddings = None
        
        print(f"[INFO] QueryEmbedder initialized")
        print(f"[INFO] Mode: {self.mode}")
        print(f"[INFO] Model: {self.model_name}")
        print(f"[INFO] Workers: {self.max_workers}")
    
    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            print(f"[INFO] Loading model: {self.model_name}")
            ST = _load_sentence_transformer()
            self._model = ST(self.model_name, trust_remote_code=True)
            
            if self.config.max_seq_length:
                self._model.max_seq_length = self.config.max_seq_length
        
        return self._model
    
    def prepare_text(self, query: Dict) -> str:
        """
        Prepare text based on mode.
        
        Args:
            query: Query dictionary
            
        Returns:
            Text string for embedding
        """
        if self.mode == 'sql':
            return prepare_sql_text(query)
        else:
            return prepare_semantic_text(query)
    
    def prepare_texts_batch(
        self,
        queries: List[Dict],
        start_idx: int,
        end_idx: int
    ) -> List[str]:
        """Prepare texts for a batch (for parallel processing)."""
        return [self.prepare_text(q) for q in queries[start_idx:end_idx]]
    
    def _prepare_texts_parallel(self, queries: List[Dict]) -> List[str]:
        """Prepare texts in parallel using ThreadPoolExecutor."""
        print(f"[INFO] Using {self.max_workers} workers for parallel text preparation...")
        
        # Split into chunks
        chunk_size = max(100, len(queries) // (self.max_workers * 4))
        chunks = []
        for i in range(0, len(queries), chunk_size):
            chunks.append((i, min(i + chunk_size, len(queries))))
        
        texts = [''] * len(queries)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.prepare_texts_batch, queries, start, end): (start, end)
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
    
    def embed(
        self,
        queries: List[Dict],
        batch_size: Optional[int] = None,
        use_parallel: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for all queries.
        
        Args:
            queries: List of query dictionaries
            batch_size: Batch size for encoding (auto-selected if None)
            use_parallel: Use parallel processing for text preparation
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings (n_queries, embedding_dim)
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        n_queries = len(queries)
        print(f"[INFO] Preparing text for {n_queries:,} queries (mode={self.mode})...")
        
        # Parallel text preparation for large datasets
        if use_parallel and n_queries > 10000:
            texts = self._prepare_texts_parallel(queries)
        else:
            texts = [self.prepare_text(q) for q in tqdm(queries, desc="Preparing text", disable=not show_progress)]
        
        # Check for empty texts
        empty_count = sum(1 for t in texts if not t.strip())
        if empty_count > 0:
            print(f"[WARN] {empty_count} queries have empty text in {self.mode} mode")
        
        print(f"[INFO] Generating embeddings (batch_size={batch_size})...")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        )
        
        self.embeddings = embeddings
        print(f"[INFO] Created embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    def save_embeddings(self, filepath: Path):
        """Save embeddings to disk."""
        if self.embeddings is None:
            raise ValueError("No embeddings to save. Call embed() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(filepath, self.embeddings)
        print(f"[INFO] Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: Path) -> np.ndarray:
        """Load embeddings from disk."""
        filepath = Path(filepath)
        self.embeddings = np.load(filepath)
        print(f"[INFO] Loaded embeddings from {filepath}: {self.embeddings.shape}")
        return self.embeddings


# =============================================================================
# Standalone Functions
# =============================================================================

def embed_queries(
    queries: List[Dict],
    mode: Literal['semantic', 'sql'] = 'semantic',
    model_name: Optional[str] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Convenience function to embed queries.
    
    Args:
        queries: List of query dictionaries
        mode: 'semantic' or 'sql'
        model_name: Model to use (auto-selected if None)
        batch_size: Batch size for encoding
        
    Returns:
        Numpy array of embeddings
    """
    embedder = QueryEmbedder(mode=mode, model_name=model_name)
    return embedder.embed(queries, batch_size=batch_size)


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Generate query embeddings')
    parser.add_argument('--mode', choices=['semantic', 'sql'], default='semantic')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output .npy file')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    
    # Load queries
    print(f"[INFO] Loading queries from {args.input}...")
    with open(args.input) as f:
        queries = [json.loads(line) for line in f if line.strip()]
    
    if args.limit:
        queries = queries[:args.limit]
    
    print(f"[INFO] Loaded {len(queries)} queries")
    
    # Generate embeddings
    embedder = QueryEmbedder(mode=args.mode, model_name=args.model)
    embeddings = embedder.embed(queries, batch_size=args.batch_size)
    
    # Save
    embedder.save_embeddings(args.output)
    print(f"[INFO] Done!")