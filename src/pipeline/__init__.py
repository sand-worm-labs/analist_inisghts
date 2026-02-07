"""
Pipeline modules for orchestrating data processing workflows.

Available pipelines:
- collect: Fetch queries from Dune API
- cluster: Cluster queries by semantic or SQL patterns
- extract: Extract intent features from queries
"""

from src.pipeline.collect import run_collection
from src.pipeline.cluster import run_clustering
from src.pipeline.extract import run_extraction

__all__ = [
    "run_collection",
    "run_clustering", 
    "run_extraction",
]