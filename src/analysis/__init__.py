"""
Analysis module for Dune query dataset.

Provides visualization and statistical analysis for:
- gaps: ID coverage and missing data visualization
- duplicates: Duplicate detection and deduplication stats
- tags: Tag distribution and enrichment analysis

Usage:
    python -m src.analysis.gaps
    python -m src.analysis.duplicates  
    python -m src.analysis.tags
"""

from src.analysis.gaps import analyze_gaps, plot_histogram, plot_scatter, plot_coverage_pie
from src.analysis.duplicates import analyze_duplicates, find_exact_duplicates, normalize_sql
from src.analysis.tags import analyze_tags, extract_implicit_tags

__all__ = [
    # Gaps
    "analyze_gaps",
    "plot_histogram",
    "plot_scatter", 
    "plot_coverage_pie",
    # Duplicates
    "analyze_duplicates",
    "find_exact_duplicates",
    "normalize_sql",
    # Tags
    "analyze_tags",
    "extract_implicit_tags",
]