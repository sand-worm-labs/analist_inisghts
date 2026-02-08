# Analist Insights

A pipeline for collecting, normalizing, and clustering [Dune Analytics](https://dune.com/) queries using machine learning. It fetches query metadata via the Dune API, then groups similar queries using HDBSCAN clustering with sentence embeddings.

## Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org/)
- Docker + NVIDIA GPU (optional, for containerized clustering)

## Setup

```bash
# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Edit .env and add your Dune API key(s)
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DUNE_API_KEYS` | Comma-separated Dune API keys (required) | — |
| `DATA_PATH` | Directory for downloaded query data | `./dataset` |
| `LOGS_PATH` | Logging output directory | `./logs` |
| `PROGRAM_CURSOR` | Cursor tracking directory | `./cursor` |
| `DEBUG` | Enable debug logging | `False` |

## Usage

### 1. Collect Queries

Fetches query metadata from the Dune API with concurrent requests, retry logic, and rate limiting. Data is saved as Parquet files.

```bash
poetry run python src/collector.py
```

### 2. Cluster Queries

Groups queries using HDBSCAN clustering. Supports two modes:

- **semantic** — clusters by name + description + tags (what queries are about)
- **sql** — clusters by SQL patterns (how queries are structured)

```bash
# Semantic clustering
poetry run python src/cluster_queries.py --mode semantic

# SQL pattern clustering
poetry run python src/cluster_queries.py --mode sql
```

**Clustering config defaults:**

| Parameter | Value |
|---|---|
| Embedding model (semantic) | `nvidia/NV-Embed-v2` |
| Embedding model (sql) | `s2593817/sft-sql-embedding` |
| HDBSCAN min_cluster_size | 15 |
| HDBSCAN min_samples | 3 |
| UMAP components | 5 |

Output includes Parquet, CSV, and JSON files with cluster assignments, quality metrics (silhouette, Davies-Bouldin, Calinski-Harabasz), and per-cluster keyword extraction via TF-IDF.

### 3. Tag Frequency Report

Extracts all tags from query metadata and writes a frequency-sorted report.

```bash
poetry run python scripts/cluster_sql_only.py
```

## Docker (GPU)

For running clustering on a GPU-enabled machine:

```bash
docker compose -f .docker/docker-compose.yml up --build
```

This runs SQL-mode clustering inside an NVIDIA CUDA container with automatic HuggingFace model caching.

## Project Structure

```
src/
  collector.py       # Dune API data collection with concurrency and retries
  cluster_queries.py # HDBSCAN clustering (semantic + SQL modes)
  config.py          # Environment config and validation
  utils.py           # Shared utilities (Parquet I/O, text normalization, logging)
scripts/
  cluster_sql_only.py # Tag frequency extraction script
.docker/
  Dockerfile           # GPU-enabled container image
  docker-compose.yml   # Clustering service orchestration
```
