# CUDA base for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Set pip defaults
ENV PIP_DEFAULT_TIMEOUT=600
ENV PIP_RETRIES=10

# Install PyTorch from PyTorch servers (not PyPI)
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining packages with pip
RUN pip3 install \
    sentence-transformers \
    transformers \
    einops \
    datasets \
    hdbscan \
    umap-learn \
    scikit-learn \
    pandas \
    pyarrow \
    numpy \
    tqdm

# Copy source code
COPY src/ ./src/

CMD ["python", "src/cluster_queries.py"]