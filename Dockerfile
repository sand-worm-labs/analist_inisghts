# CUDA base for GPU support
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

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

# Install Poetry
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.in-project true

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Copy source code
COPY src/ ./src/

# Install project
RUN poetry install --no-interaction --no-ansi

# Default command
CMD ["poetry", "run", "python", "src/cluster_queries.py"]