# ==============================================================================
# SPAR Training — Docker Image
# ==============================================================================
# CUDA 12.1 runtime with Python 3.11 for GPU training on AWS Batch.
# Contains all project dependencies and source code.
#
# Build:
#   docker build -t tmoe-training .
#
# Run locally:
#   docker run --gpus all tmoe-training --mode container --config gptneo_125m_lora
# ==============================================================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# --- Set working directory ---
WORKDIR /app

# --- Install Python dependencies (layer cached) ---
COPY requirements.txt /app/requirements.txt
COPY infra/data_ingestion/requirements.txt /app/infra_requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir -r infra_requirements.txt

# --- Copy project source ---
COPY . /app

# --- Default: run training pipeline in container mode ---
ENTRYPOINT ["python", "run_aws_training.py", "--mode", "container"]

# --- Default config (overridden by Batch job command) ---
CMD ["--config", "gptneo_125m_lora"]
