FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/workspace/experiments/shared/code

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash git curl build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch by default (smaller image, runs anywhere). For GPU runs, install
# a CUDA build of torch/torchvision instead and use `--gpus all`.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x scripts/*.sh

# Default: prepare data, then regenerate every figure from the shipped results.
CMD ["bash", "-lc", "bash scripts/prepare_data.sh && bash scripts/run.sh figures"]
