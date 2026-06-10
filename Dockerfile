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

# PyTorch build channel. Default CPU (universal, smaller, runs the figures path
# and CPU experiments anywhere). For GPU reproduction build with:
#   docker build --build-arg TORCH_CHANNEL=cu121 -t fl-topology-tradeoff:gpu .
# and run the container with `--gpus all` (device auto-detects CUDA).
ARG TORCH_CHANNEL=cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision \
         --index-url https://download.pytorch.org/whl/${TORCH_CHANNEL} \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x scripts/*.sh

# Entrypoint prepares data (when reproducing) then dispatches to run.sh.
# Default target rebuilds the figures from the shipped results.
ENTRYPOINT ["bash", "scripts/entrypoint.sh"]
CMD ["figures"]
