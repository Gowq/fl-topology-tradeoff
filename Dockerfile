ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/workspace/experiments/shared/code \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash ca-certificates curl git build-essential libgomp1 unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-runtime.txt \
    && python - <<'PY'
import importlib
import torch

required = ["torchvision", "opacus", "numpy", "pandas", "sklearn", "scipy", "matplotlib", "tqdm"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing required Python packages: {missing}")
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

COPY . .
RUN chmod +x scripts/*.sh scripts/grid/*.sh scripts/pegasus/*.sh 2>/dev/null || true

ENTRYPOINT ["bash", "scripts/entrypoint.sh"]
CMD ["list"]
