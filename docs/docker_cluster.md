# Cluster Docker image

This image is intended for GPU clusters that can run Docker, Enroot, Apptainer
or Singularity. It contains the Python stack needed by the current experiment
suite, including Exp. 07 and Exp. 08.

The default base image is:

```text
pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime
```

That matches the Pegasus smoke environment where Exp. 07 passed on RTX 5080
with `torch 2.11.0+cu130`.

## Build

```bash
docker build -t fl-topology-tradeoff:cluster-cu130 .
```

For a CUDA 12.8 cluster, override the base image:

```bash
docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
  -t fl-topology-tradeoff:cluster-cu128 .
```

## Export

```bash
docker save fl-topology-tradeoff:cluster-cu130 | gzip -c > fl-topology-tradeoff_cluster-cu130.tar.gz
```

## Run with Docker

Mount datasets and outputs from the host. The image does not include `data/`,
experiment `results/` or generated attack `artifacts/`.

```bash
docker run --rm --gpus all --shm-size=8g \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/experiments:/workspace/experiments" \
  fl-topology-tradeoff:cluster-cu130 \
  exp07 --smoke-only --config-index 0 --device cuda
```

Useful targets:

```bash
docker run --rm --gpus all --shm-size=8g \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/experiments:/workspace/experiments" \
  fl-topology-tradeoff:cluster-cu130 \
  exp07 --config-start 0 --config-count 6 --device cuda

docker run --rm --gpus all --shm-size=8g \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/experiments:/workspace/experiments" \
  fl-topology-tradeoff:cluster-cu130 \
  exp08 --config-index 0 --device cuda
```

Set `PREPARE_DATA=1` only if the cluster node is allowed to download public
datasets from UCI:

```bash
docker run --rm --gpus all --shm-size=8g -e PREPARE_DATA=1 \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/experiments:/workspace/experiments" \
  fl-topology-tradeoff:cluster-cu130 list
```

## Apptainer or Singularity

Many HPC clusters do not run Docker directly. Convert from the saved tarball or
pull from a registry according to the local cluster policy. A typical flow is:

```bash
apptainer build fl-topology-tradeoff_cluster-cu130.sif docker-archive://fl-topology-tradeoff_cluster-cu130.tar
apptainer exec --nv \
  --bind "$PWD/data:/workspace/data" \
  --bind "$PWD/experiments:/workspace/experiments" \
  fl-topology-tradeoff_cluster-cu130.sif \
  bash scripts/run.sh exp07 --config-index 0 --device cuda
```
