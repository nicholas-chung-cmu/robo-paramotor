# Paramotor PPO training image (JAX + MuJoCo MJX).
#
# The jax[cudaNN] wheels bundle the CUDA runtime, cuDNN and NCCL, so a plain
# Python base is enough. The host only needs an NVIDIA driver and the NVIDIA
# Container Toolkit. See docs/RL.md -> Docker.
#
# Build from the repo root (the context), pointing -f at this file. BuildKit
# picks up docker/Dockerfile.dockerignore automatically.
#
#   docker build -f docker/Dockerfile -t paramotor-rl .                         # CUDA 13 (default)
#   docker build -f docker/Dockerfile -t paramotor-rl --build-arg JAX_CUDA=cuda12 .
#   docker build -f docker/Dockerfile -t paramotor-rl --build-arg JAX_CUDA=cpu .  # no GPU, debugging
FROM python:3.12-slim

ARG JAX_CUDA=cuda13

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Disable unrelated pytest plugins, as docs/RL.md does on the host.
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    # Let JAX grow GPU memory instead of grabbing 75% up front.
    XLA_PYTHON_CLIENT_PREALLOCATE=false

WORKDIR /workspace

# Dependencies first so code edits do not invalidate this layer.
COPY rl/requirements-rl.txt rl/
RUN python -m pip install -r rl/requirements-rl.txt \
 && if [ "$JAX_CUDA" != "cpu" ]; then \
        python -m pip install "jax[${JAX_CUDA}]==$(python -c 'import jax; print(jax.__version__)')"; \
    fi

COPY . .

CMD ["python", "-c", "import jax; print(jax.devices())"]
