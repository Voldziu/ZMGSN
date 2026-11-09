FROM nvidia/cuda:13.0.2-base-ubuntu24.04

# Host compiler for nvcc / Triton JIT; the CUDA toolkit itself comes from torch's pip wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV UV_PYTHON=3.12 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --only-group llm --no-install-project
ENV CUDA_HOME=/app/.venv/lib/python3.12/site-packages/nvidia/cu13
ENV PATH=/app/.venv/bin:$CUDA_HOME/bin:$PATH

COPY src/tool_eval/server/templates /app/templates
COPY src/tool_eval/server/presets/cuda /app/presets/cuda

EXPOSE 8000
ENTRYPOINT ["vllm", "serve", "--host", "0.0.0.0", "--port", "8000"]
