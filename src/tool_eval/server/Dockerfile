FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV UV_PYTHON=3.12 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --only-group llm --no-install-project
ENV PATH=/app/.venv/bin:$PATH
# Precompiled FlashInfer kernels; otherwise Blackwell JIT-compiles them on every cold start.
RUN vllm download-kernels

COPY src/tool_eval/server/templates /app/templates
COPY src/tool_eval/server/presets/cuda /app/presets/cuda

EXPOSE 8000
ENTRYPOINT ["vllm", "serve", "--host", "0.0.0.0", "--port", "8000"]
