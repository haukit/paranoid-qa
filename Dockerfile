# ---- builder: resolve deps into a venv ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# deps first, so this layer caches unless the lockfile changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra serve

# prebuilt index artifact from a GitHub Release
ARG INDEX_RELEASE=ntsb-demo-v1
ADD --checksum=sha256:adb4f000b9047ed0893595316ea4533efc346b8cd69dc91eed75491ee6f4e2a5 \
    https://github.com/haukit/paranoid-qa/releases/download/${INDEX_RELEASE}/ntsb-demo-index.tar.gz \
    /tmp/index.tar.gz
RUN mkdir -p /artifacts && tar -xzf /tmp/index.tar.gz -C /artifacts && rm /tmp/index.tar.gz

# ---- runtime: slim image with venv + source + prebuilt index ----
FROM python:3.12-slim
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app

RUN useradd -m -u 1000 app

COPY --chown=app:app --from=builder /app/.venv /app/.venv
COPY --chown=app:app paranoid_qa ./paranoid_qa
COPY --chown=app:app --from=builder /artifacts/.storage ./.storage
COPY --chown=app:app --from=builder /artifacts/.lightrag ./.lightrag

USER app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn paranoid_qa.serving.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
