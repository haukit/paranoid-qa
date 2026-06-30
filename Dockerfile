# ---- builder: resolve deps into a venv ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# deps first, so this layer caches unless the lockfile changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra serve

# ---- runtime: slim image with venv + source + prebuilt index ----
FROM python:3.12-slim
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app

COPY --from=builder /app/.venv /app/.venv
COPY paranoid_qa ./paranoid_qa
COPY .storage ./.storage

RUN useradd -m -u 1000 app && chown -R app /app
USER app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn paranoid_qa.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
