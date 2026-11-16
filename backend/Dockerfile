# SentinelFL Backend Dockerfile
# Production-ready Python 3.11 container with PyTorch and dependencies

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# curl for health checks (all Python deps ship as wheels; no compiler needed)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt /app/requirements.txt
# CPU-only PyTorch first: the default Linux wheel bundles CUDA (several GB), which free
# hosts cannot build or run. requirements.txt then sees torch as already satisfied.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application and ML/FL code
COPY backend /app/backend
COPY federated /app/federated
COPY ml /app/ml
COPY experiments /app/experiments

# Hosts such as Render set $PORT; 8000 locally. One worker by default keeps memory
# within small instances (override with WEB_CONCURRENCY).
ENV PORT=8000
ENV WEB_CONCURRENCY=1
EXPOSE 8000

WORKDIR /app/backend

# Healthcheck definition
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch uvicorn production server
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY}"]
