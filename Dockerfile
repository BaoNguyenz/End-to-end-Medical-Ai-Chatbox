# ============================================================
#  Stage 1: Builder - cai dependencies voi uv
# ============================================================
FROM python:3.11-slim AS builder

# Cai uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy khai bao dependencies
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY README.md ./

# Tao venv va cai dat toan bo package
RUN uv venv /app/.venv && \
    VIRTUAL_ENV=/app/.venv uv pip install --no-cache .

# Pre-download NLTK data ngay trong stage builder
RUN /app/.venv/bin/python -m nltk.downloader -d /root/nltk_data punkt punkt_tab


# ============================================================
#  Stage 2: Runtime - image gon nhe cho production
# ============================================================
FROM python:3.11-slim AS runtime

# Tao user khong phai root voi home dir
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

WORKDIR /app

# Sao chep venv va NLTK data tu builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /root/nltk_data /home/appuser/nltk_data
RUN chown -R appuser:appuser /home/appuser/nltk_data

# Sao chep toan bo source code
COPY --chown=appuser:appuser . .

# Cau hinh moi truong su dung venv va NLTK data
ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv" \
    NLTK_DATA="/home/appuser/nltk_data" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QDRANT_HOST=qdrant \
    NEO4J_URI=bolt://neo4j:7687 \
    REDIS_HOST=redis \
    REDIS_PORT=6379

# Cong FastAPI
EXPOSE 8000

# Chuyen sang user an toan
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# Khoi dong FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
