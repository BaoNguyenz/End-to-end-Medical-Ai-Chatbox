# ============================================================
#  Stage 1: Builder - cai dependencies voi uv (Python 3.13)
# ============================================================
FROM python:3.13-slim AS builder

# Cai uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy khai bao dependencies
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY README.md ./

# Cai dat truc tiep vao system site-packages cua Python 3.13
RUN uv pip install --system --no-cache .

# Pre-download NLTK data ngay trong stage builder
RUN python -m nltk.downloader -d /root/nltk_data punkt punkt_tab


# ============================================================
#  Stage 2: Runtime - image gon nhe cho production (Python 3.13)
# ============================================================
FROM python:3.13-slim AS runtime

# Tao user khong phai root voi home dir
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

WORKDIR /app

# Sao chep Python packages va binaries tu stage builder (Python 3.13)
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/nltk_data /home/appuser/nltk_data
RUN chown -R appuser:appuser /home/appuser/nltk_data

# Sao chep toan bo source code
COPY --chown=appuser:appuser . .

# Cau hinh moi truong
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA="/home/appuser/nltk_data" \
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

# Khoi dong FastAPI server bang python module syntax
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
