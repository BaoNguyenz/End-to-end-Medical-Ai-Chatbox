# ============================================================
#  Stage 1: Builder - cai dependencies voi uv
# ============================================================
FROM python:3.11-slim AS builder

# Cai uv (package manager nhanh hon pip)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy file khai bao dependencies truoc (tan dung Docker layer cache)
COPY pyproject.toml uv.lock ./

# Cai tat ca dependencies vao venv rieng
# --relocatable: dam bao symlinks trong venv hoat dong sau khi copy sang stage khac
RUN uv venv /app/.venv && \
    VIRTUAL_ENV=/app/.venv uv pip install --no-cache -e ".[dev]"


# ============================================================
#  Stage 2: Runtime - image gon nhe cho production
# ============================================================
FROM python:3.11-slim AS runtime

# Tao user khong phai root voi home dir de tang bao mat
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

WORKDIR /app

# Sao chep venv da build tu stage builder
COPY --from=builder /app/.venv /app/.venv

# Sao chep toan bo source code
COPY --chown=appuser:appuser . .

# ── QUAN TRONG: Set PATH truoc khi dung python tu venv ───────────────────────
# uv venv tao symlinks tuong doi (/app/.venv/bin/python -> python3.11)
# Ca builder va runtime deu dung python:3.11-slim nen symlinks van hop le.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv \
    # Tro cac host toi service name trong docker-compose
    QDRANT_HOST=qdrant \
    NEO4J_URI=bolt://neo4j:7687

# Pre-download NLTK data trong luc build (tranh permission error luc runtime)
# Dung "python" sau khi PATH da duoc set -> tim thay /app/.venv/bin/python
RUN python -m nltk.downloader -d /home/appuser/nltk_data punkt punkt_tab && \
    chown -R appuser:appuser /home/appuser/nltk_data

# Cong FastAPI
EXPOSE 8000

# Chuyen sang user an toan
USER appuser

# Healthcheck de docker-compose biet app da san sang
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# Khoi dong FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
