# ============================================================
#  Stage 1: Builder — cài dependencies với uv
# ============================================================
FROM python:3.11-slim AS builder

# Cài uv (package manager nhanh hơn pip)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy file khai báo dependencies trước (tận dụng Docker layer cache)
COPY pyproject.toml uv.lock ./

# Cài tất cả dependencies vào venv riêng
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python --no-cache -e ".[dev]"


# ============================================================
#  Stage 2: Runtime — image gọn nhẹ cho production
# ============================================================
FROM python:3.11-slim AS runtime

# Tạo user không phải root với home dir để tăng bảo mật
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

WORKDIR /app

# Sao chép venv đã build từ stage builder
COPY --from=builder /app/.venv /app/.venv

# Sao chép toàn bộ source code
COPY --chown=appuser:appuser . .

# Pre-download NLTK data trong lúc build (tránh permission error lúc runtime)
RUN /app/.venv/bin/python -m nltk.downloader -d /home/appuser/nltk_data punkt punkt_tab

# Đảm bảo Python dùng venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Trỏ các host tới service name trong docker-compose
    QDRANT_HOST=qdrant \
    NEO4J_URI=bolt://neo4j:7687

# Cổng FastAPI
EXPOSE 8000

# Chuyển sang user an toàn
USER appuser

# Healthcheck để docker-compose biết app đã sẵn sàng
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# Khởi động FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
