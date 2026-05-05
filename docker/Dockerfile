# ===========================================================================
# VIDYUT — main app (dashboard + training)
# ===========================================================================

FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgomp1 curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-ui.txt setup.py README.md ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-ui.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY notebooks/ ./notebooks/
RUN pip install --no-deps -e .

RUN groupadd -g 1000 vidyut && useradd -u 1000 -g vidyut -m vidyut \
    && mkdir -p /app/data /app/logs /app/reports \
    && chown -R vidyut:vidyut /app
USER vidyut

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/dashboard/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]