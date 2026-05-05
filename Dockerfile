# ===========================================================================
# VIDYUT - Unified Production Container (API + Dashboard)
# ===========================================================================

FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt requirements-ui.txt setup.py README.md ./

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-ui.txt

COPY src/ ./src/
RUN pip install --no-deps -e .

# ===========================================================================
FROM python:3.10-slim AS runtime

# Install supervisor to run multiple processes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl supervisor \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    APP_ENV=production

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app

COPY src/ ./src/
COPY data/ ./data/
COPY setup.py README.md ./

COPY .env.example ./.env

# Create necessary directories
RUN mkdir -p /app/data/models /app/data/audit /app/logs /app/reports

# Configuration for Supervisor (API only)
RUN echo '[supervisord]\nnodaemon=true\nuser=root\n\n[program:api]\ncommand=uvicorn src.api.app:app --host 0.0.0.0 --port 8000\nautostart=true\nautorestart=true\nstdout_logfile=/app/logs/api.log\nstderr_logfile=/app/logs/api.err.log' > /etc/supervisor/conf.d/supervisord.conf

# Entrypoint script: Start API, wait for health, then start Dashboard
RUN echo '#!/bin/bash\n\n# Start API via supervisor\n/usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf &\n\n# Wait for API to be ready\necho "Waiting for Vidyut API to initialize models..."\nfor i in {1..60}; do\n  if curl -s http://127.0.0.1:8000/healthz > /dev/null; then\n    echo "API is UP and models are loaded!"\n    break\n  fi\n  echo "Initializing... ($i/60)"\n  sleep 2\ndone\n\n# Start Dashboard\necho "Starting Vidyut Dashboard..."\nstreamlit run src/dashboard/app.py --server.port 7860 --server.address 0.0.0.0' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh


# Hugging Face Spaces usually listens on port 7860
EXPOSE 7860

CMD ["/app/entrypoint.sh"]

