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
COPY setup.py README.md ./
COPY .env.example ./.env

# Create necessary directories
RUN mkdir -p /app/data/models /app/data/audit /app/logs /app/reports

# Configuration for Supervisor
RUN echo '[supervisord]\nnodaemon=true\nuser=root\n\n[program:api]\ncommand=uvicorn src.api.app:app --host 0.0.0.0 --port 8000\nautostart=true\nautorestart=true\nstdout_logfile=/app/logs/api.log\nstderr_logfile=/app/logs/api.err.log\n\n[program:dashboard]\ncommand=streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0\nautostart=true\nautorestart=true\nstdout_logfile=/app/logs/dashboard.log\nstderr_logfile=/app/logs/dashboard.err.log' > /etc/supervisor/conf.d/supervisord.conf

# Hugging Face Spaces usually listens on port 7860
# We will expose 7860 and proxy it to Streamlit (8501)
EXPOSE 7860

# Simple script to handle port mapping for HF
RUN echo '#!/bin/bash\n# Start supervisor\n/usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Note: For Hugging Face, you might need to change the Streamlit port to 7860 directly 
# or use a proxy. Let's set Streamlit to 7860 directly for simplicity in HF.
RUN sed -i "s/8501/7860/g" /etc/supervisor/conf.d/supervisord.conf

CMD ["/app/entrypoint.sh"]
