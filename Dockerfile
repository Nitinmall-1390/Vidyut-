# ===========================================================================
# VIDYUT - Streamlit-Only Container (No FastAPI process needed)
# ===========================================================================

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH="/app"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-ui.txt setup.py README.md ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-ui.txt

COPY src/ ./src/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/

RUN pip install --no-deps -e .

COPY .env.example ./.env

RUN mkdir -p /app/data/models /app/data/audit /app/logs /app/reports

EXPOSE 7860

CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
