FROM python:3.11-slim AS base

# System deps needed to build scientific Python wheels (pymc/pytensor/scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 mmm \
    && chown -R mmm:mmm /app
USER mmm

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Ships with pre-generated sample data + model results (data/raw/) so the
# dashboard is populated immediately. Re-run the pipeline inside the
# container (e.g. `docker exec ... python run.py --no-insights --geo`) to
# regenerate against fresh config or data.
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
