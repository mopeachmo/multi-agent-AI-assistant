# ---- Base Python image ----
FROM python:3.11-slim

# ---- Basics ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- System deps (lean) ----
# ca-certificates: TLS, curl: healthcheck, tini: proper PID 1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tini \
 && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
RUN pip install --no-cache-dir \
    streamlit \
    langchain \
    langgraph \
    langchain-openai \
    langchain-community \
    python-dotenv \
    requests


# ---- App code ----
COPY . /app

# ---- Non-root user ----
RUN useradd -m appuser
USER appuser

# ---- Streamlit runtime config ----
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_BASE=light \
    # App defaults (override with -e in `docker run` as needed)
    BOOKS_PATH=/app/data/books.json

# Required at runtime:
# - OPENAI_API_KEY
# - (optional) WEATHER_API_KEY  …for weather agent

EXPOSE 8501

# ---- Healthcheck ----
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# ---- Entrypoint/CMD ----
# Use tini to handle signals cleanly (Ctrl+C etc.)
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "main.py"]
