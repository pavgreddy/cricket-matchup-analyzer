# Lightweight backend image -- FastAPI + pandas + scikit-learn, no ML
# embedding model, so this stays well within Render's 512MB free tier.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across rebuilds
# unless requirements actually change.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Only copy what the backend actually needs at runtime -- the raw
# 295K-row deliveries CSVs used for training are NOT needed here and
# are excluded via .dockerignore to keep the image small.
COPY backend/ backend/
COPY data/processed/batter_matchup_stats.csv data/processed/batter_matchup_stats.csv

EXPOSE 8000

# Render injects $PORT at runtime; default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
