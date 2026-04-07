FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow + networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY explorer/ ./explorer/

# NOTE: This container is a runnable CLI, not a long-running server.
# The backend calls `python -m explorer.cli ...` as a subprocess.
# iOS simulator / AXe CLI / Metro live OUTSIDE the container on the host.
# This container is used for headless modes (vision, web, android later)
# and for the backend to orchestrate subprocess runs.
ENTRYPOINT ["python", "-m", "explorer.cli"]
