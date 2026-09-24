# Multi-stage Dockerfile for Executive Communication Coach
FROM python:3.11-slim

WORKDIR /app

# Install system audio and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libasound2-dev \
    libsndfile1 \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging and source files
COPY pyproject.toml setup.py README.md requirements.txt /app/
COPY core/ /app/core/
COPY emulator/ /app/emulator/

# Install the package and dependencies
RUN pip install --no-cache-dir -e .

EXPOSE 8080

ENTRYPOINT ["comm-coach-server", "8080"]
