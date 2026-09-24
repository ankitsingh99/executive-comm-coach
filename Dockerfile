# Multi-stage hardened Dockerfile for Executive Communication Coach
FROM python:3.11-slim

WORKDIR /app

# Install system audio, networking and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libasound2-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user for security
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -s /bin/bash -m appuser

# Copy packaging specifications and code
COPY pyproject.toml setup.py README.md requirements.txt /app/
COPY core/ /app/core/
COPY emulator/ /app/emulator/

# Install the package and dependencies
RUN pip install --no-cache-dir -e .

# Set permissions for non-root execution
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://127.0.0.1:8080/ || exit 1

ENTRYPOINT ["comm-coach-server", "8080"]
