# Fire Fighter Assistance Drone — production image for Render
FROM python:3.12-slim

# Keep CPU/memory usage low (important on Render's smaller instances)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

# System libraries required by PyTorch (OpenMP) and OpenCV on Debian slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Render injects PORT (defaults to 10000); fall back to 8000 for local docker run
ENV PORT=8000
EXPOSE 8000

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
