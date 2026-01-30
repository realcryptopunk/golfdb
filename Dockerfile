FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY *.html ./
COPY model.py ./
COPY MobileNetV2.py ./
COPY mobilenet_v2.pth.tar .
COPY models/ ./models/
COPY cbam_swingnet/ ./cbam_swingnet/
COPY comparison/ ./comparison/

# Expose port
EXPOSE 8080

# Run with gunicorn
CMD ["gunicorn", "--config", "gunicorn_config.py", "app:app"]
