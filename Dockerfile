# Use official Python runtime with CUDA support (optional)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including those needed for PyTorch
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    curl \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p uploads results models cache

# Download pre-trained models during build
RUN mkdir -p /app/models/pretrained

# Copy application code
COPY app.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV MONGODB_URI=mongodb://mongodb:27017/
ENV UPLOAD_DIR=/app/uploads
ENV RESULT_DIR=/app/results
ENV MODEL_CACHE_DIR=/app/models/pretrained
ENV PYTHONWARNINGS=ignore::UserWarning
ENV OMP_NUM_THREADS=4
ENV TORCH_HOME=/app/cache

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Pre-download models during container startup
RUN python -c "import torch; print('PyTorch installed successfully')"

# Run the application
CMD ["python", "app.py"]