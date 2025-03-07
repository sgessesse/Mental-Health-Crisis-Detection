# Dockerfile - Optimized for Memory Efficiency
FROM python:3.9-slim

# Install only essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libtinfo5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install CPU-only PyTorch with optimized settings
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables for memory optimization
ENV CUDA_VISIBLE_DEVICES="" \
    TORCH_HOME="/tmp/torch" \
    OMP_NUM_THREADS="1" \
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32" \
    MALLOC_TRIM_THRESHOLD_="0" \
    MALLOC_MMAP_THRESHOLD_="131072"

# Download only English stopwords to save space
RUN python -m nltk.downloader stopwords -d /usr/share/nltk_data

# Set Python optimization flags
ENV PYTHONOPTIMIZE="1" \
    PYTHONDONTWRITEBYTECODE="1"

EXPOSE 8080

# Run with optimized settings for memory usage
CMD ["gunicorn", "--workers", "1", "--threads", "2", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "120", "main:app", "--bind", "0.0.0.0:8080"]
