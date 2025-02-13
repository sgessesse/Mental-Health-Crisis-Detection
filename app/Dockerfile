# Dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtinfo5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install CPU-only PyTorch
RUN pip install --no-cache-dir -r requirements.txt

ENV CUDA_VISIBLE_DEVICES="" \
    TORCH_HOME="/tmp/torch"

RUN python -m nltk.downloader stopwords -d /usr/share/nltk_data

EXPOSE 8080

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080"]
