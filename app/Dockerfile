# Step 3: Model Inference and UI
FROM python:3.9-slim

WORKDIR /app

# Copy files directly
COPY main.py /app/main.py
COPY requirements.txt /app/requirements.txt
COPY static/ /app/static/
COPY model/ /app/model/

# Install base dependencies first - ensure specific versions for compatibility
RUN pip install --no-cache-dir numpy==1.23.5 joblib==1.1.1

# Install PyTorch CPU from specific URL
RUN pip install --no-cache-dir --find-links https://download.pytorch.org/whl/cpu/torch_stable.html torch==2.1.0+cpu torchvision==0.16.0+cpu

# Install transformers and safetensors explicitly
RUN pip install --no-cache-dir transformers==4.35.0 safetensors==0.4.0

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for PyTorch
ENV CUDA_VISIBLE_DEVICES="" TORCH_HOME="/tmp/torch" OMP_NUM_THREADS="1"
ENV TRANSFORMERS_CACHE="/app/model"

# Create startup script with debugging
RUN echo '#!/bin/sh' > /app/start.sh && \
    echo 'echo "Starting FastAPI application with PyTorch and Transformers..."' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Check installed packages' >> /app/start.sh && \
    echo 'echo "Installed packages:"' >> /app/start.sh && \
    echo 'pip list | grep -E "starlette|fastapi|numpy|joblib|scikit-learn|torch|transformers"' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Test imports' >> /app/start.sh && \
    echo 'echo "Testing imports..."' >> /app/start.sh && \
    echo 'python -c "import numpy; import joblib; from sklearn.preprocessing import LabelEncoder; import torch; from transformers import AutoTokenizer, AutoModelForSequenceClassification; print(\"PyTorch version:\", torch.__version__, \", CUDA available:\", torch.cuda.is_available(), \", Transformers available\")"' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Start the application' >> /app/start.sh && \
    echo 'echo "Starting gunicorn server..."' >> /app/start.sh && \
    echo 'exec gunicorn \\' >> /app/start.sh && \
    echo '  --workers 1 \\' >> /app/start.sh && \
    echo '  --threads 2 \\' >> /app/start.sh && \
    echo '  --timeout 300 \\' >> /app/start.sh && \
    echo '  --bind 0.0.0.0:8080 \\' >> /app/start.sh && \
    echo '  -k uvicorn.workers.UvicornWorker \\' >> /app/start.sh && \
    echo '  main:app' >> /app/start.sh && \
    chmod +x /app/start.sh

# Expose port
EXPOSE 8080

# Run app
CMD ["/app/start.sh"] 