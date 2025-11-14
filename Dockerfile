FROM python:3.11-slim
LABEL maintainer="Ankit Kumar"

# Install system packages needed for numeric libs and builds
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential git curl libsndfile1 gcc libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

ENV APP_HOME=/app
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR $APP_HOME

# Copy requirements and install CPU-only PyTorch then other Python deps
COPY app/requirements.txt $APP_HOME/app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir torch==2.2.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
 && pip install --no-cache-dir -r app/requirements.txt

# Copy application code
COPY app $APP_HOME/app

# Prepare folders and switch to non-root user
RUN mkdir -p /models /tmp/data && chown -R appuser:appuser /models /tmp/data
USER appuser

# Runtime config
EXPOSE 8000
ENV PYTHONUNBUFFERED=1 \
    HUGGINGFACE_HUB_CACHE="/models" \
    EMBEDDING_DEVICE="cpu" \
    APP_PORT=8000

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
