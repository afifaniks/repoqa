FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    wget \
    git \
    gcc \
    curl \
    gfortran \
    python3-pkgconfig \
    libopenblas-dev \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* --verbose


RUN curl -fsSL https://ollama.com/install.sh | bash


WORKDIR /app


COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install the package in editable mode
RUN pip install -e .

# Create directory for Ollama data
RUN mkdir -p /root/.ollama

# Define volume for Ollama models
VOLUME ["/root/.ollama"]

# Expose API port
EXPOSE 8000

ENV OLLAMA_CONTEXT_LENGTH=16000

# Start services
CMD echo "Starting Ollama server..." && \
    ollama serve & \
    echo "Waiting for Ollama to be ready..." && \
    until curl -s http://localhost:11434/api/tags ; do \
        echo "Ollama not ready yet, waiting..." && \
        sleep 1; \
    done && \
    echo "Ollama is ready!" && \
    echo "Pulling qwen3:1.7b model..." && \
    ollama pull qwen3:1.7b && \
    echo "Starting RepoQA API server..." && \
    exec python -m uvicorn repoqa.api:app --host 0.0.0.0 --port 8000
