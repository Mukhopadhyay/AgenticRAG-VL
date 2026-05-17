FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Install uv
RUN pip install --no-cache-dir uv

# Copy project metadata & install deps
COPY pyproject.toml .
RUN uv pip install --system --no-cache .

# Pre downloading the embedding model at build ttime
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', encode_kwargs={'normalize_embeddings': True})"

# Application soruce
COPY src/ ./src/
COPY chainlit.md .

# Mounting data dirs
RUN mkdir -p /app/data/raw /app/data/processed

ENV PYTHONPATH=/app/src

EXPOSE 8000

WORKDIR /app/src

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
