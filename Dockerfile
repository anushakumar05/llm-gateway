FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so first boot isn't a 90MB download
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

COPY gateway/ ./gateway/

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=40s \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]