FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run requires port 8080. Run schema then start server (schema also runs in lifespan as backup).
CMD ["sh", "-c", "python -c \"import db; db.ensure_schema()\" && exec uvicorn routes:app --host 0.0.0.0 --port 8080"]
