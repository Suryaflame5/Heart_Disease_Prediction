# ── Backend image ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Generate dataset and train model at build time
RUN python datasets/generate_data.py && python models/train.py

EXPOSE 8000

# Run FastAPI with shared module path
CMD ["sh", "-c", "PYTHONPATH=/app uvicorn backend.main:app --host 0.0.0.0 --port 8000"]
