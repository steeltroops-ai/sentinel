FROM python:3.11-slim
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY services/tav/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shared lib
COPY kive/ ./kive/

# Service code
COPY services/tav/ ./services/tav/

# Seed the tech timeline DB
RUN python services/tav/detector.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "services.tav.main:app", "--host", "0.0.0.0", "--port", "8000"]
