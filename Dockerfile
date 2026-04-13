FROM python:3.11-slim

WORKDIR /app

# System dependencies: Docker CLI (for docker exec into sibling containers),
# libfreesasa for surface area calculations
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories expected by the platform
RUN mkdir -p /data/oih/inputs /data/oih/outputs /data/oih/tmp

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
