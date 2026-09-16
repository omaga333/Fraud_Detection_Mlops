FROM python:3.10.16-slim-bookworm

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

WORKDIR /app

# Install minimal system dependencies
# (confluent-kafka Python wheel already bundles librdkafka with SASL/SSL support)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc \
      python3-dev \
      && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . .

# Run as a non-root user for better security
RUN useradd -m produceruser && chown -R produceruser:produceruser /app
USER produceruser

# Entrypoint
ENTRYPOINT ["python3", "main.py"]