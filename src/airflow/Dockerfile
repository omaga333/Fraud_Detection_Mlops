FROM apache/airflow:3.3.1-python3.10

USER root
RUN mkdir -p /app/models && chown -R airflow:root /app/models

USER airflow
COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt