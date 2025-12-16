# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for building/ocr/libs (adjust as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models \
    && wget -O /tmp/models.tar.gz \
        https://github.com/ShuiyuanSJTU/discourse-image-enhancement-service/releases/download/models-zh-cn-v1/models.tar.gz \
    && tar -xzf /tmp/models.tar.gz -C /models \
    && rm /tmp/models.tar.gz

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN cp config.json.template config.json && ln -s /models ./models

EXPOSE 80

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--max-requests", "1000", "--timeout", "120", "--bind", "0.0.0.0:80", "app:app"]
