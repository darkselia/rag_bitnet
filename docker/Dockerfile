FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BITNET_DIR=/opt/bitnet \
    BITNET_SERVER_SCRIPT=run_inference_server.py \
    BITNET_MODEL=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    BITNET_THREADS=6 \
    BITNET_PORT=8080 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app

COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libglib2.0-0 libgl1 netcat-openbsd \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY . .
COPY docker/start_services.sh /usr/local/bin/start_services.sh
RUN chmod +x /usr/local/bin/start_services.sh

RUN mkdir -p /app/data
VOLUME ["/app/data", "/opt/bitnet"]

EXPOSE 8000 8080
CMD ["/usr/local/bin/start_services.sh"]
