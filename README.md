# RAG BitNet (FastAPI)

## Запуск локально
```bash
cp .env.example .env
conda create -n rag-bitnet python=3.10 -y
conda activate rag-bitnet
pip install -r requirements.txt
```
```bash
python main.py
python3 run_inference_server.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -t 6 
```

## Docker (API + BitNet в одном контейнере)
Собрать образ:
```bash
docker build -f docker/Dockerfile -t rag-bitnet:latest .
```

Запустить контейнер с внешней папкой BitNet:
```bash
docker run --rm -p 8000:8000 -p 8080:8080 \
  -v "$PWD/data:/app/data" \
  -v "/absolute/path/to/BitNet:/opt/bitnet" \
  -e BITNET_DIR=/opt/bitnet \
  -e BITNET_MODEL=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -e BITNET_THREADS=6 \
  rag-bitnet:latest
```

Если у вас скрипт/модель лежат в других путях внутри BitNet-папки, передайте:
- `BITNET_SERVER_SCRIPT` (по умолчанию `run_inference_server.py`)
- `BITNET_MODEL` (путь относительно `BITNET_DIR`)

## Основные env-переменные
- `API_HOST`, `API_PORT` - адрес и порт FastAPI.
- `BITNET_HOST`, `BITNET_PORT` - где доступен BitNet inference server.
- `BITNET_MAX_TOKENS`, `BITNET_TEMPERATURE`, `BITNET_TIMEOUT` - параметры генерации и тайм-аут запроса.
- `DATA_DIR` - путь к хранилищу индекса/данных.
- `USE_CROSS_RERANKER`, `CROSS_RERANKER_TOPK`, `RERANKER_BATCH`, `EMBED_BATCH` - настройки retrieval/pipeline.

## Структура
- `rag_bitnet/config.py` — env/пути и параметры моделей.
- `rag_bitnet/clients/bitnet_client.py` — клиент BitNet API.
- `rag_bitnet/prompting/prompt.py` — сборка промпта.
- `rag_bitnet/api/*` и `rag_bitnet/server.py` — FastAPI
- `rag_bitnet/storage/data_store.py` — загрузка/сохранение индекса, faiss/BM25.
- `rag_bitnet/models/*` — ленивые обёртки embedding и reranker.
- `rag_bitnet/ingest/*` — загрузка файлов и краулер.
- `rag_bitnet/retrieval/retrieve.py` — поиск + (опц.) реранк.

