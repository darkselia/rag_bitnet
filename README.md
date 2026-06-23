# RAG BitNet API

FastAPI-сервис для RAG-чата поверх локальных документов и страниц сайта. Сервер принимает сообщения, ищет релевантный
контекст в FAISS/BM25 индексе, отправляет запрос в BitNet inference server и возвращает ответ с использованным
контекстом.

## Возможности

- Индексация `.txt`, `.pdf`, `.md` файлов и HTML-страниц одного домена.
- Гибридный retrieval: dense embeddings (sentence-transformers) + BM25 + опциональный cross-encoder reranker.
- Два режима LLM: **chat** (`/v1/chat/completions`) и **completion** (`/completion`) — переключаются через
  `BITNET_MODE`.
- JSON и ndjson-streaming ответы через `/ask`.

## Быстрый старт

```bash
cp .env.example .env
pip install -r requirements.txt
python main.py || uvicorn api.server:app --host 0.0.0.0 --port 8000
```

BitNet inference server: https://github.com/microsoft/BitNet

## Docker

Сборка:

```bash
docker build -f docker/Dockerfile -t rag-bitnet:latest .
```

Запуск с внешней папкой BitNet:

```bash
docker run --rm -p 8000:8000 -p 8080:8080 \
  -v "$PWD/data/vector_store:/app/data/vector_store" \
  -v "/absolute/path/to/BitNet:/opt/bitnet" \
  -e BITNET_DIR=/opt/bitnet \
  -e BITNET_MODEL=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -e BITNET_THREADS=6 \
  rag-bitnet:latest
```

Контейнер запускает BitNet server и API одним процессом через `docker/start_services.sh`.

## Основные эндпоинты

| Метод | Путь            | Описание                            |
|-------|-----------------|-------------------------------------|
| POST  | `/ingest/files` | Индексация локальных файлов         |
| POST  | `/ingest/site`  | Индексация сайта                    |
| POST  | `/ask`          | RAG-вопрос (JSON или ndjson-stream) |

Подробнее — в [docs/endpoints.md](docs/endpoints.md).

## Структура проекта

```
api/           FastAPI routes, services, server
clients/       LLM abstraction + BitNet client
common/        Config, logger, utils, file/HTML formatters
rag/           Ingestion, retrieval, data store
schemas/       Pydantic models (app + BitNet API)
docs/          Документация проекта
data/rag_docs/      Исходные файлы для индексации (.txt, .md, .pdf)
data/vector_store/  FAISS + BM25 индексы, rows.jsonl, embeddings.npy
```

Полная структура — в [docs/project-structure.md](docs/project-structure.md).

## Конфигурация

Основные переменные окружения (см. `.env.example`):

- `BITNET_MODE` — режим LLM: `chat` или `completion`
- `LLM_PROVIDER` — `bitnet`
- `SYSTEM_PROMPT` — системный промпт для rag
- `BITNET_HOST`, `BITNET_PORT` — адрес BitNet inference server

## Зависимости

Перечислены в `requirements.txt`. Ключевые: FastAPI, Pydantic, sentence-transformers, FAISS, Jinja2, requests.
