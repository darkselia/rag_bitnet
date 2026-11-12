# RAG BitNet API

FastAPI-сервис для RAG-чата поверх локальных документов и страниц сайта. Сервер принимает сообщения, ищет релевантный
контекст в FAISS/BM25 индексе, отправляет запрос в BitNet inference server и возвращает ответ с использованным
контекстом.

## Возможности

- Индексация `.txt`, `.pdf`, `.md` файлов и HTML-страниц одного домена.
- Гибридный retrieval: dense embeddings (sentence-transformers) + BM25 + опциональный cross-encoder reranker.
- Два режима LLM: **chat** (`/v1/chat/completions`) и **completion** (`/completion`) — переключаются через
  `BITNET_MODE`.
- JSON и ndjson-streaming ответы через `/ask` (поток из чанков + финальный `AskResp`).
- CSV-логирование всех RAG-запросов с таймингами retrieve/LLM.

## Быстрый старт

```bash
cp .env.example .env
pip install -r requirements.txt        # RAG API
pip install -r requirements-bench.txt  # + benchmarks (опционально)
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
docker run --rm -p 127.0.0.1:8000:8000 -p 127.0.0.1:8080:8080 \
  -v "$PWD/data/vector_store:/app/data/vector_store" \
  -v "/absolute/path/to/BitNet:/opt/bitnet" \
  -e BITNET_DIR=/opt/bitnet \
  -e BITNET_MODEL=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -e BITNET_THREADS=6 \
  rag-bitnet:latest
```

Контейнер запускает BitNet server и API одним процессом через `docker/start_services.sh`.

Сервис рассчитан на доверенный локальный доступ: авторизации нет, а индексация принимает пути к файлам и URL. Для
внешнего доступа необходима отдельная защита сети и аутентификация. Порт BitNet в контейнере доступен только внутри
контейнера, если сервер привязан к `127.0.0.1`.

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
clients/       LLM abstraction + BitNet client (+ Jinja2 template)
common/        Config, logger (CsvLogger), utils, file/HTML formatters
rag/           Ingestion, retrieval, data store
schemas/       Pydantic models (app schemas + BitNet API schemas)
docs/          Документация проекта (в т.ч. docs/benchmarks.md)
benchmarks/    lm-eval harness адаптеры и раннеры бенчмарков
logs/          CSV-логи запросов, benchmark results
data/rag_docs/      Исходные файлы для индексации (.txt, .md, .pdf)
data/vector_store/  FAISS + BM25 индексы, rows.jsonl, embeddings.npy
```

Полная структура — в [docs/project-structure.md](docs/project-structure.md).

## Конфигурация

Основные переменные окружения (см. `.env.example`):

- `BITNET_MODE` — режим LLM: `chat` или `completion`
- `LLM_PROVIDER` — `bitnet`
- `SYSTEM_PROMPT` — системный промпт для RAG
- `BITNET_HOST`, `BITNET_PORT` — адрес BitNet inference server
- `DATA_DIR` — директория хранения индексов (по умолч. `./data/vector_store`)
- `BITNET_TIMEOUT` — таймаут HTTP в секундах, по умолчанию 60.
- `BITNET_CONTEXT_SIZE` — размер контекста одного слота, при `0` проверка размера отключена. При положительном значении
  используется `/tokenize`; для chat также необходим `/apply-template`. Сервер без этих endpoints возвращает ошибку
  вместо приблизительного подсчёта.
- `LOG_REQUEST_CONTENT` — сохранять ли вопросы, контекст и ответы в CSV; по умолчанию `false`.
- `RETRIEVAL_SIM_THRESHOLD` — порог косинусной близости для векторных кандидатов; `0` отключает порог. Его следует
  подбирать на собственном наборе запросов.

## Зависимости

| Файл                     | Назначение                                                                                            |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| `requirements.txt`       | RAG API: FastAPI, Pydantic, sentence-transformers, FAISS, Jinja2, rank-bm25, requests                 |
| `requirements-bench.txt` | Бенчмарки: `-r requirements.txt` + lm-eval, datasets, bert-score, rouge-score, bleurt, tensorflow-cpu |
| `requirements_full.txt`  | Полный frozen-слепок всех зависимостей (воспроизводимое окружение)                                    |
