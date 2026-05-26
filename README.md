# RAG BitNet API

FastAPI-сервис для RAG-чата поверх локальных документов и страниц сайта. Сервер принимает вопрос, ищет релевантный контекст в FAISS/BM25 индексе, отправляет prompt в BitNet inference server и возвращает ответ, источники и использованный контекст.

## Возможности

- Индексация локальных `.txt` и `.md` файлов.
- Индексация HTML-страниц одного домена.
- Гибридный retrieval: dense embeddings через `sentence-transformers`, BM25 и optional cross-encoder reranker.
- JSON-ответы и streaming-ответы.
- Детерминированное поле `sources` в API-ответе.
- Persisted index в папке `data/`.

## Структура

```text
api/
  server.py          FastAPI app
  routes.py          HTTP routes
  schemas.py         Pydantic request/response schemas
  ask_service.py     /ask orchestration
  ingest_service.py  /ingest orchestration

rag/
  ingest.py          file/site ingestion, text extraction, chunking
  retrieval.py       FAISS + BM25 + reranker retrieval
  prompt.py          BitNet/Llama-style prompt builder
  data_store.py      rows/embeddings/FAISS/BM25 load/save/rebuild
  models.py          lazy embedding/reranker model loading

clients/
  llm.py             LLM interface
  bitnet_client.py   BitNet /completion client

common/
  config.py          env config and paths
  text_utils.py      tokenization and excerpt helpers

data/                generated index files
docs/                local source documents
docker/              container startup files
```

## Локальный запуск

```bash
cp .env.example .env
conda create -n rag-bitnet python=3.11 -y
conda activate rag-bitnet
pip install -r requirements.txt
```

BitNet inference server запускается отдельно:

```bash
cd /path/to/BitNet
python3 run_inference_server.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -t 6
```

Запуск API:

```bash
python main.py
```

По умолчанию API слушает `http://127.0.0.1:8000`, а BitNet ожидается на `http://127.0.0.1:8080/completion`.

## Быстрый сценарий

1. Пересобрать индекс по локальным документам:

```bash
curl -X POST "http://127.0.0.1:8000/ingest/files" \
  -H "Content-Type: application/json" \
  -d '{"folder":"./docs","replace_existing":true}'
```

2. Задать вопрос:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"What bun said to wolf?","sources":["docs"],"top_k":6,"temperature":0.2}'
```

3. Streaming-вариант:

```bash
curl -N -X POST "http://127.0.0.1:8000/ask?type=stream" \
  -H "Content-Type: application/json" \
  -d '{"query":"What bun said to wolf?","sources":["docs"],"top_k":6,"temperature":0.2}'
```

Больше готовых запросов есть в `test_main.http`.

## API

### `POST /ingest/files`

Индексирует локальные `.txt` и `.md` файлы из папки.

```json
{
  "folder": "./docs",
  "replace_existing": true
}
```

Поля:

- `folder` - путь к папке с документами. Поддерживаются `.txt`, `.md`, `.html`, `.htm`, `.pdf`.
- `replace_existing` - если `true`, старые чанки этих файлов удаляются и заменяются новыми. Если `false`, уже существующие `id` чанков не добавляются повторно.

Ответ:

```json
{
  "added_chunks": 2,
  "total_chunks": 461
}
```

### `POST /ingest/site`

Индексирует HTML-страницы одного домена. Для SPA также пытается найти JSON-данные в JS-бандлах и индексировать их как текстовые документы.

```json
{
  "base_url": "https://example.com",
  "max_pages": 100
}
```

### `POST /ask`

Возвращает JSON-ответ.

```json
{
  "query": "What bun said to wolf?",
  "sources": ["docs"],
  "top_k": 6,
  "temperature": 0.2
}
```

Поля:

- `query` - вопрос пользователя.
- `sources` - список источников: `docs`, `site` или оба.
- `site_domain` - optional фильтр для `site` источников.
- `top_k` - сколько контекстов вернуть и отправить в prompt, от `1` до `50`.
- `temperature` - optional override температуры LLM, от `0.0` до `2.0`.

Ответ:

```json
{
  "answer": "...",
  "sources": ["doc:fairytale.txt:0"],
  "used_context": [
    {
      "id": "doc:fairytale.txt:0",
      "text": "...",
      "excerpt": "...",
      "score": 0.95,
      "source": "fairytale.txt",
      "kind": "docs",
      "meta": {
        "path": "docs/fairytale.txt"
      }
    }
  ]
}
```

### `POST /ask?type=stream`

Возвращает `text/plain` stream. Тело запроса такое же, как у `/ask`.

## Конфигурация

Основные переменные из `.env.example`:

```env
API_HOST=0.0.0.0
API_PORT=8000

DATA_DIR=./data

EMBED_MODEL=intfloat/multilingual-e5-base
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
USE_CROSS_RERANKER=true
CROSS_RERANKER_TOPK=12
EMBED_BATCH=64
RERANKER_BATCH=16

LLM_PROVIDER=bitnet
LLM_MAX_TOKENS=300
LLM_TEMPERATURE=0.6

BITNET_HOST=127.0.0.1
BITNET_PORT=8080
BITNET_TIMEOUT=60000
```

## Индекс и данные

`DATA_DIR` содержит generated-файлы:

```text
rows.jsonl          чанки и metadata
embeddings.npy     dense embeddings
index.faiss        FAISS index
bm25_corpus.json   тексты для BM25
meta.json          metadata индекса
```

Эти файлы можно пересобрать через `/ingest/files` с `replace_existing=true`. 

## Docker

Сборка:

```bash
docker build -f docker/Dockerfile -t rag-bitnet:latest .
```

Запуск с внешней папкой BitNet:

```bash
docker run --rm -p 8000:8000 -p 8080:8080 \
  -v "$PWD/data:/app/data" \
  -v "/absolute/path/to/BitNet:/opt/bitnet" \
  -e BITNET_DIR=/opt/bitnet \
  -e BITNET_MODEL=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -e BITNET_THREADS=6 \
  rag-bitnet:latest
```

Контейнер запускает BitNet server и API одним процессом через `docker/start_services.sh`.
