# Project Structure

```
rag_bitnet/
├── api/
│   ├── server.py              FastAPI app, router mounting, startup event
│   ├── routes.py              HTTP routes (ingest, ask)
│   ├── ask_service.py         /ask orchestration (RAG pipeline)
│   └── ingest_service.py      /ingest orchestration
│
├── clients/
│   ├── llm.py                 Abstract LLM interface (generate / generate_stream)
│   ├── bitnet_client.py       BitNetAPI — two modes: chat (/v1/chat/completions) / completion (/completion)
│   └── completion_chat_template.jinja   Jinja2 template for completion-mode prompt rendering
│
├── common/
│   ├── config.py              Environment config, typed helpers (_env_str, _env_int, etc.)
│   ├── logger.py              Logging setup (file + console)
│   ├── text_utils.py          Tokenization and excerpt helpers
│   └── formatters.py          File/HTML/JSON parsing, site crawling, document formatting
│
├── rag/
│   ├── ingest.py              File/site ingestion, text extraction, chunking
│   ├── retrieval.py           FAISS + BM25 retrieval, cross-encoder reranking
│   ├── data_store.py          Persistence layer (rows, embeddings, index, meta)
│   └── models.py              Lazy loading of embedding/reranker models
│
├── schemas/
│   ├── app_schema.py          Application schemas: Message, LLMReq, LLMResp, Timings, AskResp, etc.
│   └── bitnet_schema.py       BitNet API response schemas: ChatCompletionChunk, CompletionResponse, CompletionStreamChunk, etc.
│
├── data/
│   ├── rag_docs/              Исходные документы для индексации (.txt, .md, .pdf)
│   └── vector_store/          FAISS index, BM25 corpus, rows.jsonl, embeddings.npy, meta.json
│
├── tests/
│   └── test_bitnet_client.py  Unit tests for BitNetAPI with mocked HTTP
│
├── docs/
│   ├── endpoints.md           Endpoint documentation with examples
│   ├── project-structure.md   This file
│   └── architecture.md        LLM client architecture and data flow
│
├── main.py                    Entry point (uvicorn runner)
├── .env                       Local environment config
├── .env.example               Environment template
├── requirements.txt           Python dependencies
├── test_main.http             HTTP request examples (IDE REST Client)
├── AGENTS.md                  Conventions for AI coding agents
└── README.md                  Project overview
```

## Key Conventions

- **Schemas** — разделены на `app_schema.py` (прикладные) и `bitnet_schema.py` (сырые ответы BitNet API).
- **LLM interface** — `generate()` возвращает `LLMResp` (id, content, timings), `generate_stream()` — итератор
  `LLMResp`.
- **Two modes** — `BITNET_MODE=chat` использует `/v1/chat/completions` с messages-массивом; `BITNET_MODE=completion`
  рендерит промпт через Jinja2.
- **Context injection** — общая функция `_build_user_message()` в `bitnet_client.py` модифицирует последнее
  user-сообщение, добавляя контекст.
- **Timings** — для chat-режима вычисляются пропорционально из `usage` + elapsed time; для completion — берутся из
  ответа сервера.
- **Streaming** — ndjson (`application/x-ndjson`): каждый чанк — отдельная JSON-строка, финальная строка — полный
  `AskResp` с `used_context`.
- **Data storage** — индексные файлы (`rows.jsonl`, `embeddings.npy`, `index.faiss`, `bm25_corpus.json`, `meta.json`)
  хранятся в `data/vector_store/`, настраивается через `DATA_DIR`.
