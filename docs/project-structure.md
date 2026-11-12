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
│   ├── llm.py                     Abstract LLM interface (generate / generate_stream)
│   ├── bitnet_client.py           BitNetAPI — two modes: chat (/v1/chat/completions) / completion (/completion)
│   └── completion_chat_template.jinja  Jinja2 template for completion-mode prompt rendering
│
├── common/
│   ├── config.py                  Environment config, typed helpers (_env_str, _env_int, etc.)
│   ├── logger.py                  Logging setup (file + console) + CsvLogger
│   ├── text_utils.py              Tokenization and excerpt helpers
│   └── formatters.py              File/HTML/JSON parsing, site crawling, document formatting
│
├── rag/
│   ├── ingest.py                  File/site ingestion, text extraction, chunking
│   ├── retrieval.py               FAISS + BM25 retrieval, cross-encoder reranking
│   ├── data_store.py              Persistence layer (rows, embeddings, index, meta, bm25 corpus)
│   └── models.py                  Lazy loading of embedding/reranker models
│
├── schemas/
│   ├── app_schema.py              Application schemas: Message, LLMReq, LLMResp, Timings, AskReq, AskResp, IngestFilesReq, IngestSiteReq
│   └── bitnet_schema.py           BitNet API response schemas: ChatCompletionChunk, ChatCompletionResponse, CompletionResponse, CompletionStreamChunk, Usage, GenerationSettings
│
├── logs/
│   ├── .gitkeep
│   └── benchmarks/mmlu_pro/       Benchmark results (JSON + summary)
│
├── data/
│   ├── rag_docs/                  Исходные документы для индексации (.txt, .md, .pdf)
│   └── vector_store/              FAISS index, BM25 corpus, rows.jsonl, embeddings.npy, meta.json
│
├── docs/
│   ├── endpoints.md           Endpoint documentation with examples
│   ├── project-structure.md   This file
│   ├── architecture.md        LLM client architecture and data flow
│   └── benchmarks.md          Benchmark setup, adapters, and running
│
├── benchmarks/                lm-eval harness adapters and benchmark runners
│   ├── lm_eval_adapter.py     BitNetLM — lm-eval model adapter
│   ├── stub_lm.py             StubLM — dummy model (returns gold labels)
│   ├── run_all.py             Run all benchmarks sequentially
│   ├── run_mmlu_pro.py        MMLU-Pro (14 subjects)
│   ├── run_ifeval.py          IFEval (instruction following)
│   ├── run_medqa.py           MedQA (4 options)
│   ├── run_meddialog.py       MedDialog (dialogue, qsumm, perplexity)
│   ├── test_stub.py           Smoke-test with StubLM
│   └── _utils.py              Save helpers (JSON + summary table)
│
├── main.py                    Entry point (uvicorn runner)
├── .env                       Local environment config
├── .env.example               Environment template
├── requirements.txt           RAG API dependencies
├── requirements-bench.txt     Benchmark dependencies (включает -r requirements.txt)
├── requirements_full.txt      Полный frozen-слепок зависимостей (воспроизводимое окружение)
├── test_main.http             HTTP request examples (IDE REST Client)
├── AGENTS.md                  Conventions for AI coding agents
└── README.md                  Project overview
```

## Key Conventions

- **Schemas** — разделены на `app_schema.py` (прикладные: AskReq/Resp, IngestReq/Resp, Message, Timings, LLMReq/Resp)
  и `bitnet_schema.py` (сырые ответы BitNet API: ChatCompletionChunk, CompletionResponse, ChatCompletionResponse и др.).
- **LLM interface** — `generate()` возвращает `LLMResp` (id, content, timings, model), `generate_stream()` — итератор
  `LLMResp`.
- **Two modes** — `BITNET_MODE=chat` использует `/v1/chat/completions` с messages-массивом; `BITNET_MODE=completion`
  рендерит промпт через Jinja2-шаблон `completion_chat_template.jinja`.
- **Context injection** — общая функция `_build_user_message()` в `bitnet_client.py` модифицирует последнее
  user-сообщение, добавляя контекст.
- **Timings** — в chat-режиме число токенов берётся из `usage`, неизвестные времена равны `null`; в completion они
  берутся из ответа сервера.
- **Streaming** — ndjson (`application/x-ndjson`): каждый чанк — отдельная JSON-строка с `{"id","content","model"}`,
  финальная строка — `AskResp` с `id`, `answer`, `timings`, `used_context` или строка с ошибкой.
- **CSV logging** — содержимое запроса и ответа записывается только при `LOG_REQUEST_CONTENT=true`.
- **Data storage** — версии корпуса, FAISS и метаданных хранятся в `DATA_DIR/versions`; `current.json` указывает на
  опубликованную версию.
- **Config** — `SYSTEM_PROMPT` и `BITNET_MODE` валидируются в `common/config.py`; `BITNET_MODE` принимает только
  `chat` или `completion`.
