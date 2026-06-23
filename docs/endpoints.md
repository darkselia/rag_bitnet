# Endpoints

## `POST /ingest/files`

Индексирует локальные `.txt`, `.md`, `.pdf` файлы из указанной папки.

**Request:**

```json
{
  "folder": "data/rag_docs",
  "replace_existing": true
}
```

- `folder` — путь к папке с документами.
- `replace_existing` — если `true`, старые чанки этих файлов удаляются и заменяются новыми.

**Response:**

```json
{
  "added_chunks": 2,
  "total_chunks": 461
}
```

---

## `POST /ingest/site`

Индексирует HTML-страницы одного домена с обходом до указанного лимита страниц.

**Request:**

```json
{
  "base_url": "https://example.com",
  "max_pages": 100
}
```

---

## `POST /ask`

RAG-вопрос. Сервер ищет релевантный контекст в индексе, формирует prompt и отправляет в LLM.

**Request (JSON, body):**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What bun said to wolf?"
    }
  ],
  "stream": false,
  "sources": [
    "docs",
    "site"
  ],
  "top_k": 6,
  "temperature": 0.6
}
```

Поля:

- `messages` — массив сообщений (обязательно хотя бы одно с `role: "user"`). Системный промпт добавляется автоматически
  из `SYSTEM_PROMPT`.
- `stream` — если `true`, ответ в формате ndjson-потока.
- `sources` — список источников: `docs`, `site` или оба.
- `site_domain` — опциональный фильтр для `site`-источников.
- `top_k` — количество контекстов для ретрива, от `1` до `50`.
- `temperature` — температура LLM, от `0.0` до `2.0`.

---

**Response (JSON, `stream: false`):**

```json
{
  "id": "chatcmpl-xxx",
  "answer": {
    "role": "assistant",
    "content": "Kolobok visited the Hare first."
  },
  "timings": {
    "prompt_n": 1112,
    "predicted_n": 22,
    "predicted_per_second": 6.72
  },
  "used_context": [
    {
      "id": "doc:fairytaleEN.txt:8",
      "text": "...",
      "excerpt": "...",
      "score": 0.38,
      "source": "fairytaleEN.txt",
      "kind": "docs",
      "meta": {
        "path": "data/rag_docs/fairytaleEN.txt"
      }
    }
  ]
}
```

---

**Response (ndjson-stream, `stream: true`):**

Каждая строка — валидный JSON (`application/x-ndjson`).

```jsonlines
{"content": "Kolobok"}
{"content": " visited"}
{"content": " the"}
...
{"id": "chatcmpl-xxx", "answer": {"role": "assistant", "content": "Kolobok visited the ..."}, "timings": {"prompt_n": 1112, ...}, "used_context": [...]}
```

Поток состоит из:

1. **Чанки контента** — `{"content": "..."}` — по одному токену/слову.
2. **Финальный JSON** — полный `AskResp` с `id`, `answer`, `timings` и `used_context`.
