# Endpoints

## Доступ

Все перечисленные маршруты требуют заголовок `X-API-Key` со значением `RAG_API_KEY`, настроенным на сервере. При
отсутствующем или неверном ключе возвращается HTTP 401. Ключ проверяется до запуска индексации или генерации, в том
числе
для потокового ответа. Если `RAG_API_KEY` не задан, API не запускается.

```http
X-API-Key: <RAG_API_KEY>
```

В `/docs` ключ вводится через **Authorize**; страницы документации и OpenAPI-схема доступны без ключа. Общий ключ даёт
доступ и к запросам, и к изменению базы. Интегрирующее приложение хранит его на сервере и самостоятельно проверяет права
своих пользователей. Сервер BitNet не использует этот ключ; его сетевой доступ ограничивается отдельно.

---

## `POST /ingest/files`

Рекурсивно индексирует локальные `.pdf`, `.docx`, `.txt`, `.csv`, `.md`, `.html` и `.htm` файлы из указанной папки.
Исходные файлы остаются в этой папке; сервис сохраняет извлечённые фрагменты и индекс.

**Request:**

```json
{
  "folder": "data/rag_docs",
  "replace_existing": true
}
```

- `folder` — путь к папке с документами.
- `replace_existing` — если `true`, старые чанки этих файлов удаляются и заменяются новыми.

CSV должен содержать заголовки и данные в UTF-8 или UTF-8 с BOM. Разделитель — запятая или точка с запятой;
значения в кавычках могут содержать переносы строк. Количество столбцов должно совпадать во всех строках.
DOCX обрабатывается с сохранением порядка абзацев и таблиц. Из PDF извлекается текстовый слой, OCR не выполняется.

По умолчанию максимальный размер файла — 25 МиБ, сумма файлов за операцию — 100 МиБ, распакованный DOCX — 100 МиБ.
Пределы изменяются через `INGEST_MAX_FILE_BYTES`, `INGEST_MAX_BATCH_BYTES` и `INGEST_MAX_DOCX_UNCOMPRESSED_BYTES`.
Пустой, повреждённый, нераспознаваемый или превышающий пределы файл завершает операцию с HTTP 400 до публикации
нового индекса. При `replace_existing=true` ранее сохранённые фрагменты таких файлов остаются доступными.

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
      "content": "What the kolobok said to wolf?"
    }
  ],
  "stream": false,
  "sources": [
    "docs",
    "site"
  ],
  "top_k": 6,
  "temperature": 0.4
}
```

Поля:

- `messages` — массив сообщений (обязательно хотя бы одно с `role: "user"`). Системный промпт добавляется автоматически
  из `SYSTEM_PROMPT`.
- `stream` — если `true`, ответ в формате ndjson-потока.
- `sources` — список источников: `docs`, `site` или оба.
- `site_domain` — опциональный hostname сайта: точный домен и его поддомены.
- `top_k` — максимальное количество контекстов, от `1` до `50`; результатов может быть меньше.
- `temperature` — температура LLM, от `0.0` до `2.0`.

Если поиск не вернул фрагментов, ответ имеет HTTP 200, `used_context: []` и сообщение
`Couldn't find the answer in the database.`. Для такого ответа генерация в LLM не запускается. Это изменение прежнего
контракта, при котором возвращался HTTP 404.

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
    "prompt_ms": null,
    "prompt_per_token_ms": null,
    "prompt_per_second": null,
    "predicted_n": 22,
    "predicted_ms": null,
    "predicted_per_token_ms": null,
    "predicted_per_second": null
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
{"id": "", "content": "Kolobok", "model": ""}
{"id": "", "content": " visited", "model": ""}
{"id": "", "content": " the", "model": ""}
...
{"id": "chatcmpl-xxx", "answer": {"role": "assistant", "content": "Kolobok visited the ..."}, "used_context": [...]}
```

Поток состоит из:

1. **Чанки контента** — `{"id": "", "content": "...", "model": ""}` — по одному токену/слову.
2. **Финальный JSON** — `AskResp` с `id`, `answer`, необязательными `timings` и `used_context`.

При отсутствии контекста поток содержит один чанк с сообщением об отсутствии ответа и финальный `AskResp` с тем же
сообщением и пустым `used_context`.

При ошибке после начала потока последней строкой передаётся `{"error": "LLM stream failed"}` без успешного `AskResp`.
Ошибка соединения до начала потока возвращается как HTTP 502 или 504. В chat-режиме времена отдельных стадий равны
`null`, если сервер не сообщил их; число токенов доступно при наличии `usage`.

Если включён `BITNET_CONTEXT_SIZE`, наименее релевантные чанки исключаются до помещения промпта в окно модели;
`used_context` содержит только отправленные чанки. Если сами сообщения не помещаются, возвращается HTTP 413. Для точной
проверки server должен поддерживать `/tokenize`, а в chat также `/apply-template`.
