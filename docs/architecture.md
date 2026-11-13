# Architecture

## Data Flow (RAG pipeline)

```
User Request
     │
     ▼
  /ask ──► ask_service ──► retrieve_async() ──► FAISS + BM25 + reranker
     │                                              │
     │                                              ▼
     │                                      context chunks
     │                                              │
     ▼                                              ▼
   LLM.generate(messages, context) ◄──── _build_user_message()
      │
      ▼
   журнал операций + CsvLogger (метрики без содержимого по умолчанию)
     │
     ▼
  BitNetAPI ──► HTTP POST ──► BitNet inference server
     │                           (chat or completion mode)
     ▼
  LLMResp { id, content, timings }
     │
     ├── stream=false ──► AskResp (JSON)
     └── stream=true  ──► ndjson-stream (chunks + final AskResp)
```

## Публикация поискового индекса

`data_store` строит FAISS, BM25 и снимок данных в памяти, затем записывает новую версию в отдельный каталог
`DATA_DIR/versions/`. После записи файлов и метаданных с SHA-256 атомарно заменяется `current.json`, а новый снимок
становится доступным запросам. Каждый поиск использует один снимок, полученный при его начале. При запуске проверяются
версия формата, embedding-модель, контрольные суммы и количество векторов.

Индексация сериализована потоковой блокировкой одного процесса. Несколько процессов с общим каталогом не поддерживаются:
их блокировки и снимки не синхронизируются. Публикация через `os.replace` не гарантирует сохранность при внезапном
отключении питания; старые и незавершённые версии автоматически не очищаются. Порядок резервного копирования описан
в README.

## LLM Client Architecture

```
LLM (abstract)
  ├── generate(messages, ...) -> LLMResp
  └── generate_stream(messages, ...) -> Iterator[LLMResp]

BitNetAPI(LLM)
  ├── mode = "chat" | "completion"  (from config.BITNET_MODE)
  │
  ├── generate():
  │   ├── _generate_chat()       POST /v1/chat/completions, stream=false
  │   └── _generate_completion()  POST /completion, stream=false
  │
  └── generate_stream():
      ├── _generate_chat_stream()       SSE parse via ChatCompletionChunk
      └── _generate_completion_stream()  SSE parse via CompletionStreamChunk
```

## Two Modes

### Chat Mode (`BITNET_MODE=chat`)

- Endpoint: `/v1/chat/completions` (OpenAI-compatible)
- Payload: `{ messages: [...], stream: true/false, max_tokens, temperature }`
- Контекст вплетается в последнее user-сообщение.
- Из `usage` берётся количество токенов; неизвестные времена отдельных стадий равны `null`.
- Парсинг SSE через `ChatCompletionChunk` → `choices[0].delta.content`.

### Completion Mode (`BITNET_MODE=completion`)

- Endpoint: `/completion` (llama.cpp / BitNet native)
- Payload: `{ prompt: "...", stream: true/false, n_predict, temperature, stop: [...] }`
- Сообщения рендерятся через Jinja2-шаблон (`completion_chat_template.jinja`) в единую строку с `<|begin_of_text|>`,
  `<|eot_id|>` и `Assistant:`.
- Timings извлекаются из ответа сервера (`timings`).
- Парсинг SSE через `CompletionStreamChunk` → `content`, `stop`.

## LLMResp — единый формат ответа

```python
class LLMResp(BaseModel):
    id: str = ""
    content: str
    timings: Optional[Timings] = None
    model: str = ""
    final: bool = False
```

- `content` — текст; в стриме финальное событие с `final=True` содержит метаданные, а не повтор полного ответа.
- `id` — идентификатор ответа (chat-completion id или id_slot).
- `timings` — метрики производительности (только в последнем чанке стрима или в JSON-ответе).
- `model` — имя модели из ответа сервера.

## Timings

```python
class Timings(BaseModel):
  prompt_n: int
  prompt_ms: float
  prompt_per_token_ms: float
  prompt_per_second: float
  predicted_n: int
  predicted_ms: float
  predicted_per_token_ms: float
  predicted_per_second: float
```

## Журнал операций

ASGI middleware создаёт `operation_id` для каждого прикладного POST-запроса и записывает начало и итог в
`logs/operations.jsonl`. `ContextVar` передаёт общие сведения сервисам и рабочим потокам; финальная запись появляется
после завершения ответа, включая streaming. Сервис индексации фиксирует версию и размер непосредственно при публикации,
пока удерживается блокировка записи. Поиск и генерация добавляют длительности, ошибки — безопасную категорию и тип.

`CsvLogger` сохраняет метрики `/ask` в `logs/ask_service.csv` и связывает обновления строки по `operation_id`.
Вопросы, контекст и ответы добавляются только при `LOG_REQUEST_CONTENT=true`. Оба журнала и технические файлы имеют
ротацию по размеру; подробности, ограничения и правила доступа приведены в README.
