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
   CsvLogger (логирование в .csv с таймингами)
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

## CsvLogger

Каждый RAG-запрос логируется в `logs/ask_service.csv` через потокобезопасный `CsvLogger`.
Поддерживает обновление строки по ключу (используется `timestamp`): сначала пишется строка с retrieve-полями,
затем при ответе LLM она дополняется LLM-полями.

## CsvLogger

```python
class CsvLogger:
  def __init__(self, name: str, fieldnames: list[str] | None = None, key: str | None = None)

    def log(self, row: dict) -> None
```
