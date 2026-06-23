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
  BitNetAPI ──► HTTP POST ──► BitNet inference server
     │                           (chat or completion mode)
     ▼
  LLMResp { id, content, timings }
     │
     ├── stream=false ──► AskResp (JSON)
     └── stream=true  ──► ndjson-stream (chunks + final AskResp)
```

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
- Timings вычисляются клиентом из `usage` (prompt_tokens / completion_tokens) и elapsed time.
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
```

- `content` — текст (в стриме — один токен/слово, в последнем чанке — полный ответ).
- `id` — идентификатор ответа (chat-completion id или id_slot).
- `timings` — метрики производительности (только в последнем чанке стрима или в JSON-ответе).
