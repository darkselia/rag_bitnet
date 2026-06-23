# AGENTS.md — для AI-агентов

## Общие правила

- Проект Python/FastAPI.
- **Не плодить файлы без необходимости** — предпочитать редактирование существующих.
- **Следовать существующим паттернам** — смотреть соседние файлы перед созданием новых.
- Показывать diff файлов, чтобы они отображались в IDE.
- Коммиты должны быть атомарными и описательными с использованием правил Conventional commits.

## Ключевые файлы и их роль

| Файл                       | Назначение                                                                                                                                        |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `api/routes.py`            | Все HTTP-роуты. Добавлять новые эндпоинты здесь.                                                                                                  |
| `api/ask_service.py`       | Оркестрация RAG-пайплайна.                                                                                                                        |
| `rag/ingest.py`            | Функции для индексации файлов и сайтов.                                                                                                           |
| `rag/retrieve.py`          | Функции для извлечения релевантного контекста (FAISS + BM25 + reranker).                                                                          |
| `api/ingest_service.py`    | Логика обработки запросов на индексацию.                                                                                                          |
| `clients/llm.py`           | Абстрактный класс LLM. Сигнатуры: `generate(messages, max_tokens, temperature, context) -> LLMResp`, `generate_stream(...) -> Iterator[LLMResp]`. |
| `clients/bitnet_client.py` | Реализация BitNet. Два режима: `chat` и `completion`. Внутренние методы `_generate_chat*` / `_generate_completion*`.                              |
| `schemas/app_schema.py`    | Pydantic-схемы приложения: `Message`, `LLMReq`, `LLMResp`, `Timings`, `AskResp`.                                                                  |
| `schemas/bitnet_schema.py` | Схемы ответов BitNet API: `ChatCompletionChunk`, `CompletionResponse`, `CompletionStreamChunk`.                                                   |
| `common/config.py`         | Конфигурация из env. Добавлять новые переменные через `_env_str/_env_int` и т.д.                                                                  |
| `common/logger.py`         | Настройка логирования. Использовать `get_logger(name)`.                                                                                           |

## Правила написания кода

1. **Pydantic** — v2. Использовать `model_validate()` / `model_validate_json()`.
2. **Типизация** — обязательна. Использовать `list[...]`, `dict[...]`, `Optional`, `|` union syntax.
3. **Imports** — сначала стандартная библиотека, потом сторонние, потом внутренние.
