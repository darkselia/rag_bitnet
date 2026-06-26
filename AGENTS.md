# AGENTS.md — для AI-агентов

## Общие правила

- Проект Python/FastAPI.
- **Не плодить файлы без необходимости** — предпочитать редактирование существующих.
- **Следовать существующим паттернам** — смотреть соседние файлы перед созданием новых.
- Показывать diff файлов, чтобы они отображались в IDE.
- Коммиты должны быть атомарными и описательными с использованием правил Conventional commits.

## Файлы зависимостей

| Файл                     | Назначение                                                                 |
|--------------------------|----------------------------------------------------------------------------|
| `requirements.txt`       | RAG API: FastAPI, Pydantic, sentence-transformers, FAISS, Jinja2, requests |
| `requirements-bench.txt` | Бенчмарки: `-r requirements.txt` + lm-eval, datasets, метрики              |
| `requirements_full.txt`  | Полный frozen-слепок всех зависимостей (воспроизводимое окружение)         |

## Ключевые файлы и их роль

| Файл                                     | Назначение                                                                                                                                                  |
|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `api/routes.py`                          | Все HTTP-роуты. Добавлять новые эндпоинты здесь.                                                                                                            |
| `api/ask_service.py`                     | Оркестрация RAG-пайплайна с CSV-логированием запросов.                                                                                                      |
| `api/ingest_service.py`                  | Логика обработки запросов на индексацию.                                                                                                                    |
| `rag/ingest.py`                          | Функции для индексации файлов и сайтов.                                                                                                                     |
| `rag/retrieval.py`                       | Функции для извлечения релевантного контекста (FAISS + BM25 + reranker).                                                                                    |
| `rag/data_store.py`                      | Слой персистентности: загрузка/сохранение индексов, rows.jsonl, BM25-корпус, перестройка индексов.                                                          |
| `clients/llm.py`                         | Абстрактный класс LLM. Сигнатуры: `generate(messages, max_tokens, temperature, context) -> LLMResp`, `generate_stream(...) -> Iterator[LLMResp]`.           |
| `clients/bitnet_client.py`               | Реализация BitNet. Два режима: `chat` и `completion`. Внутренние методы `_generate_chat*` / `_generate_completion*`.                                        |
| `clients/completion_chat_template.jinja` | Jinja2-шаблон для рендеринга промпта в completion-режиме.                                                                                                   |
| `schemas/app_schema.py`                  | Pydantic-схемы приложения: `Message`, `LLMReq`, `LLMResp`, `Timings`, `AskReq`, `AskResp`, `IngestFilesReq`, `IngestSiteReq`.                               |
| `schemas/bitnet_schema.py`               | Схемы ответов BitNet API: `ChatCompletionChunk`, `ChatCompletionResponse`, `CompletionResponse`, `CompletionStreamChunk`, `Usage`, `GenerationSettings`.    |
| `common/config.py`                       | Конфигурация из env. Добавлять новые переменные через `_env_str/_env_int` и т.д. Поддерживает `SYSTEM_PROMPT` и валидацию `BITNET_MODE`.                    |
| `common/logger.py`                       | Настройка логирования (`get_logger(name)`, `get_console_logger(name)`) + `CsvLogger` для записи CSV-логов с потокобезопасной вставкой/обновлением по ключу. |

## Правила написания кода

1. **Pydantic** — v2. Использовать `model_validate()` / `model_validate_json()`.
2. **Типизация** — обязательна. Использовать `list[...]`, `dict[...]`, `Optional`, `|` union syntax.
3. **Imports** — сначала стандартная библиотека, потом сторонние, потом внутренние.
