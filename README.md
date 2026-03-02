# RAG BitNet (FastAPI)

## Запуск локально
```bash
cp .env.example .env
python main.py
```

`core/config.py` автоматически читает `.env` (через `python-dotenv`) и применяет `os.getenv` с дефолтами и проверкой типов.

## Docker (API + BitNet в одном контейнере)
Собрать образ:
```bash
docker build -f docker/Dockerfile -t rag-bitnet:latest .
```

Запустить контейнер с внешней папкой BitNet:
```bash
docker run --rm -p 8000:8000 -p 8080:8080 \
  -v "$PWD/data:/app/data" \
  -v "/absolute/path/to/BitNet:/opt/bitnet" \
  -e BITNET_DIR=/opt/bitnet \
  -e BITNET_MODEL=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -e BITNET_THREADS=6 \
  rag-bitnet:latest
```

Если у вас скрипт/модель лежат в других путях внутри BitNet-папки, передайте:
- `BITNET_SERVER_SCRIPT` (по умолчанию `run_inference_server.py`)
- `BITNET_MODEL` (путь относительно `BITNET_DIR`)

## Основные env-переменные
- `API_HOST`, `API_PORT` - адрес и порт FastAPI.
- `BITNET_HOST`, `BITNET_PORT` - где доступен BitNet inference server.
- `BITNET_MAX_TOKENS`, `BITNET_TEMPERATURE`, `BITNET_TIMEOUT` - параметры генерации и тайм-аут запроса.

## Структура
- `rag_bitnet/config.py` — env/пути и параметры моделей.
- `rag_bitnet/clients/bitnet_client.py` — клиент BitNet API.
- `rag_bitnet/prompting/prompt.py` — сборка промпта.
- `rag_bitnet/api/*` и `rag_bitnet/server.py` — FastAPI

