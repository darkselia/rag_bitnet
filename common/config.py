import math
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as e:
            raise ValueError(f"{name} must be an integer, got: {raw!r}") from e

    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got: {value}")
    return value


def _env_float(
        name: str,
        default: float,
        *,
        min_value: float | None = None,
        max_value: float | None = None,
) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as e:
            raise ValueError(f"{name} must be a float, got: {raw!r}") from e

    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got: {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got: {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default

    value = raw.strip().lower()
    truthy = {"1", "true", "yes", "y", "on"}
    falsy = {"0", "false", "no", "n", "off"}
    if value in truthy:
        return True
    if value in falsy:
        return False
    raise ValueError(f"{name} must be boolean-like (true/false/1/0), got: {raw!r}")


def _env_path(name: str, default: str) -> Path:
    value = Path(_env_str(name, default))
    if value.is_absolute():
        return value.resolve()
    return (PROJECT_ROOT / value).resolve()


DATA_DIR = _env_path("DATA_DIR", "./data/vector_store")
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = _env_path("LOG_DIR", "./logs")
LOG_MAX_BYTES = _env_int("LOG_MAX_BYTES", 5 * 1024 * 1024, min_value=1)
LOG_BACKUP_COUNT = _env_int("LOG_BACKUP_COUNT", 3, min_value=1)
LOG_LEVEL = _env_str("LOG_LEVEL", "DEBUG").upper()
if LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
INGEST_CHUNK_CHARS = _env_int("INGEST_CHUNK_CHARS", 1800, min_value=1)
INGEST_CHUNK_OVERLAP = _env_int("INGEST_CHUNK_OVERLAP", 200, min_value=0)
if INGEST_CHUNK_OVERLAP >= INGEST_CHUNK_CHARS:
    raise ValueError("INGEST_CHUNK_OVERLAP must be less than INGEST_CHUNK_CHARS")
SITE_TIMEOUT = _env_int("SITE_TIMEOUT", 10, min_value=1)
SITE_MIN_TEXT_CHARS = _env_int("SITE_MIN_TEXT_CHARS", 200, min_value=1)
INGEST_MAX_FILE_BYTES = _env_int("INGEST_MAX_FILE_BYTES", 25 * 1024 * 1024, min_value=1)
INGEST_MAX_BATCH_BYTES = _env_int("INGEST_MAX_BATCH_BYTES", 100 * 1024 * 1024, min_value=1)
INGEST_MAX_DOCX_UNCOMPRESSED_BYTES = _env_int(
    "INGEST_MAX_DOCX_UNCOMPRESSED_BYTES", 100 * 1024 * 1024, min_value=1,
)

EMBED_MODEL = _env_str("EMBED_MODEL", "intfloat/multilingual-e5-base")
RERANKER_MODEL = _env_str("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
USE_CROSS_RERANKER = _env_bool("USE_CROSS_RERANKER", True)
CROSS_RERANKER_TOPK = _env_int("CROSS_RERANKER_TOPK", 12, min_value=1)
EMBED_BATCH = _env_int("EMBED_BATCH", 64, min_value=1)
RERANKER_BATCH = _env_int("RERANKER_BATCH", 1, min_value=1)
RERANKER_QUANTIZE_CPU = _env_bool("RERANKER_QUANTIZE_CPU", True)
RERANKER_MIN_SCORE: float | None = (
    _env_float("RERANKER_MIN_SCORE", 0.0) if _env_str("RERANKER_MIN_SCORE", "") else None
)
if RERANKER_MIN_SCORE is not None:
    if not math.isfinite(RERANKER_MIN_SCORE):
        raise ValueError("RERANKER_MIN_SCORE must be finite")
    if not USE_CROSS_RERANKER:
        raise ValueError("RERANKER_MIN_SCORE requires USE_CROSS_RERANKER=true")

LLM_PROVIDER = _env_str("LLM_PROVIDER", "BITNET").lower()
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 300, min_value=1)
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.6, min_value=0.0, max_value=2.0)
SYSTEM_PROMPT: str = _env_str("SYSTEM_PROMPT", "You are a helpful AI assistant.")

_raw_bitnet_mode = _env_str("BITNET_MODE", "chat").lower()
if _raw_bitnet_mode not in ("chat", "completion"):
    raise ValueError(
        f"BITNET_MODE must be 'chat' or 'completion', got: {_raw_bitnet_mode!r}"
    )
BITNET_MODE: Literal["chat", "completion"] = _raw_bitnet_mode

BITNET_HOST = _env_str("BITNET_HOST", "127.0.0.1")
BITNET_PORT = _env_int("BITNET_PORT", 8080, min_value=1)
BITNET_TIMEOUT = _env_int("BITNET_TIMEOUT", 60, min_value=1)
BITNET_CONTEXT_SIZE = _env_int("BITNET_CONTEXT_SIZE", 0, min_value=0)
LOG_REQUEST_CONTENT = _env_bool("LOG_REQUEST_CONTENT", False)
RETRIEVAL_SIM_THRESHOLD = _env_float("RETRIEVAL_SIM_THRESHOLD", 0.0, min_value=-1.0, max_value=1.0)

API_HOST = _env_str("API_HOST", "127.0.0.1")
API_PORT = _env_int("API_PORT", 8000, min_value=1)
RAG_API_KEY = _env_str("RAG_API_KEY", "")

CORPUS_JSONL = DATA_DIR / "rows.jsonl"
EMB_NPY = DATA_DIR / "embeddings.npy"
FAISS_INDEX = DATA_DIR / "index.faiss"
BM25_JSON = DATA_DIR / "bm25_corpus.json"
META_JSON = DATA_DIR / "meta.json"


def meta_payload(count: int) -> dict:
    return {
        "count": count,
        "updated_at": time.time(),
        "embed_model": EMBED_MODEL,
        "reranker": RERANKER_MODEL,
    }
