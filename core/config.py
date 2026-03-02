import os
import time

from dotenv import load_dotenv

load_dotenv()


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


BITNET_HOST = _env_str("BITNET_HOST", "127.0.0.1")
BITNET_PORT = _env_int("BITNET_PORT", 8080, min_value=1)
BITNET_MAX_TOKENS = _env_int("BITNET_MAX_TOKENS", 300, min_value=1)
BITNET_TEMPERATURE = _env_float("BITNET_TEMPERATURE", 0.6, min_value=0.0, max_value=2.0)
BITNET_TIMEOUT = _env_int("BITNET_TIMEOUT", 60000, min_value=1)

API_HOST = _env_str("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8000, min_value=1)


def meta_payload(count: int) -> dict:
    return {
        "count": count,
        "updated_at": time.time(),
    }
