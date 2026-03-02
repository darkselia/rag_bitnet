import json
import requests
from collections.abc import Iterator

from core import config


class BitNetAPI:
    def __init__(self, host: str | None = None, port: int | None = None, timeout: int | None = None):
        self.host = host or config.BITNET_HOST
        self.port = port or config.BITNET_PORT
        self.base = f"http://{self.host}:{self.port}"
        self.url = f"{self.base}/completion"
        self.timeout = timeout or config.BITNET_TIMEOUT

    def generate(self, prompt: str, max_tokens: int | None = 200, temperature: float | None = 0.6) -> str:
        chunks = list(self.generate_stream(prompt, max_tokens, temperature))
        if chunks:
            return "".join(chunks)
        return ""

    def generate_stream(self, prompt: str, max_tokens: int | None = config.BITNET_MAX_TOKENS, temperature: float | None = config.BITNET_TEMPERATURE) -> Iterator[str]:
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        with requests.post(self.url, json=payload, timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()

            # Stream chunks (SSE-like) and collect content pieces
            chunks_count = 0
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                data_part = line[len("data:"):].strip() if line.startswith("data:") else line.strip()
                if not data_part:
                    continue
                try:
                    payload_obj = json.loads(data_part)
                except json.JSONDecodeError:
                    continue
                piece = payload_obj.get("content", "")
                if piece:
                    chunks_count += 1
                    yield piece
                if payload_obj.get("stop"):
                    break

            if chunks_count:
                return

            # Fallback to non-streaming JSON payload
            try:
                yield str(resp.json())
            except Exception:
                yield resp.text


bitnet_client = BitNetAPI()
