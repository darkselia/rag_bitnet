from __future__ import annotations

from typing import Any

import requests
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from common import config
from common.logger import get_logger

logger = get_logger("lm_eval_adapter")

_DEFAULT_MAX_TOKENS = 512


class BitNetLM(LM):
    def __init__(
            self,
            mode: str | None = None,
            host: str | None = None,
            port: int | None = None,
            timeout: int | None = None,
            batch_size: int = 1,
    ):
        super().__init__()
        self._mode = (mode or config.BITNET_MODE).lower()
        self._host = host or config.BITNET_HOST
        self._port = port or config.BITNET_PORT
        self._timeout = timeout or config.BITNET_TIMEOUT
        self._batch_size = batch_size
        self._base_url = f"http://{self._host}:{self._port}"
        self.model_name: str | None = None
        logger.info(
            "BitNetLM: mode=%s host=%s port=%d timeout=%d",
            self._mode, self._host, self._port, self._timeout,
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def _completion_url(self) -> str:
        if self._mode == "chat":
            return f"{self._base_url}/v1/chat/completions"
        return f"{self._base_url}/completion"

    def _build_payload(self, context: str, gen_kwargs: dict) -> dict[str, Any]:
        n_predict = (
                gen_kwargs.get("max_gen_toks")
                or gen_kwargs.get("max_gen_tokens")
                or gen_kwargs.get("max_length")
                or _DEFAULT_MAX_TOKENS
        )
        temperature = gen_kwargs.get("temperature", config.LLM_TEMPERATURE)

        do_sample = gen_kwargs.get("do_sample", True)
        if not do_sample:
            temperature = 0.0

        until = gen_kwargs.get("until", [])
        if isinstance(until, str):
            until = [until]
        until = [s for s in until if s]

        if self._mode == "completion":
            return {
                "prompt": context,
                "n_predict": n_predict,
                "temperature": temperature,
                "stop": until,
                "stream": False,
            }

        return {
            "messages": [{"role": "user", "content": context}],
            "max_tokens": n_predict,
            "temperature": temperature,
            "stop": until,
            "stream": False,
        }

    def _call_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = requests.post(self._completion_url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        model = data.get("model")
        if model:
            if self.model_name and self.model_name != model:
                raise ValueError("Server model changed during the benchmark")
            self.model_name = model
        return data

    def model_identity(self) -> dict[str, Any]:
        try:
            response = requests.get(f"{self._base_url}/v1/models", timeout=self._timeout)
            response.raise_for_status()
            models = response.json()["data"]
            ids = [item["id"] for item in models if isinstance(item, dict) and isinstance(item.get("id"), str)]
            actual = ids[0] if len(ids) == 1 else None
        except (requests.RequestException, ValueError, KeyError, TypeError):
            actual = None
        return {"server_model": actual, "response_model": self.model_name,
                "server_model_verified": actual is not None and actual == self.model_name}

    def _extract_text(self, data: dict[str, Any]) -> str:
        if self._mode == "completion":
            if data.get("truncated"):
                raise ValueError("Server truncated the benchmark prompt")
            return data["content"]
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Server returned no choices")
        return choices[0]["message"]["content"]

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results: list[str] = []
        logger.info("generate_until: %d requests", len(requests))
        for i, req in enumerate(requests):
            context, gen_kwargs = req.args
            payload = self._build_payload(context, gen_kwargs)
            logger.debug(
                "  [%d/%d] context_len=%d gen_kwargs=%s",
                i + 1, len(requests), len(context), gen_kwargs,
            )
            data = self._call_api(payload)
            text = self._extract_text(data)
            if not isinstance(text, str) or not text:
                raise ValueError(f"Empty response for benchmark request {i + 1}")
            results.append(text)
        return results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("True per-token loglikelihood is not supported by this adapter")

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "BitNet HTTP API does not expose per-token log probabilities. "
            "loglikelihood_rolling (perplexity) is not supported."
        )
