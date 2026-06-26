from __future__ import annotations

import re
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
        return resp.json()

    def _extract_text(self, data: dict[str, Any]) -> str:
        if self._mode == "completion":
            return data.get("content", "")
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

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
            try:
                data = self._call_api(payload)
                text = self._extract_text(data)
            except requests.exceptions.ConnectionError:
                logger.error("Cannot connect to BitNet server at %s", self._base_url)
                text = ""
            except Exception as exc:
                logger.warning("  request %d failed: %s", i + 1, exc)
                text = ""
            text = re.sub(r"\s+", " ", text).strip()
            results.append(text)
        return results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        _gen_cache: dict[str, str] = {}
        results: list[tuple[float, bool]] = []
        logger.info("loglikelihood: %d requests", len(requests))

        for req in requests:
            context, continuation = req.args

            if context not in _gen_cache:
                payload = self._build_payload(context, {
                    "max_gen_toks": 10,
                    "do_sample": False,
                    "temperature": 0.0,
                })
                try:
                    data = self._call_api(payload)
                    _gen_cache[context] = self._extract_text(data).strip()
                except Exception as exc:
                    logger.warning("loglikelihood generate failed: %s", exc)
                    _gen_cache[context] = ""

            gen_text = _gen_cache[context]
            cont_clean = continuation.strip()

            if gen_text.startswith(cont_clean):
                results.append((0.0, True))
            else:
                results.append((-100.0, False))

            logger.info(
                "loglikelihood: continuation=%r generated=%r match=%s",
                cont_clean, gen_text, gen_text.startswith(cont_clean),
            )

        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "BitNet HTTP API does not expose per-token log probabilities. "
            "loglikelihood_rolling (perplexity) is not supported."
        )
