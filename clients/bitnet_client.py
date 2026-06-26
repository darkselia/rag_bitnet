import time
from collections.abc import Iterator
from pathlib import Path

import jinja2
import requests

from clients.llm import LLM
from common import config
from common.logger import get_logger
from schemas.app_schema import LLMResp, Message, Timings
from schemas.bitnet_schema import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    CompletionResponse,
    CompletionStreamChunk,
)

_HERE = Path(__file__).resolve().parent

_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_HERE)),
    autoescape=False,
)
_completion_template = _env.get_template("completion_chat_template.jinja")

logger = get_logger("bitnet_client")


def _render_prompt(messages: list[Message]) -> str:
    dicts = [m.model_dump() for m in messages]
    result = _completion_template.render(messages=dicts, add_generation_prompt=True)
    logger.info(f"build_user_message: {result}")
    return result


def _build_user_message(messages: list[Message], context: str) -> list[Message]:
    result = list(messages)
    for i in range(len(result) - 1, -1, -1):
        if result[i].role == "user":
            question = result[i].content
            result[i] = Message(
                role="user",
                content=(
                    "Context:\n"
                    f"{context}\n\n"
                    f"Question: {question}\n"
                ),
            )
            logger.info(f"build_user_message: {result[i]}")
            break
    return result


def _compute_chat_timings(usage, elapsed_ms: float) -> Timings:
    if usage is None:
        return Timings(
            prompt_n=0,
            prompt_ms=0,
            prompt_per_token_ms=0,
            prompt_per_second=0,
            predicted_n=0,
            predicted_ms=0,
            predicted_per_token_ms=0,
            predicted_per_second=0,
        )
    prompt_n = usage.prompt_tokens
    predicted_n = usage.completion_tokens
    total_n = prompt_n + predicted_n
    if total_n > 0:
        prompt_ms = elapsed_ms * (prompt_n / total_n)
        predicted_ms = elapsed_ms * (predicted_n / total_n)
    else:
        prompt_ms = elapsed_ms / 2
        predicted_ms = elapsed_ms / 2
    prompt_per_token_ms = prompt_ms / prompt_n if prompt_n > 0 else 0.0
    predicted_per_token_ms = predicted_ms / predicted_n if predicted_n > 0 else 0.0
    return Timings(
        prompt_n=prompt_n,
        prompt_ms=prompt_ms,
        prompt_per_token_ms=prompt_per_token_ms,
        prompt_per_second=1000.0 / prompt_per_token_ms if prompt_per_token_ms > 0 else 0.0,
        predicted_n=predicted_n,
        predicted_ms=predicted_ms,
        predicted_per_token_ms=predicted_per_token_ms,
        predicted_per_second=1000.0 / predicted_per_token_ms if predicted_per_token_ms > 0 else 0.0,
    )


class BitNetAPI(LLM):
    def __init__(self, mode: str | None = None):
        self.host = config.BITNET_HOST
        self.port = config.BITNET_PORT
        self.base = f"http://{self.host}:{self.port}"
        self.timeout = config.BITNET_TIMEOUT
        self.mode = mode if mode is not None else config.BITNET_MODE

    @property
    def _url(self) -> str:
        if self.mode == "chat":
            return f"{self.base}/v1/chat/completions"
        return f"{self.base}/completion"

    def generate(
            self,
            messages: list[Message],
            max_tokens: int | None = None,
            temperature: float | None = None,
            context: str = "",
    ) -> LLMResp:
        if self.mode == "chat":
            return self._generate_chat(messages, max_tokens, temperature, context)
        return self._generate_completion(messages, max_tokens, temperature, context)

    def generate_stream(
            self,
            messages: list[Message],
            max_tokens: int | None = None,
            temperature: float | None = None,
            context: str = "",
    ) -> Iterator[LLMResp]:
        if self.mode == "chat":
            yield from self._generate_chat_stream(messages, max_tokens, temperature, context)
        else:
            yield from self._generate_completion_stream(messages, max_tokens, temperature, context)

    def _generate_chat(
            self,
            messages: list[Message],
            max_tokens: int | None = None,
            temperature: float | None = None,
            context: str = "",
    ) -> LLMResp:
        messages = _build_user_message(messages, context)
        payload = {
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
            "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
            "stream": False,
        }
        start = time.monotonic()
        resp = requests.post(self._url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        elapsed_ms = (time.monotonic() - start) * 1000
        data = ChatCompletionResponse.model_validate(resp.json())
        timings = _compute_chat_timings(data.usage, elapsed_ms) if data.usage else None
        return LLMResp(
            id=data.id,
            content=data.choices[0].message.content,
            timings=timings,
            model=data.model,
        )

    def _generate_completion(
            self,
            messages: list[Message],
            max_tokens: int | None = None,
            temperature: float | None = None,
            context: str = "",
    ) -> LLMResp:
        messages = _build_user_message(messages, context)
        prompt = _render_prompt(messages)
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
            "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
            "stream": False,
            "stop": ["<|eot_id|>", "<|start_header_id|>", "\nQuestion:", "\nContext:"],
        }
        resp = requests.post(self._url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = CompletionResponse.model_validate(resp.json())
        timings = Timings.model_validate(data.timings) if data.timings else None
        return LLMResp(
            id=str(data.index),
            content=data.content,
            timings=timings,
            model=data.model,
        )

    def _generate_chat_stream(
            self,
            messages: list[Message],
            max_tokens: int | None = None,
            temperature: float | None = None,
            context: str = "",
    ) -> Iterator[LLMResp]:
        messages = _build_user_message(messages, context)
        payload = {
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
            "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
            "stream": True,
        }
        start = time.monotonic()
        content_pieces: list[str] = []
        response_id = ""
        usage = None
        model = ""
        with requests.post(self._url, json=payload, timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            resp.encoding = "utf-8"
            for raw_line in resp.iter_lines(decode_unicode=False):
                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8", errors="replace")
                else:
                    line = raw_line
                if not line:
                    continue
                data_part = line[len("data:"):].strip() if line.startswith("data:") else line.strip()
                if not data_part or data_part == "[DONE]":
                    continue
                try:
                    chunk = ChatCompletionChunk.model_validate_json(data_part)
                except Exception:
                    continue
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content or ""
                if piece:
                    content_pieces.append(piece)
                    yield LLMResp(id=chunk.id, content=piece, model=chunk.model or "")
                if chunk.model:
                    model = chunk.model
                if chunk.usage:
                    usage = chunk.usage
                if chunk.id:
                    response_id = chunk.id
                if chunk.choices[0].finish_reason:
                    break
        elapsed_ms = (time.monotonic() - start) * 1000
        timings = _compute_chat_timings(usage, elapsed_ms) if usage else None
        yield LLMResp(id=response_id, content="".join(content_pieces), timings=timings, model=model)

    def _generate_completion_stream(
            self,
            messages: list[Message],
            max_tokens: int | None = None,
            temperature: float | None = None,
            context: str = "",
    ) -> Iterator[LLMResp]:
        messages = _build_user_message(messages, context)
        prompt = _render_prompt(messages)
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
            "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
            "stream": True,
            "stop": ["<|eot_id|>", "<|start_header_id|>", "\nQuestion:", "\nContext:"],
        }
        content_pieces: list[str] = []
        final_data = None
        model = ""
        with requests.post(self._url, json=payload, timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            resp.encoding = "utf-8"
            for raw_line in resp.iter_lines(decode_unicode=False):
                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8", errors="replace")
                else:
                    line = raw_line
                if not line:
                    continue
                data_part = line[len("data:"):].strip() if line.startswith("data:") else line.strip()
                if not data_part:
                    continue
                try:
                    chunk = CompletionStreamChunk.model_validate_json(data_part)
                except Exception:
                    continue
                if chunk.content:
                    content_pieces.append(chunk.content)
                    yield LLMResp(id=str(chunk.index), content=chunk.content)
                if chunk.stop:
                    final_data = data_part
                    break
        if final_data and content_pieces:
            try:
                full = CompletionResponse.model_validate_json(final_data)
                timings = full.timings if full.timings else None
                model = full.model
                yield LLMResp(id=str(full.index), content="".join(content_pieces), timings=timings, model=model)
            except Exception:
                yield LLMResp(content="".join(content_pieces))
        elif content_pieces:
            yield LLMResp(content="".join(content_pieces))


bitnet_client = BitNetAPI()
