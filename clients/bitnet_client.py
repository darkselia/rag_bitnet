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
            break
    return result


def _compute_chat_timings(usage, elapsed_ms: float) -> Timings:
    return Timings(
        prompt_n=usage.prompt_tokens,
        predicted_n=usage.completion_tokens,
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

    def count_prompt_tokens(self, messages: list[Message], context: str) -> int:
        prepared = _build_user_message(messages, context)
        if self.mode == "chat":
            response = requests.post(
                f"{self.base}/apply-template",
                json={"messages": [m.model_dump() for m in prepared]}, timeout=self.timeout,
            )
            response.raise_for_status()
            prompt = response.json()["prompt"]
        else:
            prompt = _render_prompt(prepared)
        response = requests.post(
            f"{self.base}/tokenize", json={"content": prompt, "add_special": True}, timeout=self.timeout,
        )
        response.raise_for_status()
        tokens = response.json()["tokens"]
        if not isinstance(tokens, list):
            raise ValueError("Invalid tokenizer response")
        return len(tokens)

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
        if data.truncated:
            raise ValueError("BitNet truncated the input prompt")
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
        finished = False
        with requests.post(self._url, json=payload, timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            resp.encoding = "utf-8"
            for raw_line in resp.iter_lines(decode_unicode=False):
                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8")
                else:
                    line = raw_line
                if not line or line.startswith((":", "event:", "id:", "retry:")):
                    continue
                data_part = line[len("data:"):].strip() if line.startswith("data:") else line.strip()
                if data_part == "[DONE]":
                    break
                if not data_part:
                    continue
                chunk = ChatCompletionChunk.model_validate_json(data_part)
                if chunk.usage:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content or ""
                if piece:
                    content_pieces.append(piece)
                    yield LLMResp(id=chunk.id, content=piece, model=chunk.model or "")
                if chunk.model:
                    model = chunk.model
                if chunk.id:
                    response_id = chunk.id
                if chunk.choices[0].finish_reason:
                    finished = True
        if not finished:
            raise ValueError("BitNet stream ended without a finish event")
        elapsed_ms = (time.monotonic() - start) * 1000
        timings = _compute_chat_timings(usage, elapsed_ms) if usage else None
        yield LLMResp(id=response_id, content="", timings=timings, model=model, final=True)

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
                    line = raw_line.decode("utf-8")
                else:
                    line = raw_line
                if not line or line.startswith((":", "event:", "id:", "retry:")):
                    continue
                data_part = line[len("data:"):].strip() if line.startswith("data:") else line.strip()
                if not data_part:
                    continue
                chunk = CompletionStreamChunk.model_validate_json(data_part)
                if chunk.content:
                    content_pieces.append(chunk.content)
                    yield LLMResp(id=str(chunk.index), content=chunk.content)
                if chunk.stop:
                    final_data = data_part
                    break
        if not final_data:
            raise ValueError("BitNet stream ended without a finish event")
        full = CompletionResponse.model_validate_json(final_data)
        if full.truncated:
            raise ValueError("BitNet truncated the input prompt")
        yield LLMResp(id=str(full.index), content="", timings=full.timings, model=full.model, final=True)


bitnet_client = BitNetAPI()
