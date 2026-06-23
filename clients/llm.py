from abc import ABC, abstractmethod
from collections.abc import Iterator

from common import config
from schemas.app_schema import LLMResp, Message


class LLM(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate(
            self,
            messages: list[Message],
            max_tokens: int | None = None,
            temperature: float | None = None,
            context: str = "",
    ) -> LLMResp:
        pass

    @abstractmethod
    def generate_stream(
            self,
            messages: list[Message],
            max_tokens: int | None = None,
            temperature: float | None = None,
            context: str = "",
    ) -> Iterator[LLMResp]:
        pass


_llm: LLM | None = None


def get_llm() -> LLM | None:
    global _llm
    if _llm is None:
        if config.LLM_PROVIDER == "bitnet":
            from clients.bitnet_client import BitNetAPI

            _llm = BitNetAPI()
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER!r}")
    return _llm
