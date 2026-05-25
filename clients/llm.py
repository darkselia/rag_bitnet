from abc import ABC, abstractmethod
from collections.abc import Iterator

from common import config


class LLM(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
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
