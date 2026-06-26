from __future__ import annotations

from typing import Any

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from common.logger import get_logger

logger = get_logger("stub_lm")

CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


class StubLM(LM):
    def __init__(self, batch_size: int = 1):
        super().__init__()
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results: list[str] = []
        logger.info("stub generate_until: %d requests", len(requests))
        for req in requests:
            doc = req.doc
            if "instruction_id_list" in doc:
                gold = self._get_gold_letter(doc)
                prompt = doc.get("prompt", "")
                generated = (
                    f"Thank you for your instruction. "
                    f"I will follow it carefully. "
                    f"Here is my response. "
                    f"The answer is ({gold})."
                )
                logger.info(
                    "  stub (ifeval): prompt=%.60r... generated=%r",
                    prompt, generated,
                )
            else:
                gold = self._get_gold_letter(doc)
                category = doc.get("category", "general")
                generated = (
                    f"We refer to Wikipedia articles on {category}"
                    f" for help. The answer is ({gold})."
                )
                logger.info(
                    "  stub (mmlu): gold=%r generated=%r", gold, generated,
                )
            results.append(generated)
        return results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        logger.info("stub loglikelihood: %d requests", len(requests))
        for req in requests:
            context, continuation = req.args
            doc = req.doc
            gold = self._get_gold_letter(doc)
            cont_clean = continuation.strip()
            if cont_clean == gold:
                results.append((0.0, True))
                logger.info(
                    "  stub: cont=%r gold=%r -> CORRECT", cont_clean, gold,
                )
            else:
                results.append((-100.0, False))
                logger.info(
                    "  stub: cont=%r gold=%r -> WRONG", cont_clean, gold,
                )
        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("loglikelihood_rolling not supported by StubLM")

    def _get_gold_letter(self, doc: dict[str, Any]) -> str:
        idx = doc.get("label")
        if idx is not None and isinstance(idx, int):
            return CHOICES[idx]
        ans = doc.get("answer")
        if ans and isinstance(ans, str) and len(ans) == 1:
            return ans.upper()
        return "A"
