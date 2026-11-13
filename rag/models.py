import threading
from typing import Optional

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from common import config
from common.logger import get_console_logger

logger = get_console_logger("models")
torch.set_num_threads(config.CPU_THREADS)

_embed_model: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None
_device = "cpu"  # adjust if GPU available
_embed_lock = threading.Lock()
_reranker_lock = threading.Lock()


def get_embed_model() -> SentenceTransformer | None:
    global _embed_model
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                logger.info("[init] loading embed model %s on device %s", config.EMBED_MODEL, _device)
                _embed_model = SentenceTransformer(config.EMBED_MODEL, device=_device)
    return _embed_model


def get_reranker() -> Optional[CrossEncoder]:
    global _reranker
    if not config.USE_CROSS_RERANKER:
        return None
    with _reranker_lock:
        if _reranker is None:
            logger.info("[init] loading cross-encoder reranker %s on device %s", config.RERANKER_MODEL, _device)
            model = CrossEncoder(config.RERANKER_MODEL, device=_device)
            if config.RERANKER_QUANTIZE_CPU and _device == "cpu":
                try:
                    torch.ao.quantization.quantize_dynamic(model.model, {torch.nn.Linear}, dtype=torch.qint8,
                                                           inplace=True)
                    logger.info("[init] CPU reranker quantization enabled")
                except (RuntimeError, NotImplementedError) as e:
                    logger.warning("[init] reranker quantization failed; loading original model: %s", type(e).__name__)
                    model = None
            if model is None:
                model = CrossEncoder(config.RERANKER_MODEL, device=_device)
            _reranker = model
    return _reranker

